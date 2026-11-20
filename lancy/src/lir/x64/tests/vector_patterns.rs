use crate::ScalarKernel;
use crate::hir::{Binary, Function, Lane, Op, Simd, Terminator, Type, Value, VectorAddress};
use crate::lir::x64::{self, Features};
use iced_x86::{Decoder, DecoderOptions, Mnemonic};

fn function(params: &[Type], result: Type, make: impl FnOnce(&[Value]) -> Simd) -> Function {
    let mut f = Function::new(params, result);
    let block = f.entry();
    let op = make(f.params(block));
    let result = f.append(block, Op::Simd(op)).unwrap();
    f.set_terminator(block, Terminator::Return(result));
    f
}

fn instructions(f: &Function) -> Vec<Mnemonic> {
    let (module, ids) = f.verified_module().unwrap();
    let code = x64::emit(module, &ids).unwrap();
    Decoder::new(64, &code.bytes[..code.text_len], DecoderOptions::NONE)
        .iter()
        .map(|i| i.mnemonic())
        .collect()
}

// Public word ABI: narrow scalar results occupy one word and every vector
// uses the full host-selected width. Preserve raw FP bits and lane ordering.
fn words(lane: Lane, values: &[u128]) -> Vec<u64> {
    let size = lane.bytes();
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes()[..size]);
    }
    bytes.resize(bytes.len().next_multiple_of(8), 0);
    bytes
        .chunks_exact(8)
        .map(|p| u64::from_le_bytes(p.try_into().unwrap()))
        .collect()
}

fn check(k: &ScalarKernel, args: &[u64], expected: &[u64]) {
    let mut result = vec![0; expected.len()];
    k.call_words(args, &mut result).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn two_table_shuffle_checks_the_entire_unsigned_index() {
    for (lane, idx) in [
        (Lane::U8, Lane::I8),
        (Lane::U16, Lane::U16),
        (Lane::F32, Lane::I32),
        (Lane::F64, Lane::U64),
        (Lane::U128, Lane::I128),
    ] {
        let ty = Type::Vector(lane);
        let f = function(&[ty, ty, Type::Vector(idx)], ty, |v| {
            Simd::Shuffle(v[0], v[1], v[2])
        });
        let k = ScalarKernel::compile(&f).unwrap();
        let n = ScalarKernel::vector_bytes() / lane.bytes();
        let tables: Vec<_> = (0..2 * n)
            .map(|i| (i + 1) as u128 * 0x01_01_01_01u128)
            .collect();
        let limit = (2 * n) as u128;
        let indices = [
            0,
            n as u128 - 1,
            n as u128,
            limit - 1,
            limit,
            limit + 1,
            1 << (8 * idx.bytes() - 1),
            u128::MAX >> (128 - 8 * idx.bytes()),
        ];
        for phase in 0..indices.len() {
            let ix: Vec<_> = (0..n)
                .map(|i| indices[(phase + i) % indices.len()])
                .collect();
            let want: Vec<_> = ix
                .iter()
                .map(|&i| if i < limit { tables[i as usize] } else { 0 })
                .collect();
            let mut args = words(lane, &tables);
            args.extend(words(idx, &ix));
            check(&k, &args, &words(lane, &want));
        }
    }
}

#[test]
fn ordered_fp_reduce_preserves_rounding_and_the_first_lanes_sign() {
    for lane in [Lane::F32, Lane::F64] {
        let ty = Type::Vector(lane);
        let f = function(&[ty], lane.scalar(), |v| Simd::Reduce(Binary::Add, v[0]));
        let kernel = ScalarKernel::compile(&f).unwrap();
        let n = ScalarKernel::vector_bytes() / lane.bytes();
        let bits = |x: f64| {
            if lane == Lane::F32 {
                u128::from((x as f32).to_bits())
            } else {
                u128::from(x.to_bits())
            }
        };
        check(
            &kernel,
            &words(lane, &vec![bits(-0.0); n]),
            &words(lane, &[bits(-0.0)]),
        );
        if n >= 4 {
            // ((huge + 1) - huge) + 3 == 3, whereas pairing adjacent sums gives 0.
            let mut values = vec![0; n];
            values[..4].copy_from_slice(&[bits(1e30), bits(1.), bits(-1e30), bits(3.)]);
            check(&kernel, &words(lane, &values), &words(lane, &[bits(3.)]));
        }
    }
}

#[test]
fn integer_reductions_include_every_chunk_and_wrap_at_lane_width() {
    for lane in [Lane::I16, Lane::I32, Lane::I64] {
        let ty = Type::Vector(lane);
        let n = ScalarKernel::vector_bytes() / lane.bytes();
        let mask = u128::MAX >> (128 - lane.bytes() * 8);
        let sign = 1 << (8 * lane.bytes() - 1);
        // Odd multiplicands keep every chunk observable even after overflow.
        let numbers: Vec<_> = (0..n)
            .map(|i| (if i % 2 == 0 { 0 } else { sign }) | (2 * i as u128 + 3))
            .collect();
        for op in [
            Binary::Add,
            Binary::Mul,
            Binary::And,
            Binary::Or,
            Binary::Xor,
            Binary::Min,
            Binary::Max,
        ] {
            let f = function(&[ty], lane.scalar(), |v| Simd::Reduce(op, v[0]));
            let k = ScalarKernel::compile(&f).unwrap();
            let answer = numbers
                .iter()
                .copied()
                .reduce(|a, b| match op {
                    Binary::Add => a.wrapping_add(b) & mask,
                    Binary::Mul => a.wrapping_mul(b) & mask,
                    Binary::And => a & b,
                    Binary::Or => a | b,
                    Binary::Xor => a ^ b,
                    Binary::Min => ((a ^ sign).min(b ^ sign)) ^ sign,
                    Binary::Max => ((a ^ sign).max(b ^ sign)) ^ sign,
                    _ => unreachable!(),
                })
                .unwrap();
            check(&k, &words(lane, &numbers), &words(lane, &[answer]));
        }
    }
}

#[test]
fn widen_uses_source_signedness_and_narrow_concatenates_truncated_lanes() {
    for (from, to) in [
        (Lane::I8, Lane::U16),
        (Lane::U8, Lane::I16),
        (Lane::I16, Lane::U32),
        (Lane::U16, Lane::I32),
        (Lane::I32, Lane::U64),
        (Lane::U32, Lane::I64),
    ] {
        let width = ScalarKernel::vector_bytes();
        let n = width / from.bytes();
        let bits = 8 * from.bytes();
        let mask = (1u128 << bits) - 1;
        let a: Vec<_> = (0..n)
            .map(|i| (i as u128 * 73 + (1 << (bits - 1))) & mask)
            .collect();
        for high in [false, true] {
            let f = function(&[Type::Vector(from)], Type::Vector(to), |v| Simd::Widen {
                to,
                high,
                value: v[0],
            });
            let k = ScalarKernel::compile(&f).unwrap();
            let start = if high { n / 2 } else { 0 };
            let expected: Vec<_> = a[start..start + n / 2]
                .iter()
                .map(|&x| {
                    if from.signed() && x & (1 << (bits - 1)) != 0 {
                        x | !mask
                    } else {
                        x
                    }
                })
                .collect();
            check(&k, &words(from, &a), &words(to, &expected));
        }
        let f = function(&[Type::Vector(to); 2], Type::Vector(from), |v| {
            Simd::Narrow(from, v[0], v[1])
        });
        let k = ScalarKernel::compile(&f).unwrap();
        let wide: Vec<_> = (0..n)
            .map(|i| {
                let low = (i as u128 * 17 + 3) & mask;
                // Both signed and unsigned saturating packs disagree with truncation.
                if i % 2 == 0 {
                    low | (1 << bits)
                } else {
                    low | (!mask)
                }
            })
            .collect();
        check(&k, &words(to, &wide), &words(from, &wide));
    }
}

#[test]
fn indexed_memory_can_reuse_the_mask_as_indices_and_passthrough() {
    for lane in [Lane::I32, Lane::I64] {
        let ty = Type::Vector(lane);
        let mut f = Function::new(&[Type::Ptr, ty, ty], ty);
        let b = f.entry();
        let [base, both, values] = f.params(b).try_into().unwrap();
        let address = VectorAddress {
            base,
            indices: Some(both),
        };
        let loaded = f
            .append(
                b,
                Op::Simd(Simd::MaskedLoad {
                    address,
                    mask: both,
                    passthrough: both,
                }),
            )
            .unwrap();
        f.append(
            b,
            Op::Simd(Simd::MaskedStore {
                address,
                mask: both,
                value: values,
            }),
        );
        let r = f
            .append(b, Op::Simd(Simd::Bin(Binary::Xor, loaded, both)))
            .unwrap();
        f.set_terminator(b, Terminator::Return(r));
        let k = ScalarKernel::compile(&f).unwrap();
        let bytes = lane.bytes();
        let n = ScalarKernel::vector_bytes() / bytes;
        // Only negative indices are active. All active lanes address one of
        // the two initialized elements immediately preceding the base.
        let indices: Vec<_> = (0..n)
            .map(|i| match i % 4 {
                0 | 3 => -1i64,
                1 => i32::MAX as i64,
                _ => -2,
            })
            .collect();
        let idx_bits: Vec<_> = indices.iter().map(|&x| x as u128).collect();
        let replacements: Vec<_> = (0..n).map(|i| 0x4667_1900 + i as u128).collect();
        let original = [0x1234_5555u128, 0x2300_5678, 0xdead, 0xbeef];
        let mut expected = original;
        let mut expected_output = Vec::new();
        for (&i, &new) in indices.iter().zip(&replacements) {
            if i < 0 {
                let j = (2 + i) as usize;
                expected_output.push(original[j] ^ i as u128);
                expected[j] = new;
            } else {
                expected_output.push(0);
            }
        }
        let mut memory = words(lane, &original);
        let base = memory.as_mut_ptr().cast::<u8>().wrapping_add(2 * bytes);
        let mut args = vec![base as u64];
        args.extend(words(lane, &idx_bits));
        args.extend(words(lane, &replacements));
        let mut out = vec![0; ScalarKernel::vector_bytes() / 8];
        // SAFETY: the kernel and allocation remain live. All active accesses target
        // the first two elements of this exclusively borrowed allocation.
        unsafe { k.call_words_raw(&args, &mut out) }.unwrap();
        assert_eq!(out, words(lane, &expected_output));
        assert_eq!(memory, words(lane, &expected));

        let cpu = Features::host();
        let evex = cpu.bytes == 64 || cpu.avx10 >= 1;
        if evex || cpu.bytes == 32 && cpu.avx2 {
            let asm = instructions(&f);
            assert!(asm.contains(&if bytes == 4 {
                Mnemonic::Vpgatherdd
            } else {
                Mnemonic::Vpgatherqq
            }));
            if evex {
                assert!(asm.contains(&if bytes == 4 {
                    Mnemonic::Vpscatterdd
                } else {
                    Mnemonic::Vpscatterqq
                }));
            }
        }
    }
}

#[test]
fn avx512_selects_packed_permutation_reduction_and_integer_conversions() {
    use Mnemonic::*;
    if Features::host().bytes != 64 {
        return;
    }
    let i = Type::Vector(Lane::I32);
    let w = Type::Vector(Lane::I16);
    let check = |f, allowed: &[Mnemonic]| {
        let asm = instructions(&f);
        assert!(
            asm.iter().any(|mn| allowed.contains(mn)),
            "expected {allowed:?} in {asm:?}"
        );
    };
    check(
        function(&[i; 3], i, |v| Simd::Shuffle(v[0], v[1], v[2])),
        &[Vpermi2d, Vpermt2d],
    );
    check(
        function(&[i], Type::I32, |v| Simd::Reduce(Binary::Add, v[0])),
        &[Paddd, Vpaddd, Phaddd, Vphaddd],
    );
    check(
        function(&[w], i, |v| Simd::Widen {
            to: Lane::I32,
            high: true,
            value: v[0],
        }),
        &[Vpmovsxwd],
    );
    check(
        function(&[i; 2], w, |v| Simd::Narrow(Lane::I16, v[0], v[1])),
        &[Vpmovdw, Vpackusdw, Vpackssdw],
    );
}

#[test]
fn selected_rearrangements_respect_feature_subsets_and_smaller_register_widths() {
    use super::packed_code::{body, executable, host_supports, profiles, supports};
    use crate::hir::{eval::evaluate, operation::Action};
    use crate::jit::packed_test;
    use crate::lir::x64::inst::simd::Native;
    use iced_x86::code_asm::{CodeAssembler, rax, xmm0};

    let [sse2, avx2, foundation, extended] = profiles();
    let ssse3 = Features {
        ssse3: true,
        ..sse2
    };
    let sse41 = Features {
        sse41: true,
        ..ssse3
    };
    let avx = Features { bytes: 32, ..sse41 };
    let vbmi2_only = Features {
        vbmi2: true,
        ..extended
    };
    let vbmi = Features {
        vbmi: true,
        ..vbmi2_only
    };
    for cpu in [
        sse2, ssse3, sse41, avx, avx2, foundation, extended, vbmi2_only, vbmi,
    ] {
        for (small, wide) in [
            (Lane::I8, Lane::I16),
            (Lane::U8, Lane::U16),
            (Lane::I16, Lane::I32),
            (Lane::U16, Lane::U32),
            (Lane::I32, Lane::I64),
            (Lane::U32, Lane::U64),
        ] {
            let s = Type::Vector(small);
            let w = Type::Vector(wide);
            for (input, output, action, regs) in [
                (s, w, Action::Widen(false), &[1, 3][..]),
                (s, w, Action::Widen(true), &[1, 3]),
                (w, s, Action::Narrow, &[1, 2, 3]),
                (s, s, Action::Shuffle, &[1, 2, 3, 4]),
                (w, w, Action::Shuffle, &[1, 2, 3, 4]),
                (s, small.scalar(), Action::Reduce(Binary::Add), &[1, 0]),
            ] {
                let Some(native) = Native::select(action, input, output, cpu) else {
                    continue;
                };
                let (mut bytes, text_len) = body(native, regs);
                let mut host_ok = cpu.bytes <= Features::host().bytes;
                for inst in Decoder::new(64, &bytes[..text_len], DecoderOptions::NONE) {
                    assert!(
                        !inst.is_invalid(),
                        "{action:?} {input:?} {output:?} {cpu:?}"
                    );
                    for &feature in inst.cpuid_features() {
                        assert!(
                            supports(cpu, feature),
                            "{action:?} {input:?} {output:?} on {cpu:?} emitted {:?} requiring {feature:?}",
                            inst.code()
                        );
                        host_ok &= host_supports(feature);
                    }
                }
                if host_ok {
                    if matches!(action, Action::Reduce(_)) {
                        // The wrapper saves vector register 0. Move the integer
                        // result from its assigned GPR into the low bits of XMM0.
                        let mut a = CodeAssembler::new(64).unwrap();
                        a.movq(xmm0, rax).unwrap();
                        bytes.extend(a.assemble(0).unwrap());
                    }
                    let code = executable(&bytes, cpu.bytes, regs);
                    let num_inputs = regs.len() - 1;
                    let mut slots = [[0u8; 64]; 4];
                    for (arg, slot) in slots[..2].iter_mut().enumerate() {
                        for (i, b) in slot[..cpu.bytes].iter_mut().enumerate() {
                            *b = (13 + 93 * arg + 31 * i) as u8;
                        }
                    }
                    let size = input.lane().unwrap().bytes();
                    let n = (cpu.bytes / size) as u64;
                    let indices = [n - 1, n, 2 * n - 1, 2 * n, 2 * n + 1, u64::MAX];
                    // SAFETY: the wrapper saves nonvolatile registers, reads and writes
                    // only these four slots and executes instructions supported by this host.
                    unsafe {
                        packed_test::with_kernel(&code, |call| {
                            for phase in 0..if matches!(action, Action::Shuffle) {
                                indices.len()
                            } else {
                                1
                            } {
                                for (i, slot) in
                                    slots[2][..cpu.bytes].chunks_exact_mut(size).enumerate()
                                {
                                    slot.copy_from_slice(
                                        &indices[(i + phase) % indices.len()].to_le_bytes()[..size],
                                    );
                                }
                                let expected = evaluate(
                                    action,
                                    input,
                                    output,
                                    slots[..3].try_into().unwrap(),
                                    cpu.bytes,
                                )
                                .unwrap();
                                let mut actual = slots;
                                call(&mut actual);
                                let len = output.bytes(cpu.bytes);
                                assert_eq!(
                                    &actual[num_inputs.max(2)][..len],
                                    &expected[..len],
                                    "{action:?} {input:?} {output:?} {cpu:?}"
                                );
                                assert_eq!(
                                    actual[..num_inputs],
                                    slots[..num_inputs],
                                    "{action:?} {input:?} {output:?} clobbered live input"
                                );
                            }
                        })
                    };
                }
            }
        }
    }
}
