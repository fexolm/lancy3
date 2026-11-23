use crate::ScalarKernel;
use crate::hir::{
    Binary, Function, Lane, Op, Simd, Terminator, Ternary, Type, Value, VectorAddress,
};
use crate::lir::x64::{self, Cpu};
use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic};
use splice_target::X86Features;

fn function(params: &[Type], result: Type, make: impl FnOnce(&[Value]) -> Simd) -> Function {
    let mut f = Function::new(params, result);
    let block = f.entry();
    let op = make(f.params(block));
    let result = f.append(block, Op::Simd(op)).unwrap();
    f.set_terminator(block, Terminator::Return(result));
    f
}

fn instructions(f: &Function) -> Vec<Mnemonic> {
    decoded(f).iter().map(Instruction::mnemonic).collect()
}

fn decoded(f: &Function) -> Vec<Instruction> {
    let (module, ids) = f.verified_module().unwrap();
    let code = x64::emit(&crate::test_target::target(), module, &ids).unwrap();
    Decoder::new(64, &code.bytes[..code.text_len], DecoderOptions::NONE)
        .into_iter()
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
fn zero_masked_load_has_no_merge_and_negative_zero_keeps_its_bits() {
    let cpu = Cpu::host();
    let n = cpu.bytes / 4;
    let ty = Type::Vector(Lane::F32);
    for fallback in [0, 0x8000_0000] {
        let mut f = Function::new(&[Type::Ptr, Type::I32], ty);
        let b = f.entry();
        let [ptr, count] = f.params(b).try_into().unwrap();
        let iota = f.append(b, Op::Simd(Simd::Iota(Lane::I32))).unwrap();
        let count = f.append(b, Op::Simd(Simd::Splat(count))).unwrap();
        let mask = f
            .append(b, Op::Simd(Simd::Bin(Binary::Lt, iota, count)))
            .unwrap();
        let pass = f.append(b, Op::Literal(Lane::F32, fallback)).unwrap();
        let pass = f.append(b, Op::Simd(Simd::Splat(pass))).unwrap();
        let out = f
            .append(
                b,
                Op::Simd(Simd::MaskedLoad {
                    address: VectorAddress {
                        base: ptr,
                        indices: None,
                    },
                    mask,
                    passthrough: pass,
                }),
            )
            .unwrap();
        f.set_terminator(b, Terminator::Return(out));
        let kernel = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        let mem = [7.0f32];
        let mut out = vec![0; cpu.bytes / 8];
        // SAFETY: the active lane reads the one initialized word in mem.
        unsafe { kernel.call_words_raw(&[mem.as_ptr() as u64, 1], &mut out) }.unwrap();
        let mut expected = vec![fallback; n];
        expected[0] = u128::from(mem[0].to_bits());
        assert_eq!(out, words(Lane::F32, &expected));
        let ops = decoded(&f);
        if cpu.bytes == 64 {
            let load = ops
                .iter()
                .find(|i| i.mnemonic() == Mnemonic::Vmovdqu32 && i.has_op_mask())
                .unwrap();
            assert_eq!(load.zeroing_masking(), fallback == 0);
        } else if cpu.bytes == 32 && cpu.isa.avx10 == 0 {
            assert!(ops.iter().any(|i| i.mnemonic() == Mnemonic::Vmaskmovps));
            assert_eq!(
                ops.iter().any(|i| i.mnemonic() == Mnemonic::Vblendvps),
                fallback != 0
            );
        }
    }
}

#[test]
fn pack_and_expand_preserve_lane_order_for_all_byte_masks() {
    let cpu = Cpu::host();
    let n = cpu.bytes / 4;
    let ty = Type::Vector(Lane::I32);
    let mask_width = n.min(8);
    let values: Vec<_> = (0..n).map(|i| 0x8000_0000u128 + i as u128).collect();
    for expand in [false, true] {
        let f = function(&[ty, ty], ty, |args| {
            if expand {
                Simd::Expand(args[0], args[1])
            } else {
                Simd::Compress(args[0], args[1])
            }
        });
        let kernel = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        for bits in 0..(1usize << mask_width) {
            let active = |i: usize| bits & (1 << (i % mask_width)) != 0;
            let mask: Vec<_> = (0..n)
                .map(|i| if active(i) { 0x8000_0000 } else { 1 })
                .collect();
            let inputs = [words(Lane::I32, &values), words(Lane::I32, &mask)].concat();
            let mut expected = vec![0; n];
            for (k, i) in (0..n).filter(|&i| active(i)).enumerate() {
                if expand {
                    expected[i] = values[k];
                } else {
                    expected[k] = values[i];
                }
            }
            check(&kernel, &inputs, &words(Lane::I32, &expected));
        }
        if cpu.bytes == 32 && cpu.isa.avx2 && cpu.isa.avx10 == 0 {
            assert!(instructions(&f).contains(&Mnemonic::Vpermd));
        } else if cpu.bytes == 16 && cpu.isa.avx10 == 0 {
            let ops = instructions(&f);
            assert!(ops.contains(&Mnemonic::Movmskps));
            if cpu.isa.ssse3 {
                assert!(ops.contains(&Mnemonic::Pshufb));
            } else {
                assert!(ops.contains(&Mnemonic::Pshufd) && ops.contains(&Mnemonic::Pand));
            }
        }
    }
}

#[test]
fn avx10_short_fp16_arithmetic_uses_the_requested_width() {
    use super::packed_code::{body, executable, host_supports, profiles, supports};
    use crate::hir::operation::Action;
    use crate::jit::packed_test;
    use crate::lir::x64::inst::simd::Native;

    for bytes in [16, 32] {
        let cpu = Cpu {
            bytes,
            isa: X86Features {
                avx10: 1,
                fp16: true,
                ..profiles()[3].isa
            },
        };
        let ty = Type::Vector(Lane::F16);
        for (action, mnemonic, inputs, expected, regs) in [
            (
                Action::Binary(Binary::Add),
                Mnemonic::Vaddph,
                [0x03ffu16, 1, 0],
                0x0400u16,
                &[1, 2, 3][..],
            ),
            (
                Action::Binary(Binary::Mul),
                Mnemonic::Vmulph,
                [0x0400, 0x3800, 0],
                0x0200,
                &[1, 2, 3][..],
            ),
            (
                Action::Ternary(Ternary::MulAdd),
                Mnemonic::Vfmadd213ph,
                [0x0400, 0x3801, 1],
                0x0202,
                &[1, 2, 3, 1][..],
            ),
        ] {
            assert!(
                Native::select(
                    action,
                    ty,
                    ty,
                    Cpu {
                        isa: X86Features {
                            fp16: false,
                            ..cpu.isa
                        },
                        ..cpu
                    }
                )
                .is_none()
            );
            let native = Native::select(action, ty, ty, cpu).unwrap();
            let (code, len) = body(native, regs);
            let mut host_ok = bytes <= Cpu::host().bytes;
            let mut selected = false;
            for inst in Decoder::new(64, &code[..len], DecoderOptions::NONE) {
                assert!(!inst.is_invalid());
                if inst.mnemonic() == mnemonic {
                    selected = true;
                    assert_eq!(inst.op0_register().size(), bytes);
                }
                for &feature in inst.cpuid_features() {
                    assert!(supports(cpu, feature));
                    host_ok &= host_supports(feature);
                }
            }
            assert!(selected);
            if host_ok {
                let mut slots = [[0; 64]; 4];
                for (slot, bits) in slots.iter_mut().zip(inputs) {
                    for lane in slot[..bytes].chunks_exact_mut(2) {
                        lane.copy_from_slice(&bits.to_le_bytes());
                    }
                }
                let code = executable(&code, bytes, regs);
                // SAFETY: the verified wrapper accesses only the four slots
                // and the host supports each decoded instruction.
                unsafe { packed_test::with_kernel(&code, |call| call(&mut slots)) };
                assert!(
                    slots[regs.len() - 1][..bytes]
                        .chunks_exact(2)
                        .all(|lane| lane == expected.to_le_bytes())
                );
            }
        }
    }
}

#[test]
fn live_predicates_spill_safely_across_an_outlined_call() {
    use std::sync::Arc;
    let mut child = Function::scalar(1);
    let b = child.entry();
    let x = child.params(b)[0];
    let i = child.append(b, Op::Simd(Simd::Iota(Lane::I32))).unwrap();
    let limit = child.append(b, Op::Simd(Simd::Splat(x))).unwrap();
    let cmp = child
        .append(b, Op::Simd(Simd::Bin(Binary::Lt, i, limit)))
        .unwrap();
    let status = child.append(b, Op::Simd(Simd::AnyTrue(cmp))).unwrap();
    child.set_terminator(b, Terminator::Return(status));

    let mut f = Function::scalar(9);
    let b = f.entry();
    let args = f.params(b).to_vec();
    let i = f.append(b, Op::Simd(Simd::Iota(Lane::I32))).unwrap();
    let masks: Vec<_> = args[..8]
        .iter()
        .map(|&x| {
            let limit = f.append(b, Op::Simd(Simd::Splat(x))).unwrap();
            f.append(b, Op::Simd(Simd::Bin(Binary::Lt, i, limit)))
                .unwrap()
        })
        .collect();
    let call = f.add_call(Arc::new(child), &args[8..]);
    let mut bits = f.append(b, Op::Call(call)).unwrap();
    for (i, mask) in masks.into_iter().enumerate() {
        let any = f.append(b, Op::Simd(Simd::AnyTrue(mask))).unwrap();
        let weight = f.append(b, Op::Const(1 << i)).unwrap();
        let bit = f.append(b, Op::Bin(Binary::And, any, weight)).unwrap();
        bits = f.append(b, Op::Bin(Binary::Or, bits, bit)).unwrap();
    }
    f.set_terminator(b, Terminator::Return(bits));
    let k = ScalarKernel::compile(
        &f,
        &crate::test_target::options().with_inlining(crate::Inlining::Never),
    )
    .unwrap();
    assert_eq!(k.call(&[0, 2, 0, 4, 5, 0, 0, 8, 0]).unwrap(), 0b1001_1010);
    assert_eq!(k.call(&[3, 0, 6, 0, 0, 9, 1, 0, 0]).unwrap(), 0b0110_0101);
    if Cpu::host().bytes == 64 {
        let ops = decoded(&f);
        assert!(ops.iter().any(
            |i| matches!(i.mnemonic(), Mnemonic::Kmovw | Mnemonic::Kmovq)
                && (i.op0_kind() == iced_x86::OpKind::Memory
                    || i.op1_kind() == iced_x86::OpKind::Memory)
        ));
    }
}

#[cfg(feature = "experimental-simd")]
#[test]
fn adjacent_vector_load_can_be_a_packed_memory_operand() {
    let ty = Type::Vector(Lane::I32);
    let mut f = Function::new(&[Type::Ptr, ty], ty);
    let b = f.entry();
    let [p, mask] = f.params(b).try_into().unwrap();
    let v = f.append(b, Op::Simd(Simd::Load(Lane::I32, p))).unwrap();
    let r = f
        .append(b, Op::Simd(Simd::Bin(Binary::And, v, mask)))
        .unwrap();
    f.set_terminator(b, Terminator::Return(r));
    let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
    let n = crate::test_target::width();
    let mem = [0xbcu8; 65];
    // Use an address with an odd low bit regardless of the stack layout.
    let offset = usize::from(mem.as_ptr() as usize & 1 == 0);
    let p = mem[offset..].as_ptr();
    let mut args = vec![p as u64];
    args.extend(vec![0x7f7f_7f7f_7f7f_7f7f; n / 8]);
    let mut out = vec![0; n / 8];
    // SAFETY: p addresses at least n readable bytes in the live array.
    unsafe { k.call_words_raw(&args, &mut out) }.unwrap();
    assert_eq!(out, vec![0x3c3c_3c3c_3c3c_3c3c; n / 8]);
    let cpu = Cpu::host();
    if cpu.bytes == 64 || cpu.bytes == 32 && cpu.isa.avx2 {
        let (m, ids) = f.verified_module().unwrap();
        let code = x64::emit(&crate::test_target::target(), m, &ids).unwrap();
        assert!(
            Decoder::new(64, &code.bytes[..code.text_len], DecoderOptions::NONE)
                .iter()
                .any(
                    |i| matches!(i.mnemonic(), Mnemonic::Vpand | Mnemonic::Vpandd)
                        && i.op2_kind() == iced_x86::OpKind::Memory
                )
        );
    }
}

#[cfg(feature = "experimental-simd")]
#[test]
fn constant_shuffle_broadcasts_the_selected_table_lane() {
    let ty = Type::Vector(Lane::I32);
    let n = crate::test_target::width() / 4;
    let values: Vec<_> = (1..=2 * n).map(|v| v as u128).collect();
    let args = words(Lane::I32, &values);
    for index in [0, n - 1, n, 2 * n - 1, 2 * n] {
        let mut f = Function::new(&[ty, ty], ty);
        let b = f.entry();
        let [x, y] = f.params(b).try_into().unwrap();
        let i = f.append(b, Op::Const(index as i32)).unwrap();
        let i = f.append(b, Op::Simd(Simd::Splat(i))).unwrap();
        let out = f.append(b, Op::Simd(Simd::Shuffle(x, y, i))).unwrap();
        f.set_terminator(b, Terminator::Return(out));
        let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        let value = values.get(index).copied().unwrap_or(0);
        check(&k, &args, &words(Lane::I32, &vec![value; n]));
        let cpu = Cpu::host();
        if index < 2 * n && (cpu.bytes != 32 || cpu.isa.avx2) {
            let expected = if cpu.bytes == 16 {
                Mnemonic::Pshufd
            } else {
                Mnemonic::Vpbroadcastd
            };
            assert!(instructions(&f).contains(&expected));
        }
    }
}

#[test]
fn masked_load_uses_the_comparison_mask_directly_on_avx512() {
    for lane in [Lane::I32, Lane::U32, Lane::I64, Lane::U64] {
        let ty = Type::Vector(lane);
        let mut f = Function::new(&[Type::Ptr, ty, ty], ty);
        let b = f.entry();
        let [base, lhs, rhs] = f.params(b).try_into().unwrap();
        let mask = f
            .append(b, Op::Simd(Simd::Bin(Binary::Lt, lhs, rhs)))
            .unwrap();
        let out = f
            .append(
                b,
                Op::Simd(Simd::MaskedLoad {
                    address: VectorAddress {
                        base,
                        indices: None,
                    },
                    mask,
                    passthrough: lhs,
                }),
            )
            .unwrap();
        f.set_terminator(b, Terminator::Return(out));
        let size = lane.bytes();
        let n = crate::test_target::width() / size;
        let top = 1u128 << (size * 8 - 1);
        let lhs: Vec<_> = (0..n).map(|i| if i % 2 == 0 { top } else { 0 }).collect();
        let memory = words(lane, &vec![42; n]);
        let mut args = vec![memory.as_ptr() as u64];
        args.extend(words(lane, &lhs));
        args.extend(words(lane, &vec![1; n]));
        let mut out = vec![0; memory.len()];
        let expected: Vec<_> = lhs
            .into_iter()
            .map(|x| if lane.signed() || x == 0 { 42 } else { x })
            .collect();
        let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        // SAFETY: the live memory buffer contains a full vector of initialized lanes.
        unsafe { k.call_words_raw(&args, &mut out) }.unwrap();
        assert_eq!(out, words(lane, &expected));
        if Cpu::host().bytes == 64 {
            let ops = instructions(&f);
            assert!(ops.contains(&if size == 4 {
                Mnemonic::Vmovdqu32
            } else {
                Mnemonic::Vmovdqu64
            }));
            assert!(!ops.iter().any(|op| matches!(
                op,
                Mnemonic::Vpternlogd
                    | Mnemonic::Vpternlogq
                    | Mnemonic::Vpmovd2m
                    | Mnemonic::Vpmovq2m
            )));
        }
    }
}

#[test]
fn comparison_and_any_test_every_lane_and_every_bit() {
    let cpu = Cpu::host();
    for lane in [Lane::I8, Lane::I16, Lane::I32, Lane::I64] {
        let ty = Type::Vector(lane);
        let n = cpu.bytes / lane.bytes();
        let mut f = Function::new(&[ty, ty, ty], Type::I32);
        let b = f.entry();
        let [x, y, enabled] = f.params(b).try_into().unwrap();
        let less = f.append(b, Op::Simd(Simd::Bin(Binary::Lt, x, y))).unwrap();
        let mask = f
            .append(b, Op::Simd(Simd::Bin(Binary::And, enabled, less)))
            .unwrap();
        let any = f.append(b, Op::Simd(Simd::AnyTrue(mask))).unwrap();
        f.set_terminator(b, Terminator::Return(any));
        let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        let mut rhs = vec![0; n];
        rhs[n - 1] = 1;
        let mut gate = vec![0; n];
        for (last, expected) in [(0, 0), (1, u32::MAX as u64)] {
            gate[n - 1] = last;
            let args = [
                words(lane, &vec![0; n]),
                words(lane, &rhs),
                words(lane, &gate),
            ]
            .concat();
            check(&k, &args, &[expected]);
        }
        if cpu.bytes == 64 && (lane.bytes() >= 4 || cpu.isa.bw) {
            let ops = instructions(&f);
            assert!(ops.contains(&match lane {
                Lane::I8 => Mnemonic::Kortestq,
                Lane::I16 => Mnemonic::Kortestd,
                _ => Mnemonic::Kortestw,
            }));
            assert!(!ops.iter().any(|op| matches!(
                op,
                Mnemonic::Vpmovm2b
                    | Mnemonic::Vpmovm2w
                    | Mnemonic::Vpternlogd
                    | Mnemonic::Vpternlogq
            )));
        }
    }
}

#[test]
fn comparing_float_lanes_for_select_preserves_nan_and_zero_bits() {
    let cpu = Cpu::host();
    let lhs_values = [f32::from_bits(0x7f81_1234), -0.0, 2.0, -1.0];
    let rhs_values = [1.0f32, 0.0, 1.0, 3.0];
    let n = cpu.bytes / 4;
    let pack = |values: &[f32]| {
        words(
            Lane::F32,
            &values
                .iter()
                .cycle()
                .take(n)
                .map(|v| u128::from(v.to_bits()))
                .collect::<Vec<_>>(),
        )
    };
    let args = [pack(&lhs_values), pack(&rhs_values)].concat();
    let ty = Type::Vector(Lane::F32);
    for op in [
        Binary::Eq,
        Binary::Ne,
        Binary::Lt,
        Binary::Ge,
        Binary::Unordered,
    ] {
        let mut f = Function::new(&[ty, ty], ty);
        let b = f.entry();
        let [lhs, rhs] = f.params(b).try_into().unwrap();
        let condition = f.append(b, Op::Simd(Simd::Bin(op, lhs, rhs))).unwrap();
        let picked = f
            .append(
                b,
                Op::Simd(Simd::Ternary(Ternary::Select, condition, lhs, rhs)),
            )
            .unwrap();
        f.set_terminator(b, Terminator::Return(picked));
        let expected: Vec<_> = lhs_values
            .iter()
            .zip(rhs_values)
            .map(|(&x, y)| {
                let pick_x = match op {
                    Binary::Eq => x == y,
                    Binary::Ne => x != y,
                    Binary::Lt => x < y,
                    Binary::Ge => x >= y,
                    _ => x.is_nan() || y.is_nan(),
                };
                if pick_x { x } else { y }
            })
            .collect();
        check(
            &ScalarKernel::compile(&f, &crate::test_target::options()).unwrap(),
            &args,
            &pack(&expected),
        );
        if cpu.bytes == 64 {
            let ops = instructions(&f);
            assert!(ops.contains(&Mnemonic::Vcmpps));
            assert!(ops.contains(&Mnemonic::Vpblendmd));
            assert!(!ops.contains(&Mnemonic::Vpternlogd));
        }
    }
}

#[test]
fn dword_shifts_carry_lanes_across_register_chunks() {
    let cpu = Cpu::host();
    let n = cpu.bytes / 4;
    let ty = Type::Vector(Lane::I32);
    let table: Vec<_> = (1..=n as u128).collect();
    let args = [words(Lane::I32, &table), words(Lane::I32, &vec![255; n])].concat();
    for shift in 1..n {
        let mut f = Function::new(&[ty, ty], ty);
        let b = f.entry();
        let [x, y] = f.params(b).try_into().unwrap();
        let offset = f.append(b, Op::Literal(Lane::U32, shift as u128)).unwrap();
        let offset = f.append(b, Op::Simd(Simd::Splat(offset))).unwrap();
        let iota = f.append(b, Op::Simd(Simd::Iota(Lane::U32))).unwrap();
        let indices = f
            .append(b, Op::Simd(Simd::Bin(Binary::Sub, iota, offset)))
            .unwrap();
        let result = f.append(b, Op::Simd(Simd::Shuffle(x, y, indices))).unwrap();
        f.set_terminator(b, Terminator::Return(result));
        let mut expected = vec![0; n];
        expected[shift..].copy_from_slice(&table[..n - shift]);
        check(
            &ScalarKernel::compile(&f, &crate::test_target::options()).unwrap(),
            &args,
            &words(Lane::I32, &expected),
        );
        if cpu.bytes != 32 || cpu.isa.avx2 {
            assert!(instructions(&f).contains(&match cpu.bytes {
                16 => Mnemonic::Pslldq,
                32 => Mnemonic::Vperm2i128,
                _ => Mnemonic::Valignd,
            }));
        }
    }
}

#[test]
fn broadcast_shift_counts_become_immediates_with_wrapping_and_signedness() {
    let cpu = Cpu::host();
    let n = cpu.bytes / 4;
    let values = [0x80000004u32, 0x01234567, u32::MAX, 1];
    for lane in [Lane::I32, Lane::U32] {
        let ty = Type::Vector(lane);
        let pack = |data: &[u32]| {
            let lanes = data
                .iter()
                .cycle()
                .take(n)
                .map(|&v| u128::from(v))
                .collect::<Vec<_>>();
            words(lane, &lanes)
        };
        for op in [
            Binary::Shl,
            Binary::Shr,
            Binary::RotateLeft,
            Binary::RotateRight,
        ] {
            for count in [0u32, 31, 32, 33, u32::MAX] {
                let mut f = Function::new(&[ty], ty);
                let b = f.entry();
                let x = f.params(b)[0];
                let shift = f.append(b, Op::Literal(lane, count.into())).unwrap();
                let counts = f.append(b, Op::Simd(Simd::Splat(shift))).unwrap();
                let y = f.append(b, Op::Simd(Simd::Bin(op, x, counts))).unwrap();
                let z = f.append(b, Op::Simd(Simd::Bin(Binary::Xor, x, y))).unwrap();
                f.set_terminator(b, Terminator::Return(z));
                let expected = values.map(|x| {
                    let y = match op {
                        Binary::Shl => x.wrapping_shl(count),
                        Binary::Shr if lane.signed() => (x as i32).wrapping_shr(count) as u32,
                        Binary::Shr => x.wrapping_shr(count),
                        Binary::RotateLeft => x.rotate_left(count),
                        _ => x.rotate_right(count),
                    };
                    x ^ y
                });
                check(
                    &ScalarKernel::compile(&f, &crate::test_target::options()).unwrap(),
                    &pack(&values),
                    &pack(&expected),
                );
                if cpu.bytes == 64 && count & 31 != 0 {
                    let expected = match op {
                        Binary::Shl => Mnemonic::Vpslld,
                        Binary::Shr if lane.signed() => Mnemonic::Vpsrad,
                        Binary::Shr => Mnemonic::Vpsrld,
                        Binary::RotateLeft => Mnemonic::Vprold,
                        _ => Mnemonic::Vprord,
                    };
                    assert!(instructions(&f).contains(&expected));
                }
            }
        }
    }
}

#[test]
fn constant_extract_reads_low_and_high_chunks_without_changing_lane_bits() {
    let cpu = Cpu::host();
    for lane in [
        Lane::I8,
        Lane::U16,
        Lane::I32,
        Lane::I64,
        Lane::F16,
        Lane::F32,
        Lane::F64,
        Lane::I128,
    ] {
        let size = lane.bytes();
        let n = cpu.bytes / size;
        let values: Vec<_> = (0..n)
            .map(|i| {
                let top = 1u128 << (8 * size - 1);
                match (lane, i) {
                    (Lane::F32, 0) => u128::from(f32::NEG_INFINITY.to_bits()),
                    (Lane::F64, 0) => u128::from(f64::NEG_INFINITY.to_bits()),
                    _ => top | i as u128,
                }
            })
            .collect();
        let args = words(lane, &values);
        for index in 0..n {
            let mut f = Function::new(&[Type::Vector(lane)], lane.scalar());
            let b = f.entry();
            let x = f.params(b)[0];
            let i = f.append(b, Op::Const(index as i32)).unwrap();
            let y = f.append(b, Op::Simd(Simd::ExtractLane(x, i))).unwrap();
            f.set_terminator(b, Terminator::Return(y));
            let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
            check(&k, &args, &words(lane, &values[index..index + 1]));
            if index * size >= 16 {
                assert!(instructions(&f).contains(&if cpu.bytes == 32 {
                    Mnemonic::Vextractf128
                } else {
                    Mnemonic::Vextracti32x4
                }));
            }
        }
    }
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
        let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        let n = crate::test_target::width() / lane.bytes();
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
        let kernel = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        let n = crate::test_target::width() / lane.bytes();
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
        let n = crate::test_target::width() / lane.bytes();
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
            let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
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
        let width = crate::test_target::width();
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
            let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
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
        let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
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
        let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        let bytes = lane.bytes();
        let n = crate::test_target::width() / bytes;
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
        let mut out = vec![0; crate::test_target::width() / 8];
        // SAFETY: the kernel and allocation remain live. All active accesses target
        // the first two elements of this exclusively borrowed allocation.
        unsafe { k.call_words_raw(&args, &mut out) }.unwrap();
        assert_eq!(out, words(lane, &expected_output));
        assert_eq!(memory, words(lane, &expected));

        let cpu = Cpu::host();
        let evex = cpu.bytes == 64 || cpu.isa.avx10 >= 1;
        if evex || cpu.bytes == 32 && cpu.isa.avx2 {
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
    if Cpu::host().bytes != 64 {
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
    let ssse3 = Cpu {
        isa: X86Features {
            ssse3: true,
            ..sse2.isa
        },
        ..sse2
    };
    let sse41 = Cpu {
        isa: X86Features {
            sse41: true,
            ..ssse3.isa
        },
        ..ssse3
    };
    let vbmi2_only = Cpu {
        isa: X86Features {
            vbmi2: true,
            ..extended.isa
        },
        ..extended
    };
    let vbmi = Cpu {
        isa: X86Features {
            vbmi: true,
            ..vbmi2_only.isa
        },
        ..vbmi2_only
    };
    let avx10 = [16, 32].map(|bytes| Cpu {
        bytes,
        isa: X86Features {
            avx10: 1,
            ..vbmi.isa
        },
    });
    for cpu in [
        sse2, ssse3, sse41, avx2, foundation, extended, vbmi2_only, vbmi,
    ]
    .into_iter()
    .chain(avx10)
    {
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
                (s, small.scalar(), Action::Reduce(Binary::Xor), &[1, 0]),
            ] {
                let Some(native) = Native::select(action, input, output, cpu) else {
                    assert!(!(cpu.isa.avx10 != 0 && matches!(action, Action::Shuffle)));
                    continue;
                };
                let (mut bytes, text_len) = body(native, regs);
                let mut host_ok = cpu.bytes <= Cpu::host().bytes;
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
                    if cpu.isa.avx10 != 0 {
                        for i in 0..inst.op_count() {
                            let reg = inst.op_register(i);
                            assert!(!reg.is_vector_register() || reg.size() <= cpu.bytes);
                        }
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
