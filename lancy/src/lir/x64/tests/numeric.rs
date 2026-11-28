use crate::{
    CallError, Inlining, Kernel, SignatureError, Trap,
    hir::{Binary, Function, Lane, Op, Simd, Terminator, Ternary, Type, Unary},
};
use splice_target::X86Features;
use std::sync::Arc;

fn kernel(types: &[Type], result: Type, body: impl FnOnce(&[crate::hir::Value]) -> Op) -> Kernel {
    let mut f = Function::new(types, result);
    let b = f.entry();
    let op = body(f.params(b));
    let r = f.append(b, op).unwrap();
    f.set_terminator(b, Terminator::Return(r));
    Kernel::compile(&f, &crate::test_target::options()).unwrap()
}

#[test]
fn each_integer_width_wraps_and_divides_with_its_signedness() {
    use Lane::*;
    for l in [
        I8, U8, I16, U16, I32, U32, I64, U64, I128, U128, Isize, Usize,
    ] {
        for ty in [l.scalar(), Type::Vector(l)] {
            let bytes = ty.bytes(crate::test_target::width());
            let words = bytes.div_ceil(8);
            for op in [Binary::Add, Binary::Div, Binary::Lt] {
                let k = kernel(&[ty, ty], op.result_type(ty), |v| Op::Bin(op, v[0], v[1]));
                let mut a = vec![0xff; words * 8];
                let mut b = vec![0; words * 8];
                let mut expected = Vec::new();
                for (lane, (x, y)) in a[..bytes]
                    .chunks_mut(l.bytes())
                    .zip(b[..bytes].chunks_mut(l.bytes()))
                    .enumerate()
                {
                    let val = if lane % 2 == 0 { u128::MAX } else { 18 };
                    x.copy_from_slice(&val.to_le_bytes()[..l.bytes()]);
                    y[0] = 2;
                    let mask = u128::MAX >> (128 - 8 * l.bytes());
                    let n = match op {
                        Binary::Add => val.wrapping_add(2),
                        Binary::Div if l.signed() && val == u128::MAX => 0,
                        Binary::Div => (val & mask) / 2,
                        Binary::Lt if l.signed() && val == u128::MAX => u128::MAX,
                        Binary::Lt => 0,
                        _ => unreachable!(),
                    };
                    expected.extend_from_slice(
                        &n.to_le_bytes()[..if ty == l.scalar() && op == Binary::Lt {
                            4
                        } else {
                            l.bytes()
                        }],
                    );
                }
                let args: Vec<_> = a
                    .into_iter()
                    .chain(b)
                    .collect::<Vec<_>>()
                    .chunks_exact(8)
                    .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
                    .collect();
                let mut out = vec![0; expected.len().div_ceil(8)];
                k.call_packed(&args, &mut out).unwrap();
                let actual: Vec<_> = out.into_iter().flat_map(u64::to_le_bytes).collect();
                assert_eq!(&actual[..expected.len()], expected, "{ty:?} {op:?}");
            }
        }
    }
}

#[test]
fn fp_results_include_signed_zero_nan_and_fused_rounding() {
    for ty in [
        Type::F32,
        Type::Vector(Lane::F32),
        Type::F64,
        Type::Vector(Lane::F64),
    ] {
        let bytes = ty.bytes(crate::test_target::width());
        let lane = ty.lane().unwrap();
        let from = |x: f64| {
            if lane == Lane::F32 {
                u64::from((x as f32).to_bits())
            } else {
                x.to_bits()
            }
        };
        let run = |k: &Kernel, inputs: &[u64]| {
            let mut args = Vec::new();
            for &n in inputs {
                let mut a = vec![0; bytes.div_ceil(8) * 8];
                for p in a[..bytes].chunks_mut(lane.bytes()) {
                    p.copy_from_slice(&n.to_le_bytes()[..lane.bytes()]);
                }
                args.extend(
                    a.chunks_exact(8)
                        .map(|x| u64::from_le_bytes(x.try_into().unwrap())),
                );
            }
            let mut out = vec![0; bytes.div_ceil(8)];
            k.call_packed(&args, &mut out).unwrap();
            let bits: Vec<_> = out.into_iter().flat_map(u64::to_le_bytes).collect();
            bits[..bytes]
                .chunks_exact(lane.bytes())
                .map(|x| crate::support::numeric::read(x) as u64)
                .collect::<Vec<_>>()
        };
        let sqrt = kernel(&[ty], ty, |v| Op::Unary(Unary::Sqrt, v[0]));
        assert!(run(&sqrt, &[from(-0.)]).iter().all(|&x| x == from(-0.)));
        let div = kernel(&[ty, ty], ty, |v| Op::Bin(Binary::Div, v[0], v[1]));
        for x in run(&div, &[0, 0]) {
            assert!(if lane == Lane::F32 {
                f32::from_bits(x as u32).is_nan()
            } else {
                f64::from_bits(x).is_nan()
            });
        }
        let fma = kernel(&[ty; 3], ty, |v| {
            Op::Ternary(Ternary::MulAdd, v[0], v[1], v[2])
        });
        let eps = if lane == Lane::F32 {
            f32::EPSILON as f64
        } else {
            f64::EPSILON
        };
        assert!(
            run(&fma, &[from(1. + eps), from(1. - eps), from(-1.)])
                .iter()
                .all(|&x| x == from(-eps * eps))
        );
    }
}

#[test]
fn sse_fma_helper_preserves_live_values_for_both_x64_abis() {
    use splice_target::{CallingConvention, Profile, Target};
    let vector = Type::Vector(Lane::F32);
    let bits = Type::Vector(Lane::I32);
    let mut f = Function::new(&[vector, vector, vector, bits], bits);
    let b = f.entry();
    let [a, c, d, mask] = f.params(b).try_into().unwrap();
    let fused = f.append(b, Op::Ternary(Ternary::MulAdd, a, c, d)).unwrap();
    let fused = f.append(b, Op::Bitcast(bits, fused)).unwrap();
    let result = f.append(b, Op::Bin(Binary::Xor, fused, mask)).unwrap();
    f.set_terminator(b, Terminator::Return(result));
    let args = [
        [0x3f800001, 0x40000000, 0x80000000, 0x7f7fffff],
        [0x3f7ffffe, 0x40400000, 0x3f800000, 0x40000000],
        [0xbf800000, 0x40800000, 0x80000000, 0xff7fffff],
        [1, 2, 3, 4],
    ];
    let pack = |v: [u32; 4]| -> [u64; 2] {
        [
            u64::from(v[0]) | (u64::from(v[1]) << 32),
            u64::from(v[2]) | (u64::from(v[3]) << 32),
        ]
    };
    let words = args.into_iter().flat_map(pack).collect::<Vec<_>>();
    let expected = pack(std::array::from_fn(|i| {
        f32::from_bits(args[0][i])
            .mul_add(f32::from_bits(args[1][i]), f32::from_bits(args[2][i]))
            .to_bits()
            ^ args[3][i]
    }));
    for convention in [CallingConvention::SysV, CallingConvention::Win64] {
        let target = Target::from_profile(Profile::Sse)
            .unwrap()
            .with_calling_convention(convention)
            .unwrap();
        let code = Kernel::compile(&f, &crate::CompileOptions::new(target)).unwrap();
        let mut output = [0; 2];
        code.call_packed(&words, &mut output).unwrap();
        assert_eq!(output, expected, "{convention:?}");
    }
}

#[test]
fn wide_spills_and_outlined_calls_preserve_every_lane() {
    let v = Type::Vector(Lane::U64);
    let w = crate::test_target::width() / 8;
    let mut leaf = Function::new(&[v, Type::I32, v], v);
    let b = leaf.entry();
    let [x, idx, y] = leaf.params(b).try_into().unwrap();
    let z = leaf.append(b, Op::Bin(Binary::Add, x, y)).unwrap();
    leaf.append_inst(b, Op::Simd(Simd::ExtractLane(z, idx)))
        .unwrap();
    leaf.set_terminator(b, Terminator::Return(z));
    let mut types = vec![v; 22];
    types.push(Type::I32);
    let mut root = Function::new(&types, v);
    let b = root.entry();
    let args = root.params(b).to_vec();
    let site = root.add_call(Arc::new(leaf), &[args[0], args[22], args[1]]);
    let mut sum = root.append(b, Op::Call(site)).unwrap();
    for &arg in &args[2..22] {
        sum = root.append(b, Op::Bin(Binary::Add, sum, arg)).unwrap();
    }
    root.set_terminator(b, Terminator::Return(sum));
    for policy in [Inlining::Never, Inlining::Always] {
        let k =
            Kernel::compile(&root, &crate::test_target::options().with_inlining(policy)).unwrap();
        let mut input: Vec<_> = (0..22 * w).map(|i| 0x100000003 * (i + 1) as u64).collect();
        input.push(0);
        let mut out = vec![0; w];
        k.call_packed(&input, &mut out).unwrap();
        for i in 0..w {
            assert_eq!(out[i], (0..22).map(|j| input[j * w + i]).sum());
        }
        *input.last_mut().unwrap() = w as u64;
        assert_eq!(
            k.call_packed(&input, &mut out),
            Err(CallError::Trap(Trap::Bounds))
        );
    }
}

#[test]
fn unaligned_vector_store_precedes_trap() {
    let mut f = Function::new(&[Type::Ptr, Type::I32], Type::I32);
    let b = f.entry();
    let [p, i] = f.params(b).try_into().unwrap();
    let x = f.append(b, Op::Load(Type::Vector(Lane::I32), p)).unwrap();
    let y = f.append(b, Op::Bin(Binary::Add, x, x)).unwrap();
    f.append_inst(b, Op::Store(p, y)).unwrap();
    let n = f.append(b, Op::Simd(Simd::LaneCount(Lane::I32))).unwrap();
    f.append_inst(b, Op::BoundsCheck(i, n)).unwrap();
    f.append_inst(b, Op::Store(p, x)).unwrap();
    f.set_terminator(b, Terminator::Return(i));
    let k = Kernel::compile(&f, &crate::test_target::options()).unwrap();
    let w = crate::test_target::width();
    let original: Vec<_> = (0..w + 2).map(|i| (201 + 19 * i) as u8).collect();
    for (i, result) in [
        (0, Ok(0)),
        (w as u64 / 4, Err(CallError::Trap(Trap::Bounds))),
    ] {
        let mut buf = original.clone();
        let mut expected = original.clone();
        if result.is_err() {
            for x in expected[1..1 + w].chunks_exact_mut(4) {
                x.copy_from_slice(
                    &i32::from_le_bytes((*x).try_into().unwrap())
                        .wrapping_mul(2)
                        .to_le_bytes(),
                );
            }
        }
        // SAFETY: p points one byte into this live, exclusively owned width+2 byte buffer.
        let actual = unsafe { k.call_word(&[buf[1..].as_mut_ptr() as u64, i]) };
        assert_eq!(actual, result);
        assert_eq!(buf, expected);
    }
}

#[test]
fn instruction_encodings_require_their_exact_isa() {
    use crate::{
        lir::x64::{Cpu, inst::simd::Native},
        support::numeric::Action,
    };
    use iced_x86::{Code, Decoder, DecoderOptions};
    let base = Cpu {
        bytes: 16,
        isa: X86Features {
            avx2: false,
            avx10: 0,
            bw: false,
            dq: false,
            sse41: false,
            ssse3: false,
            sse42: false,
            vbmi2: false,
            vbmi: false,
            f16c: false,
            bf16: false,
            fp16: false,
            fma: false,
            vpopcnt: false,
            bitalg: false,
            cd: false,
            ..X86Features::default()
        },
    };
    let avx2 = Cpu {
        bytes: 32,
        isa: X86Features {
            avx: true,
            avx2: true,
            ..base.isa
        },
    };
    let avx512 = Cpu {
        bytes: 64,
        isa: X86Features {
            avx512f: true,
            ..avx2.isa
        },
    };
    let vector = Type::Vector;
    let encode = |features, action, input, output, regs: &[usize]| {
        let op = Native::select(action, input, output, features)?;
        Some(super::packed_code::body(op, regs).0)
    };
    let binary = |cpu, l, op| {
        encode(cpu, Action::Binary(op), vector(l), vector(l), &[1, 2, 1]).map(|bytes| {
            Decoder::new(64, &bytes, DecoderOptions::NONE)
                .decode()
                .code()
        })
    };
    assert_eq!(
        binary(base, Lane::I32, Binary::Add),
        Some(Code::Paddd_xmm_xmmm128)
    );
    assert_eq!(
        binary(avx2, Lane::F32, Binary::Add),
        Some(Code::VEX_Vaddps_ymm_ymm_ymmm256)
    );
    assert_eq!(
        binary(avx2, Lane::I32, Binary::Add),
        Some(Code::VEX_Vpaddd_ymm_ymm_ymmm256)
    );
    assert!(binary(avx512, Lane::I32, Binary::Add).is_some());
    assert!(binary(avx512, Lane::U8, Binary::Add).is_some());
    assert!(
        binary(
            Cpu {
                isa: X86Features {
                    bw: true,
                    ..avx512.isa
                },
                ..avx512
            },
            Lane::U8,
            Binary::Add
        )
        .is_some()
    );
    let mul = encode(
        avx512,
        Action::Binary(Binary::Mul),
        vector(Lane::U64),
        vector(Lane::U64),
        &[1, 2, 3],
    )
    .unwrap();
    let ops = Decoder::new(64, &mul, DecoderOptions::NONE)
        .iter()
        .map(|i| i.mnemonic())
        .collect::<Vec<_>>();
    assert!(ops.contains(&iced_x86::Mnemonic::Vpmuludq));
    assert!(!ops.contains(&iced_x86::Mnemonic::Vpmullq));
    assert!(
        binary(
            Cpu {
                isa: X86Features {
                    dq: true,
                    ..avx512.isa
                },
                ..avx512
            },
            Lane::U64,
            Binary::Mul
        )
        .is_some()
    );
    let reduce_mask = |cpu, action, lane| {
        encode(cpu, action, vector(lane), Type::I32, &[1, 0]).map(|bytes| {
            Decoder::new(64, &bytes, DecoderOptions::NONE)
                .decode()
                .mnemonic()
        })
    };
    use iced_x86::Mnemonic::{Vptestmd, Vptestmq};
    assert_eq!(
        reduce_mask(avx512, Action::AnyTrue, Lane::I64),
        Some(Vptestmd)
    );
    assert_eq!(
        reduce_mask(avx512, Action::AllTrue, Lane::I32),
        Some(Vptestmd)
    );
    assert_eq!(
        reduce_mask(avx512, Action::AllTrue, Lane::I64),
        Some(Vptestmq)
    );
    assert_eq!(
        reduce_mask(
            Cpu {
                isa: X86Features {
                    dq: true,
                    ..avx512.isa
                },
                ..avx512
            },
            Action::AllTrue,
            Lane::I64
        ),
        Some(Vptestmq)
    );
    // Golden register-register forms independently assembled by LLVM 21 llvm-mc.
    for (width, src, dst, from, to, golden) in [
        (
            32,
            2,
            1,
            Lane::F32,
            Lane::I32,
            [0x62, 0xf5, 0x7c, 0x28, 0x6d, 0xca],
        ),
        (
            64,
            11,
            13,
            Lane::F32,
            Lane::U32,
            [0x62, 0x55, 0x7c, 0x48, 0x6c, 0xeb],
        ),
        (
            64,
            2,
            1,
            Lane::F64,
            Lane::I64,
            [0x62, 0xf5, 0xfd, 0x48, 0x6d, 0xca],
        ),
        (
            32,
            11,
            13,
            Lane::F64,
            Lane::U64,
            [0x62, 0x55, 0xfd, 0x28, 0x6c, 0xeb],
        ),
    ] {
        let cpu = Cpu {
            bytes: width,
            isa: X86Features {
                avx10: 2,
                ..avx512.isa
            },
        };
        assert_eq!(
            encode(cpu, Action::Cast, vector(from), vector(to), &[src, dst]).unwrap(),
            golden
        );
        // Earlier targets may synthesize saturation, but cannot execute this
        // AVX10.2-only instruction.
        let older = encode(
            Cpu {
                isa: X86Features {
                    avx10: 1,
                    ..cpu.isa
                },
                ..cpu
            },
            Action::Cast,
            vector(from),
            vector(to),
            &[src, dst],
        );
        assert!(!older.is_some_and(|code| code.windows(golden.len()).any(|inst| inst == golden)));
    }
}

#[test]
fn avx512_splat_reads_integer_gprs_and_floating_point_xmm_sources() {
    use super::packed_code::{body, profiles};
    use crate::{lir::x64::inst::simd::Native, support::numeric::Action};
    use iced_x86::{Decoder, DecoderOptions, Mnemonic, Register};

    let [_, avx2, foundation, full] = profiles();
    let encoding = |cpu, lane: Lane| {
        let op = Native::select(Action::Splat, lane.scalar(), Type::Vector(lane), cpu)?;
        let (bytes, text) = body(op, &[1, 3]);
        Some(
            Decoder::new(64, &bytes[..text], DecoderOptions::NONE)
                .iter()
                .map(|i| (i.mnemonic(), i.op0_register(), i.op1_register()))
                .collect::<Vec<_>>(),
        )
    };
    for (cpu, lane, opcode, source) in [
        (full, Lane::I8, Mnemonic::Vpbroadcastb, Register::ECX),
        (full, Lane::U16, Mnemonic::Vpbroadcastw, Register::ECX),
        (foundation, Lane::I32, Mnemonic::Vpbroadcastd, Register::ECX),
        (foundation, Lane::U64, Mnemonic::Vpbroadcastq, Register::RCX),
        (
            foundation,
            Lane::F32,
            Mnemonic::Vpbroadcastd,
            Register::XMM1,
        ),
    ] {
        assert_eq!(
            encoding(cpu, lane).unwrap(),
            [(opcode, Register::ZMM3, source)]
        );
    }
    assert!(encoding(foundation, Lane::U16).is_none());
    assert_eq!(
        encoding(avx2, Lane::I32).unwrap(),
        [
            (Mnemonic::Movd, Register::XMM3, Register::ECX),
            (Mnemonic::Vpbroadcastd, Register::YMM3, Register::XMM3),
        ]
    );
}

#[test]
fn signed_zero_and_unordered_fp_comparisons_make_per_lane_masks() {
    let w = crate::test_target::width() / 4;
    let ty = Type::Vector(Lane::F32);
    let xs = [-0., f32::NAN, 4., -3.];
    let ys = [0., 7., f32::NAN, -4.];
    let pack = |tab: &[f32; 4]| {
        (0..w / 2)
            .map(|j| {
                u64::from(tab[2 * j % 4].to_bits())
                    | (u64::from(tab[(2 * j + 1) % 4].to_bits()) << 32)
            })
            .collect::<Vec<_>>()
    };
    let args = [pack(&xs), pack(&ys)].concat();
    for op in [
        Binary::Eq,
        Binary::Ne,
        Binary::Lt,
        Binary::Le,
        Binary::Gt,
        Binary::Ge,
    ] {
        let k = kernel(&[ty, ty], Type::Vector(Lane::I32), |v| {
            Op::Bin(op, v[0], v[1])
        });
        let mut out = vec![0; w / 2];
        k.call_packed(&args, &mut out).unwrap();
        for i in 0..w {
            let (a, b) = (xs[i % 4], ys[i % 4]);
            let expected = match op {
                Binary::Eq => a == b,
                Binary::Ne => a != b,
                Binary::Lt => a < b,
                Binary::Le => a <= b,
                Binary::Gt => a > b,
                _ => a >= b,
            };
            assert_eq!(
                (out[i / 2] >> ((i % 2) * 32)) as u32,
                if expected { u32::MAX } else { 0 }
            );
        }
    }
}

#[test]
fn typed_word_abi_rejects_short_buffers_before_execution() {
    let ty = Type::Vector(Lane::U64);
    let f = kernel(&[ty], ty, |v| Op::Copy(v[0]));
    let n = crate::test_target::width() / 8;
    let result = &mut vec![0xdead; n];
    assert_eq!(
        f.call_packed(&[1], result),
        Err(CallError::Signature(SignatureError::ArgumentWords {
            expected: n,
            actual: 1
        }))
    );
    assert_eq!(
        f.call_packed(&vec![0; n], &mut result[..1]),
        Err(CallError::Signature(SignatureError::ResultWords {
            expected: n,
            actual: 1
        }))
    );
    assert_eq!(result, &vec![0xdead; n]);
}

#[test]
fn saturating_float_cast_uses_every_lane() {
    for ty in [Type::Vector(Lane::I32), Type::Vector(Lane::U32)] {
        let k = kernel(&[Type::Vector(Lane::F32)], ty, |v| {
            Op::Cast(Type::Vector(ty.lane().unwrap()), v[0])
        });
        let w = crate::test_target::width() / 8;
        let cases = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -3.75,
            8.875,
            2147483648.,
            4294967296.,
            -0.,
        ];
        for pair in cases.chunks_exact(2) {
            let bits = u64::from(pair[0].to_bits()) | u64::from(pair[1].to_bits()) << 32;
            let mut result = vec![0; w];
            k.call_packed(&vec![bits; w], &mut result).unwrap();
            let convert = |x: f32| {
                if ty == Type::Vector(Lane::I32) {
                    x as i32 as u32
                } else {
                    x as u32
                }
            };
            let expected = u64::from(convert(pair[0])) | u64::from(convert(pair[1])) << 32;
            assert!(result.iter().all(|&x| x == expected));
        }
    }
}

#[test]
fn mixed_scalar_mask_selects_full_width_values() {
    for ty in [Type::I8, Type::I64, Type::I128, Type::F32, Type::F64] {
        let k = kernel(&[Type::I32, ty, ty], ty, |v| {
            Op::Ternary(Ternary::Select, v[0], v[1], v[2])
        });
        let size = ty.bytes(crate::test_target::width());
        let words = size.div_ceil(8);
        for cond in [0, 1, 0x80000000] {
            let mut args = vec![cond];
            args.extend(vec![0x1111; words]);
            args.extend(vec![0x7777; words]);
            let mut out = vec![0; words];
            k.call_packed(&args, &mut out).unwrap();
            let mask = if size == 1 { 0xff } else { u64::MAX };
            assert!(
                out.iter()
                    .all(|&x| x == (if cond != 0 { 0x1111 } else { 0x7777 }) & mask)
            );
        }
    }
}
