//! Packed FP results are compared bit-for-bit with independent Rust operations.
//! Inputs contain a different lane pattern on each invocation, including NaN payloads.
mod conversions;
use super::super::packed_code::{body, executable, host_supports, profiles, supports};
use super::{Binary, Function, Lane, Op, ScalarKernel, Simd, Terminator, Type, Unary};
use crate::{
    lir::x64::{Features, inst::simd::Native},
    numbers::{Action, Math},
};
use iced_x86::{Decoder, DecoderOptions, EncodingKind, Instruction, Mnemonic};
use std::num::FpCategory;

fn arity(op: Action) -> usize {
    if matches!(op, Action::Binary(_) | Action::Narrow) {
        2
    } else {
        1
    }
}

fn function(from: Lane, to: Lane, op: Action) -> Function {
    let mut f = Function::new(&vec![Type::Vector(from); arity(op)], Type::Vector(to));
    let block = f.entry();
    let args = f.params(block);
    let simd = match op {
        Action::Unary(op) => Simd::Unary(op, args[0]),
        Action::Binary(op) => Simd::Bin(op, args[0], args[1]),
        Action::Cast => Simd::Cast(to, args[0]),
        Action::Widen(high) => Simd::Widen {
            to,
            high,
            value: args[0],
        },
        Action::Narrow => Simd::Narrow(to, args[0], args[1]),
        _ => unreachable!(),
    };
    let result = f.append(block, Op::Simd(simd)).unwrap();
    f.set_terminator(block, Terminator::Return(result));
    f
}

fn check(
    from: Lane,
    to: Lane,
    op: Action,
    inputs: &[(u64, u64)],
    expected: impl Fn(u64, u64) -> u64,
) {
    assert_eq!(from.bytes(), to.bytes());
    let size = from.bytes();
    run_profiles(from, to, op, |width, call| {
        let count = width / size;
        for offset in (0..inputs.len()).step_by(count) {
            let mut slots = [[0u8; 64]; 3];
            let mut reference = [0u8; 64];
            for i in 0..count {
                let (a, b) = inputs[(offset + i) % inputs.len()];
                let range = i * size..(i + 1) * size;
                slots[0][range.clone()].copy_from_slice(&a.to_le_bytes()[..size]);
                slots[1][range.clone()].copy_from_slice(&b.to_le_bytes()[..size]);
                reference[range].copy_from_slice(&expected(a, b).to_le_bytes()[..size]);
            }
            call(&mut slots);
            assert_eq!(
                slots[2], reference,
                "{from:?} {op:?} -> {to:?}, width={width}, case={offset}"
            );
        }
    });
}

type Slots = [[u8; 64]; 3];
fn registers(op: Native, arity: usize) -> Vec<usize> {
    let mut regs = if arity == 2 {
        vec![1, 2, 3]
    } else {
        vec![1, 3]
    };
    if op.tied() {
        regs[arity] = regs[0];
    }
    regs
}

fn decoded(cpu: Features, spec: Math, bytes: &[u8]) -> Vec<Instruction> {
    Decoder::new(64, bytes, DecoderOptions::NONE)
        .into_iter()
        .inspect(|i| {
            assert!(!i.is_invalid(), "{spec:?} for {cpu:?}");
            for operand in 0..i.op_count() {
                let reg = i.op_register(operand);
                assert!(
                    !reg.is_vector_register() || reg.size() <= cpu.bytes,
                    "{spec:?} for {cpu:?}: {i:?} exceeds the supported vector width"
                );
            }
            for &feature in i.cpuid_features() {
                assert!(
                    supports(cpu, feature),
                    "{spec:?} for {cpu:?}: {i:?} requires {feature:?}"
                );
            }
        })
        .collect()
}

fn avx10_1_profiles() -> [Features; 2] {
    let features = profiles()[3];
    [16, 32].map(|bytes| Features {
        bytes,
        avx10: 1,
        ..features
    })
}

fn run_profiles(from: Lane, to: Lane, op: Action, run_cases: impl Fn(usize, &dyn Fn(&mut Slots))) {
    let host = Features::host();
    let arity = arity(op);
    let k = ScalarKernel::compile(&function(from, to, op)).unwrap();
    run_cases(host.bytes, &|slots| {
        let args: Vec<_> = slots[..arity]
            .iter()
            .flat_map(|slot| slot[..host.bytes].chunks_exact(8))
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut out = vec![0; host.bytes / 8];
        k.call_words(&args, &mut out).unwrap();
        for (p, word) in slots[2].chunks_exact_mut(8).zip(out) {
            p.copy_from_slice(&word.to_le_bytes());
        }
    });

    let spec = Math {
        action: op,
        input: Type::Vector(from),
        output: Type::Vector(to),
    };
    let [sse2, avx2, foundation, extended] = profiles();
    let avx1 = Features { bytes: 32, ..sse2 };
    let mut targets = vec![sse2, avx1, avx2, foundation, extended];
    if matches!(op, Action::Cast) {
        targets.extend(avx10_1_profiles());
    }
    if matches!(from, Lane::F16) || matches!(to, Lane::F16) {
        targets.push(Features { f16c: true, ..avx1 });
    }
    if matches!(from, Lane::BF16) || matches!(to, Lane::BF16) {
        targets.push(Features {
            bf16: true,
            ..extended
        });
    }
    for cpu in targets {
        let Some(native) = Native::select(spec, cpu) else {
            continue;
        };
        let regs = registers(native, arity);
        let (bytes, text_len) = body(native, &regs);
        let instructions = decoded(cpu, spec, &bytes[..text_len]);
        let available = instructions
            .iter()
            .flat_map(Instruction::cpuid_features)
            .copied()
            .all(host_supports);
        if cpu.bytes <= host.bytes && available {
            let code = executable(&bytes, cpu.bytes, &regs);
            // SAFETY: executable() preserves the host ABI and touches only the three
            // live 64-byte slots. Every selected instruction was checked against the host.
            unsafe {
                crate::jit::packed_test::with_kernel(&code, |call| run_cases(cpu.bytes, call))
            };
        }
    }
}

fn classes(e: Lane) -> Vec<u64> {
    let (fraction, sign, inf) = match e {
        Lane::F16 => (10, 1 << 15, 0x7c00),
        Lane::BF16 => (7, 1 << 15, 0x7f80),
        Lane::F32 => (23, 1 << 31, u64::from(f32::INFINITY.to_bits())),
        Lane::F64 => (52, 1 << 63, f64::INFINITY.to_bits()),
        _ => unreachable!(),
    };
    let normal = 1 << fraction;
    let quiet = normal >> 1;
    [
        0,
        1,
        normal - 1,
        normal,
        inf - 1,
        inf,
        inf + 0x15,
        inf + quiet + 0x17,
    ]
    .into_iter()
    .flat_map(|bits| [bits, bits | sign])
    .collect()
}

#[test]
fn sign_and_classification_preserve_snan_payload_and_subnormal_boundaries() {
    for (e, mask, sign) in [
        (Lane::F16, Lane::I16, 1 << 15),
        (Lane::BF16, Lane::I16, 1 << 15),
        (Lane::F32, Lane::I32, 1 << 31),
        (Lane::F64, Lane::I64, 1 << 63),
    ] {
        let inputs = classes(e);
        let pairs: Vec<_> = inputs
            .iter()
            .flat_map(|&a| inputs.iter().map(move |&b| (a, b)))
            .collect();
        let single: Vec<_> = inputs.iter().map(|&a| (a, 0)).collect();
        check(e, e, Action::Unary(Unary::Neg), &single, |a, _| a ^ sign);
        check(e, e, Action::Unary(Unary::Abs), &single, |a, _| a & !sign);
        check(e, e, Action::Binary(Binary::CopySign), &pairs, |a, b| {
            (a & !sign) | (b & sign)
        });
        let category = |a: u64| match e {
            Lane::F32 => f32::from_bits(a as u32).classify(),
            Lane::F64 => f64::from_bits(a).classify(),
            _ => {
                let (exp, fraction) = if e == Lane::F16 {
                    (0x7c00, 0x03ff)
                } else {
                    (0x7f80, 0x007f)
                };
                match (a & exp, a & fraction) {
                    (0, 0) => FpCategory::Zero,
                    (0, _) => FpCategory::Subnormal,
                    (x, 0) if x == exp => FpCategory::Infinite,
                    (x, _) if x == exp => FpCategory::Nan,
                    _ => FpCategory::Normal,
                }
            }
        };
        let mask_bits = |yes| if yes { u64::MAX } else { 0 };
        for op in [
            Unary::IsNan,
            Unary::IsInfinite,
            Unary::IsFinite,
            Unary::IsNormal,
            Unary::IsSubnormal,
            Unary::SignBit,
        ] {
            check(e, mask, Action::Unary(op), &single, |a, _| {
                mask_bits(match op {
                    Unary::IsNan => category(a) == FpCategory::Nan,
                    Unary::IsInfinite => category(a) == FpCategory::Infinite,
                    Unary::IsFinite => {
                        !matches!(category(a), FpCategory::Infinite | FpCategory::Nan)
                    }
                    Unary::IsNormal => category(a) == FpCategory::Normal,
                    Unary::IsSubnormal => category(a) == FpCategory::Subnormal,
                    Unary::SignBit => a & sign != 0,
                    _ => unreachable!(),
                })
            });
        }
        for op in [Binary::Ordered, Binary::Unordered] {
            check(e, mask, Action::Binary(op), &pairs, |a, b| {
                let unordered = category(a) == FpCategory::Nan || category(b) == FpCategory::Nan;
                mask_bits(unordered == (op == Binary::Unordered))
            });
        }
    }
}

#[test]
fn packed_reciprocal_and_fraction_obey_ieee_arithmetic() {
    for e in [Lane::F32, Lane::F64] {
        let raw = |n: f64| {
            if e == Lane::F32 {
                u64::from((n as f32).to_bits())
            } else {
                n.to_bits()
            }
        };
        let finite: Vec<_> = [-3., -2.75, -1., -0., 0., 1., 2.75, 3., 7.]
            .map(|n| (raw(n), 0))
            .into();
        for op in [Unary::Recip, Unary::Fract] {
            let mut inputs = finite.clone();
            inputs.extend([(1, 0), (raw(-0.) | 1, 0)]);
            if op == Unary::Recip {
                inputs.extend([(raw(f64::INFINITY), 0), (raw(f64::NEG_INFINITY), 0)]);
            }
            check(e, e, Action::Unary(op), &inputs, |a, _| {
                if e == Lane::F32 {
                    let x = f32::from_bits(a as u32);
                    u64::from(
                        if op == Unary::Recip {
                            x.recip()
                        } else {
                            x.fract()
                        }
                        .to_bits(),
                    )
                } else {
                    let x = f64::from_bits(a);
                    if op == Unary::Recip {
                        x.recip()
                    } else {
                        x.fract()
                    }
                    .to_bits()
                }
            });
        }
    }
}

#[test]
fn packed_min_max_keep_the_numeric_operand_on_either_side_of_nan() {
    for e in [Lane::F32, Lane::F64] {
        let raw = |n: f64| {
            if e == Lane::F32 {
                u64::from((n as f32).to_bits())
            } else {
                n.to_bits()
            }
        };
        let value = |n: u64| {
            if e == Lane::F32 {
                f64::from(f32::from_bits(n as u32))
            } else {
                f64::from_bits(n)
            }
        };
        let nan = raw(f64::NAN) | 0x123;
        let mut inputs = Vec::new();
        for x in [0., -0., 3.5, -8.25, f64::INFINITY, f64::NEG_INFINITY] {
            let x = raw(x);
            inputs.extend([(x, nan), (nan, x), (x, x)]);
        }
        inputs.extend([(raw(-3.0), raw(4.0)), (raw(4.0), raw(-3.0))]);
        for op in [Binary::Min, Binary::Max] {
            check(e, e, Action::Binary(op), &inputs, |a, b| {
                if a == nan {
                    return b;
                }
                if b == nan {
                    return a;
                }
                raw(if op == Binary::Min {
                    value(a).min(value(b))
                } else {
                    value(a).max(value(b))
                })
            });
        }
    }
}

#[test]
fn packed_rounding_distinguishes_half_even_from_half_away_and_preserves_negative_zero() {
    let cases = [
        -2.75,
        -2.5,
        -1.5,
        -0.5,
        -0.125,
        -0.0,
        0.0,
        0.125,
        0.5,
        1.5,
        2.5,
        2.75,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for e in [Lane::F32, Lane::F64] {
        let raw = |x: f64| {
            if e == Lane::F32 {
                u64::from((x as f32).to_bits())
            } else {
                x.to_bits()
            }
        };
        let mut inputs: Vec<_> = cases.map(|x| (raw(x), 0)).into();
        inputs.extend([(1, 0), (1 | raw(-0.0), 0)]);
        // At this scale the upper neighbor is already an odd integer. Adding
        // 0.5 before truncation would incorrectly round it to an even integer.
        let unit_spacing = 2.0_f64.powi(if e == Lane::F32 { 23 } else { 52 });
        for n in [0.5, 1.5, 2.5, 3.5, unit_spacing] {
            for x in [raw(n) - 1, raw(n) + 1] {
                inputs.push((x, 0));
                inputs.push((x | raw(-0.0), 0));
            }
        }
        for op in [
            Unary::Floor,
            Unary::Ceil,
            Unary::Trunc,
            Unary::Nearest,
            Unary::Round,
        ] {
            check(e, e, Action::Unary(op), &inputs, |bits, _| {
                let x = if e == Lane::F32 {
                    f64::from(f32::from_bits(bits as u32))
                } else {
                    f64::from_bits(bits)
                };
                raw(match op {
                    Unary::Floor => x.floor(),
                    Unary::Ceil => x.ceil(),
                    Unary::Trunc => x.trunc(),
                    Unary::Nearest => x.round_ties_even(),
                    Unary::Round => x.round(),
                    _ => unreachable!(),
                })
            });
        }
    }
}

#[test]
fn packed_casts_saturate_before_overflow_and_round_halfway_to_even() {
    let two31 = (2.0f32.powi(31)).to_bits();
    let mut fp: Vec<_> = classes(Lane::F32).into_iter().map(|x| (x, 0)).collect();
    for bits in [
        two31 - 1,
        two31,
        two31 + 1,
        0.75f32.to_bits(),
        1.75f32.to_bits(),
    ] {
        for x in [bits, bits | (1 << 31)] {
            fp.push((u64::from(x), 0));
        }
    }
    check(Lane::F32, Lane::I32, Action::Cast, &fp, |a, _| {
        u64::from(f32::from_bits(a as u32) as i32 as u32)
    });
    let ints: Vec<_> = [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX]
        .into_iter()
        .chain((-3..=5).flat_map(|i| {
            let x = (1 << 24) + i;
            [x, -x]
        }))
        .map(|x| (u64::from(x as u32), 0))
        .collect();
    check(Lane::I32, Lane::F32, Action::Cast, &ints, |x, _| {
        u64::from((x as i32 as f32).to_bits())
    });
}

#[test]
fn signed_float_casts_select_packed_conversions_at_each_x64_width() {
    // The architectural baseline does not include AVX2, SSE4.1, or AVX-512DQ.
    let baseline = profiles()[0];
    let ordinary = [16, 32, 64].map(|bytes| Features { bytes, ..baseline });
    for cpu in ordinary.into_iter().chain(avx10_1_profiles()) {
        let bytes = cpu.bytes;
        for (from, to) in [(Lane::F32, Lane::I32), (Lane::I32, Lane::F32)] {
            let spec = Math {
                action: Action::Cast,
                input: Type::Vector(from),
                output: Type::Vector(to),
            };
            let op = Native::select(spec, cpu)
                .unwrap_or_else(|| panic!("missing packed {from:?}->{to:?} at width {bytes}"));
            let (code, text_len) = body(op, &[1, 2]);
            let instructions = decoded(cpu, spec, &code[..text_len]);
            let conversion = match (from, bytes) {
                (Lane::F32, 16) => Mnemonic::Cvttps2dq,
                (Lane::F32, _) => Mnemonic::Vcvttps2dq,
                (_, 16) => Mnemonic::Cvtdq2ps,
                _ => Mnemonic::Vcvtdq2ps,
            };
            assert!(
                instructions.iter().any(|i| i.mnemonic() == conversion
                    && i.op0_register().size() == bytes
                    && i.encoding()
                        == match bytes {
                            16 => EncodingKind::Legacy,
                            32 => EncodingKind::VEX,
                            _ => EncodingKind::EVEX,
                        }),
                "{from:?}->{to:?} ({bytes} bytes): {instructions:?}"
            );
        }
    }
}
