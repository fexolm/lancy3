use super::*;
use Lane::{BF16, F16, F32, F64, I64, U32, U64};
use splice_target::X86Features;

#[test]
fn unsigned_and_qword_casts_saturate_at_ieee_neighbors() {
    let mut singles: Vec<_> = classes(F32).into_iter().map(|x| (x, 0)).collect();
    for edge in [1.0f32, 2f32.powi(31), 2f32.powi(32)] {
        for raw in [edge.to_bits() - 1, edge.to_bits(), edge.to_bits() + 1] {
            singles.extend([raw, raw | (1 << 31)].map(|x| (u64::from(x), 0)));
        }
    }
    check(F32, U32, Action::Cast, &singles, |x, _| {
        u64::from(f32::from_bits(x as u32) as u32)
    });
    let ints: Vec<_> = [0, 1, u32::MAX - 1, u32::MAX]
        .into_iter()
        .chain((1 << 24) - 3..=(1 << 24) + 5)
        .chain((1 << 31) - 1..=(1 << 31) + 1)
        .map(|x| (u64::from(x), 0))
        .collect();
    check(U32, F32, Action::Cast, &ints, |x, _| {
        u64::from((x as u32 as f32).to_bits())
    });

    let mut doubles: Vec<_> = classes(F64).into_iter().map(|x| (x, 0)).collect();
    for edge in [1.0f64, 1.75, 2f64.powi(63), 2f64.powi(64)] {
        for raw in [edge.to_bits() - 1, edge.to_bits(), edge.to_bits() + 1] {
            doubles.extend([raw, raw | (1 << 63)].map(|x| (x, 0)));
        }
    }
    check(F64, I64, Action::Cast, &doubles, |x, _| {
        f64::from_bits(x) as i64 as u64
    });
    check(F64, U64, Action::Cast, &doubles, |x, _| {
        f64::from_bits(x) as u64
    });
    let longs: Vec<_> = [
        0,
        1,
        i64::MAX as u64,
        1 << 63,
        (1 << 63) + 1,
        u64::MAX - 1,
        u64::MAX,
    ]
    .into_iter()
    .chain(((1u64 << 53) - 3..=(1 << 53) + 5).flat_map(|x| [x, x.wrapping_neg()]))
    .map(|x| (x, 0))
    .collect();
    check(I64, F64, Action::Cast, &longs, |x, _| {
        (x as i64 as f64).to_bits()
    });
    check(U64, F64, Action::Cast, &longs, |x, _| (x as f64).to_bits());
}

// Check the bytes in API lane order, allowing two inputs to narrow and either
// half of a single input to widen. Each consecutive lane has its own test case.
fn check_width(from: Lane, to: Lane, action: Action, cases: &[(u64, u64)]) {
    run_profiles(from, to, action, |width, call| {
        let src_count = width / from.bytes();
        let dst_count = width / to.bytes();
        let start = if matches!(action, Action::Widen(true)) {
            dst_count
        } else {
            0
        };
        for offset in (0..cases.len()).step_by(dst_count) {
            let mut slots = [[0u8; 64]; 3];
            for (slot, data) in slots[..arity(action)].iter_mut().enumerate() {
                for (i, lane) in data[..width].chunks_exact_mut(from.bytes()).enumerate() {
                    let (bits, _) = cases[(offset + slot * src_count + i) % cases.len()];
                    lane.copy_from_slice(&bits.to_le_bytes()[..from.bytes()]);
                }
            }
            call(&mut slots);
            for (i, lane) in slots[2][..width].chunks_exact(to.bytes()).enumerate() {
                let (src, expected) = cases[(offset + start + i) % cases.len()];
                assert_eq!(
                    lane,
                    &expected.to_le_bytes()[..to.bytes()],
                    "{from:?} {action:?} -> {to:?}, width={width}, lane={i}, input={src:#x}"
                );
            }
        }
    });
}

fn both_signs(
    src: Lane,
    dst: Lane,
    cases: impl IntoIterator<Item = (u64, u64)>,
) -> Vec<(u64, u64)> {
    let src_sign = 1u64 << (8 * src.bytes() - 1);
    let dst_sign = 1u64 << (8 * dst.bytes() - 1);
    cases
        .into_iter()
        .flat_map(|(x, y)| [(x, y), (x | src_sign, y | dst_sign)])
        .collect()
}

#[test]
fn packed_half_widen_preserves_subnormals_and_canonicalizes_nan_sign() {
    let h = [
        (0x0000, 0.0),
        (0x0001, 2f32.powi(-24)),
        (0x0002, 2f32.powi(-23)),
        (0x03ff, 1023.0 * 2f32.powi(-24)),
        (0x0400, 2f32.powi(-14)),
        (0x3c00, 1.0),
        (0x3c01, 1.0 + 1.0 / 1024.0),
        (0x7bff, 65504.0),
        (0x7c00, f32::INFINITY),
    ];
    let mut half: Vec<_> = h
        .into_iter()
        .map(|(x, y)| (x, u64::from(y.to_bits())))
        .collect();
    half.extend([0x7c01, 0x7d37, 0x7e11, 0x7fff].map(|x| (x, 0x7fc0_0000)));

    let mut bfloat: Vec<_> = [
        0x0000, 0x0001, 0x0002, 0x007f, 0x0080, 0x3f80, 0x3f81, 0x7f7f, 0x7f80,
    ]
    .map(|x| (x, x << 16))
    .into();
    bfloat.extend([0x7f81, 0x7fbf, 0x7fc5, 0x7fff].map(|x| (x, 0x7fc0_0000)));
    for (half, pairs) in [(F16, half), (BF16, bfloat)] {
        let pairs = both_signs(half, F32, pairs);
        for high in [false, true] {
            check_width(half, F32, Action::Widen(high), &pairs);
        }
    }
}

#[test]
fn packed_half_narrow_concatenates_both_sources_and_rounds_subnormals() {
    for (half, inf, quiet, midpoints) in [
        (
            F16,
            0x7c00,
            0x7e00,
            [
                (2f32.powi(-25), 0, 0, 1),
                (3.0 * 2f32.powi(-25), 1, 2, 2),
                (2f32.powi(-14) - 2f32.powi(-25), 0x3ff, 0x400, 0x400),
                (1.0 + 2f32.powi(-11), 0x3c00, 0x3c00, 0x3c01),
                (1.0 + 3.0 * 2f32.powi(-11), 0x3c01, 0x3c02, 0x3c02),
                (65520.0, 0x7bff, 0x7c00, 0x7c00),
            ],
        ),
        (
            BF16,
            0x7f80,
            0x7fc0,
            [
                (f32::from_bits(0x0000_8000), 0, 0, 1),
                (f32::from_bits(0x0001_8000), 1, 2, 2),
                (f32::from_bits(0x007f_8000), 0x7f, 0x80, 0x80),
                (1.0 + 2f32.powi(-8), 0x3f80, 0x3f80, 0x3f81),
                (1.0 + 3.0 * 2f32.powi(-8), 0x3f81, 0x3f82, 0x3f82),
                (f32::from_bits(0x7f7f_8000), 0x7f7f, 0x7f80, 0x7f80),
            ],
        ),
    ] {
        let mut pairs = vec![(0, 0), (1, 0), (0x7f80_0000, inf)];
        for nan in [0x7f80_0001, 0x7f92_3456, 0x7fc4_5678, 0x7fff_ffff] {
            pairs.push((nan, quiet));
        }
        for (midpoint, below, tie, above) in midpoints {
            let bits = u64::from(midpoint.to_bits());
            pairs.extend([(bits - 1, below), (bits, tie), (bits + 1, above)]);
        }
        check_width(F32, half, Action::Narrow, &both_signs(F32, half, pairs));
    }
}

#[test]
fn packed_float_and_double_widen_and_narrow_keep_lane_order() {
    let wide: Vec<_> = [
        0,
        1,
        2,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0001,
        0x7f7f_ffff,
        0x7f80_0000,
    ]
    .map(|x| (u64::from(x), (f32::from_bits(x) as f64).to_bits()))
    .into();
    let wide = both_signs(F32, F64, wide);
    for high in [false, true] {
        check_width(F32, F64, Action::Widen(high), &wide);
    }
    let mut narrow = vec![0, 1, f64::MAX.to_bits(), f64::INFINITY.to_bits()];
    for x in [
        2f64.powi(-150),
        2f64.powi(-126) - 2f64.powi(-150),
        1.0 + 2f64.powi(-24),
        1.0 + 3.0 * 2f64.powi(-24),
        2f64.powi(128) - 2f64.powi(103),
    ] {
        narrow.extend([x.to_bits() - 1, x.to_bits(), x.to_bits() + 1]);
    }
    let pairs = narrow
        .into_iter()
        .map(|bits| (bits, u64::from((f64::from_bits(bits) as f32).to_bits())));
    check_width(F64, F32, Action::Narrow, &both_signs(F64, F32, pairs));
}

fn decoded_conversion(cpu: Cpu, from: Lane, to: Lane, action: Action) -> Vec<Instruction> {
    let Some(op) = Native::select(action, Type::Vector(from), Type::Vector(to), cpu) else {
        return Vec::new();
    };
    let (bytes, len) = body(op, &registers(op, arity(action)));
    decoded(cpu, op, &bytes[..len])
}

#[test]
fn integer_conversions_respect_avx512_subsets_and_avx10_vector_length() {
    use Mnemonic::*;
    for cpu in profiles().into_iter().chain(avx10_1_profiles()) {
        for (from, to, mnemonic, needs_dq) in [
            (F32, U32, Vcvttps2udq, false),
            (U32, F32, Vcvtudq2ps, false),
            (F64, I64, Vcvttpd2qq, true),
            (F64, U64, Vcvttpd2uqq, true),
            (I64, F64, Vcvtqq2pd, true),
            (U64, F64, Vcvtuqq2pd, true),
        ] {
            let instructions = decoded_conversion(cpu, from, to, Action::Cast);
            let packed = instructions.iter().any(|i| {
                i.mnemonic() == mnemonic
                    && i.encoding() == EncodingKind::EVEX
                    && i.op0_register().size() == cpu.bytes
                    && i.op1_register().size() == cpu.bytes
            });
            assert_eq!(
                packed,
                (cpu.bytes == 64 || cpu.isa.avx10 == 1) && (!needs_dq || cpu.isa.dq),
                "{from:?} -> {to:?} on {cpu:?}: {instructions:?}"
            );
        }
    }
}

#[test]
fn widen_and_narrow_use_legal_packed_instructions_at_the_available_width() {
    let [sse2, avx2, foundation, extended] = profiles();
    for cpu in [
        sse2,
        Cpu {
            isa: X86Features {
                avx: true,
                f16c: true,
                ..sse2.isa
            },
            ..sse2
        },
        avx2,
        foundation,
        extended,
    ] {
        for narrow in [F16, BF16, F32] {
            let wide = if narrow == F32 { F64 } else { F32 };
            for (from, to, action, mnemonic) in [
                (
                    narrow,
                    wide,
                    Action::Widen(false),
                    match narrow {
                        F16 => Some(Mnemonic::Vcvtph2ps),
                        BF16 => None,
                        _ if cpu.bytes == 16 => Some(Mnemonic::Cvtps2pd),
                        _ => Some(Mnemonic::Vcvtps2pd),
                    },
                ),
                (
                    wide,
                    narrow,
                    Action::Narrow,
                    match narrow {
                        F16 => Some(Mnemonic::Vcvtps2ph),
                        BF16 => None,
                        _ if cpu.bytes == 16 => Some(Mnemonic::Cvtpd2ps),
                        _ => Some(Mnemonic::Vcvtpd2ps),
                    },
                ),
            ] {
                let supported = match narrow {
                    F16 => cpu.isa.f16c || cpu.bytes == 64,
                    BF16 => cpu.bytes == 16 || cpu.isa.avx2,
                    _ => true,
                };
                let code = decoded_conversion(cpu, from, to, action);
                if supported {
                    assert!(
                        code.iter().any(|i| {
                            mnemonic.is_none_or(|m| m == i.mnemonic())
                                && (0..i.op_count()).any(|n| i.op_register(n).size() == cpu.bytes)
                        }),
                        "{from:?} {action:?} -> {to:?} on {cpu:?}: {code:?}"
                    );
                }
            }
        }
    }
}
