use super::{Function, LANES, Lane, Math, Op, ScalarKernel, Terminator, Ternary};
use crate::numbers::Action;

fn scalar_cast(from: Lane, to: Lane) -> Function {
    let mut f = Function::new(&[from.scalar()], to.scalar());
    let block = f.entry();
    let v = f
        .append(block, Op::Cast(to.scalar(), f.params(block)[0]))
        .unwrap();
    f.set_terminator(block, Terminator::Return(v));
    f
}

fn words(n: u128) -> [u64; 2] {
    [n as u64, (n >> 64) as u64]
}

fn float_layout(lane: Lane) -> (u32, u32) {
    match lane {
        Lane::F16 => (5, 10),
        Lane::BF16 => (8, 7),
        Lane::F32 => (8, 23),
        Lane::F64 => (11, 52),
        _ => unreachable!("floating-point lane"),
    }
}

fn is_nan(lane: Lane, n: u128) -> bool {
    if !lane.float() {
        return false;
    }
    let (exp, frac) = float_layout(lane);
    let inf = ((1 << exp) - 1) << frac;
    n & ((1 << (exp + frac)) - 1) > inf
}

fn inputs(lane: Lane) -> Vec<u128> {
    let size = lane.bytes() * 8;
    let mask = u128::MAX >> (128 - size);
    let mut values = vec![0, 1, mask, 1 << (size - 1), (1 << (size - 1)) - 1];
    // Mantissa cutoffs, both sides of word boundaries, signed extrema, and the
    // tie with exactly 64 discarded bits when casting 128-bit integers to floats.
    for power in [
        7, 8, 15, 16, 23, 24, 31, 32, 52, 53, 63, 64, 65, 71, 87, 116, 127,
    ] {
        let p = 1u128 << power;
        for center in [p, p | (1u128 << 63)] {
            for n in [center - 1, center, center + 1] {
                values.extend([n & mask, n.wrapping_neg() & mask]);
            }
        }
    }
    if lane.float() {
        let (exp, frac) = float_layout(lane);
        let bias = (1i32 << (exp - 1)) - 1;
        let sign = 1u128 << (exp + frac);
        let inf = ((1u128 << exp) - 1) << frac;
        let mut centers = vec![1, 1 << frac, inf, inf + (1 << (frac - 1))];
        // Signed and unsigned destinations saturate on different sides of their cutoffs.
        for e in [
            -134, -133, -127, -126, -25, -24, -14, -1, 0, 7, 8, 15, 16, 31, 32, 63, 64, 116, 127,
            128,
        ] {
            if (1..2 * bias + 1).contains(&(e + bias)) {
                centers.push(((e + bias) as u128) << frac);
            }
        }
        for center in centers {
            for n in [center - 1, center, center + 1] {
                values.extend([n, sign | n]);
            }
        }
    }
    let mut random = 0xcab4_8321_0913_bc40_ef28_0ff3_f103_e88du128;
    for _ in 0..192 {
        random ^= random << 31;
        random ^= random >> 39;
        random ^= random << 5;
        values.push(random & mask);
    }
    values.sort_unstable();
    values.dedup();
    values
}

#[test]
fn all_scalar_casts_emit_call_free_machine_code() {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};
    for from in LANES {
        for to in LANES {
            let f = scalar_cast(from, to);
            let (module, ids) = f.verified_module().unwrap();
            let code = crate::lir::x64::emit::emit(module, &ids).unwrap();
            for inst in Decoder::new(64, &code.bytes[..code.text_len], DecoderOptions::NONE) {
                assert!(!inst.is_invalid(), "{from:?} -> {to:?} at {}", inst.ip());
                assert_ne!(
                    inst.mnemonic(),
                    Mnemonic::Call,
                    "{from:?} -> {to:?} at {}",
                    inst.ip()
                );
            }
        }
    }
}

#[test]
fn scalar_casts_match_reference_across_ieee_and_word_boundaries() {
    for from in LANES {
        let cases = inputs(from);
        for to in LANES {
            let kernel = ScalarKernel::compile(&scalar_cast(from, to)).unwrap();
            let math = Math {
                action: Action::Cast,
                input: from.scalar(),
                output: to.scalar(),
            };
            for &bits in &cases {
                let mut args = [[0; 64]; 3];
                args[0][..16].copy_from_slice(&bits.to_le_bytes());
                let reference = math.eval(&args, ScalarKernel::vector_bytes()).unwrap();
                let mask = u128::MAX >> (128 - to.bytes() * 8);
                let reference = u128::from_le_bytes(reference[..16].try_into().unwrap()) & mask;
                let mut out = [0u64; 2];
                kernel
                    .call_words(
                        &words(bits)[..from.bytes().div_ceil(8)],
                        &mut out[..to.bytes().div_ceil(8)],
                    )
                    .unwrap();
                let actual = (u128::from(out[0]) | u128::from(out[1]) << 64) & mask;
                assert!(
                    actual == reference
                        || (from != to && is_nan(to, actual) && is_nan(to, reference)),
                    "{from:?}({bits:#x}) -> {to:?}: {actual:#x}, expected {reference:#x}"
                );
            }
        }
    }
}

#[test]
fn half_cast_rounds_both_sides_of_normal_subnormal_and_infinity_midpoints() {
    for (to, min_sub, min_normal, frac, inf) in [
        (Lane::F16, -24, -14, 10, 0x7c00u64),
        (Lane::BF16, -133, -126, 7, 0x7f80),
    ] {
        let kernel = ScalarKernel::compile(&scalar_cast(Lane::F64, to)).unwrap();
        let bias = 1 - min_normal;
        let half_sub = 2f64.powi(min_sub - 1);
        for (center, lower, tied, upper) in [
            (half_sub, 0, 0, 1),
            (3.0 * half_sub, 1, 2, 2),
            (
                2f64.powi(min_normal) - half_sub,
                (1 << frac) - 1,
                1 << frac,
                1 << frac,
            ),
            (
                2f64.powi(bias + 1) - 2f64.powi(bias - frac - 1),
                inf - 1,
                inf,
                inf,
            ),
        ] {
            for (x, val) in [
                (center.next_down(), lower),
                (center, tied),
                (center.next_up(), upper),
            ] {
                for sign in [false, true] {
                    let bits = if sign { (-x).to_bits() } else { x.to_bits() };
                    let mut out = [0];
                    kernel.call_words(&[bits], &mut out).unwrap();
                    assert_eq!(
                        out[0] & 0xffff,
                        val | (u64::from(sign) << 15),
                        "{x} as {to:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn half_fma_preserves_guard_bits_and_signed_zero() {
    for (lane, one, one_plus, one_minus, one_half, rounded_low, rounded_high, product_error, inf) in [
        (
            Lane::F16,
            0x3c00u64,
            0x3c01,
            0x3bfe,
            0x3e00,
            0x3e01,
            0x3e02,
            0x8010,
            0x7c00,
        ),
        (
            Lane::BF16,
            0x3f80,
            0x3f81,
            0x3f7e,
            0x3fc0,
            0x3fc1,
            0x3fc2,
            0xb880,
            0x7f80,
        ),
    ] {
        let ty = lane.scalar();
        let mut f = Function::new(&[ty; 3], ty);
        let block = f.entry();
        let [a, b, c] = f.params(block).try_into().unwrap();
        let v = f
            .append(block, Op::Ternary(Ternary::MulAdd, a, b, c))
            .unwrap();
        f.set_terminator(block, Terminator::Return(v));
        let kernel = ScalarKernel::compile(&f).unwrap();
        // The exact product lies on a target midpoint. A minimum subnormal
        // addend changes its rounding direction, even if f64 addition loses it.
        for (a, b, c, expected) in [
            (one_half, one_plus, 0, rounded_high),
            (one_half, one_plus, 1, rounded_high),
            (one_half, one_plus, 0x8001, rounded_low),
            (one_plus, one_minus, one | 0x8000, product_error),
            (0x8000, one, 0x8000, 0x8000),
            (0, one, 0x8000, 0),
            (inf, one, one, inf),
            (0, inf, one, inf | 1),
            (inf | 1, one, one, inf | 1),
        ] {
            let mut out = [0];
            kernel.call_words(&[a, b, c], &mut out).unwrap();
            let actual = out[0] & 0xffff;
            assert!(
                actual == expected || is_nan(lane, actual.into()) && is_nan(lane, expected.into()),
                "{lane:?} fma({a:#x}, {b:#x}, {c:#x}): {actual:#x}, expected {expected:#x}"
            );
        }
    }
}
