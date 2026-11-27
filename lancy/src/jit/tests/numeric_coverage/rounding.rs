use super::{Binary, Function, Kernel, Lane, Op, Terminator, Unary};
use crate::support::numeric::{Action, evaluate};

fn layout(lane: Lane) -> (u32, u64) {
    match lane {
        Lane::F16 => (10, 15),
        Lane::BF16 => (7, 127),
        Lane::F32 => (23, 127),
        Lane::F64 => (52, 1023),
        _ => unreachable!(),
    }
}

#[test]
fn rounded_values_at_halfway_and_signed_zero_match_ieee_reference() {
    for lane in [Lane::F16, Lane::BF16, Lane::F32, Lane::F64] {
        let ty = lane.scalar();
        let (frac, bias) = layout(lane);
        let sign = 1u64 << (8 * lane.bytes() - 1);
        let inf = (2 * bias + 1) << frac;
        let one = bias << frac;
        let ulp = 1u64 << frac;
        let mut inputs = vec![0, 1, ulp - 1, ulp, inf - 1, inf, inf + 1, inf + ulp / 2];
        // Closest representable neighbors of 0.5, 1.5, 2.5 and 3.5.
        // The even result alternates between the lower and upper integer.
        for center in [
            one - ulp,
            one + ulp / 2,
            one + ulp + ulp / 4,
            one + ulp + 3 * ulp / 4,
            one + u64::from(frac) * ulp,
        ] {
            inputs.extend([center - 1, center, center + 1]);
        }
        for op in [
            Unary::Floor,
            Unary::Ceil,
            Unary::Round,
            Unary::Nearest,
            Unary::Trunc,
            Unary::Fract,
            Unary::Signum,
        ] {
            let mut f = Function::new(&[ty], ty);
            let b = f.entry();
            let v = f.append(b, Op::Unary(op, f.params(b)[0])).unwrap();
            f.set_terminator(b, Terminator::Return(v));
            let k = Kernel::compile(&f, &crate::test_target::options()).unwrap();
            for &input in &inputs {
                for raw in [input, input | sign] {
                    let mut args = [[0u8; 64]; 3];
                    args[0][..8].copy_from_slice(&raw.to_le_bytes());
                    let expected = evaluate(
                        Action::Unary(op),
                        ty,
                        ty,
                        &args,
                        crate::test_target::width(),
                    )
                    .unwrap();
                    let expected = u64::from_le_bytes(expected[..8].try_into().unwrap());
                    let mut out = [0];
                    k.call_packed(&[raw], &mut out).unwrap();
                    let actual = out[0];
                    assert!(
                        actual == expected
                            || actual & (sign - 1) > inf && expected & (sign - 1) > inf,
                        "{lane:?} {op:?}({raw:#x}): {actual:#x}, expected {expected:#x}"
                    );
                }
            }
        }
    }
}

#[test]
fn min_max_preserve_numeric_operand_with_nan_on_either_side() {
    for lane in [Lane::F16, Lane::BF16, Lane::F32, Lane::F64] {
        let ty = lane.scalar();
        let (frac, bias) = layout(lane);
        let sign = 1u64 << (8 * lane.bytes() - 1);
        let inf = (2 * bias + 1) << frac;
        let nan = inf + (1 << (frac - 1));
        let one = bias << frac;
        for op in [Binary::Min, Binary::Max] {
            let mut f = Function::new(&[ty, ty], ty);
            let b = f.entry();
            let [a, c] = f.params(b).try_into().unwrap();
            let v = f.append(b, Op::Bin(op, a, c)).unwrap();
            f.set_terminator(b, Terminator::Return(v));
            let k = Kernel::compile(&f, &crate::test_target::options()).unwrap();
            let run = |x: u64, y: u64| {
                let mut out = [0];
                k.call_packed(&[x, y], &mut out).unwrap();
                out[0]
            };
            for x in [0, sign, 1, one, one | sign, inf, inf | sign] {
                assert_eq!(run(x, nan), x);
                assert_eq!(run(nan, x), x);
            }
            assert!(run(nan, nan | sign) & (sign - 1) > inf);
            for (a, b) in [(0, sign), (sign, 0)] {
                // Rust permits either zero sign when the inputs compare equal.
                assert_eq!(run(a, b) & !sign, 0);
            }
            assert_eq!(run(sign, sign), sign);
            assert_eq!(run(0, 0), 0);
            let expected = if op == Binary::Min { one | sign } else { one };
            assert_eq!(run(one, one | sign), expected);
            assert_eq!(run(one | sign, one), expected);
        }
    }
}
