//! Exercise the public numeric instruction set on real, runtime-supplied operands.
pub(crate) mod conversions;
mod rounding;
use crate::{
    CallError, ScalarKernel, Trap,
    hir::operation::operation,
    hir::{Binary, Function, Lane, Op, Simd, Terminator, Ternary, Type, Unary, Value},
    support::numeric::{Bits, evaluate},
};

pub(crate) const LANES: [Lane; 16] = {
    use Lane::*;
    [
        I8, U8, I16, U16, I32, U32, I64, U64, I128, U128, Isize, Usize, F16, BF16, F32, F64,
    ]
};

/// Ordinary operands: every operation succeeds on them.
fn ordinary(lane: Lane) -> Vec<u128> {
    match lane {
        Lane::F16 => vec![0x3800, 0x3c00, 0x3e00, 0x4000],
        Lane::BF16 => vec![0x3f00, 0x3f80, 0x3fc0, 0x4000],
        Lane::F32 => [0.5f32, 1.0, 1.5, 2.0].map(|x| x.to_bits().into()).into(),
        Lane::F64 => [0.5f64, 1.0, 1.5, 2.0].map(|x| x.to_bits().into()).into(),
        _ => vec![
            0u128.wrapping_sub(77),
            0x1020_3040_5060_7080_90a0_b0c0_d0e0_f011,
            u128::MAX / 3,
            1,
        ],
    }
}

/// Boundary operands: NaN and sNaN, signed zeros and infinities, subnormals,
/// extreme magnitudes and rounding ties, integer extremes, -1, divisors of
/// zero and shift counts at and past the lane width. Truncated to the lane.
fn boundary(lane: Lane) -> Vec<u128> {
    const F32: [u32; 20] = [
        0x7fc0_0000,
        0x7f80_0001,
        0xffc0_0001,
        0,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        1,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x3f80_0000,
        0xbf80_0000,
        0x3fc0_0000,
        0x4020_0000,
        0xc020_0000,
        0x4f00_0000,
        0xcf00_0000,
        0x5f00_0000,
    ];
    match lane {
        Lane::F32 => F32.map(u128::from).into(),
        // The upper halves of the binary32 boundaries are bfloat16 boundaries.
        Lane::BF16 => F32.map(|x| u128::from(x >> 16)).into(),
        Lane::F16 => vec![
            0x7e00, 0x7c01, 0xfe01, 0, 0x8000, 0x7c00, 0xfc00, 1, 0x03ff, 0x0400, 0x7bff, 0xfbff,
            0x3c00, 0xbc00, 0x3e00, 0x4100, 0xc100, 0x7800,
        ],
        Lane::F64 => [
            f64::NAN.to_bits(),
            0x7ff0_0000_0000_0001,
            (-f64::NAN).to_bits(),
            0,
            (-0.0f64).to_bits(),
            f64::INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
            1,
            0x000f_ffff_ffff_ffff,
            f64::MIN_POSITIVE.to_bits(),
            f64::MAX.to_bits(),
            f64::MIN.to_bits(),
            1.0f64.to_bits(),
            (-1.0f64).to_bits(),
            1.5f64.to_bits(),
            2.5f64.to_bits(),
            (-2.5f64).to_bits(),
            2f64.powi(63).to_bits(),
            (-(2f64.powi(63))).to_bits(),
            2f64.powi(64).to_bits(),
        ]
        .map(u128::from)
        .into(),
        _ => {
            let bits = lane.bytes() as u128 * 8;
            let min = 1u128 << (bits - 1);
            vec![
                0,
                1,
                2,
                u128::MAX,
                min,
                min - 1,
                min + 1,
                bits - 1,
                bits,
                bits + 1,
                2 * bits,
                u128::MAX / 3,
                7,
            ]
        }
    }
}

/// Compare `ScalarKernel` with `evaluate`, results and traps alike, on
/// operand sets that pair every value of `inputs` with every other.
fn differential(params: &[Type], ret: Type, make_op: impl FnOnce(&[Value]) -> Op) {
    let mut f = Function::new(params, ret);
    let b = f.entry();
    let op = make_op(f.params(b));
    let (action, input_type, output_type, _) = operation(op, |v| f.value_type(v)).unwrap();
    let v = f.append(b, op).unwrap();
    f.set_terminator(b, Terminator::Return(v));
    let k = ScalarKernel::compile(&f, &crate::test_target::options())
        .unwrap_or_else(|e| panic!("{params:?} {op:?}: {e}"));
    let width = crate::test_target::width();
    for inputs in [ordinary, boundary] {
        // Lanes of one vector already see every offset between two arguments;
        // a scalar needs one round per pair of the first two arguments.
        let len = inputs(params[0].lane().unwrap()).len();
        let rounds = if matches!(params[0], Type::Vector(_)) {
            len
        } else {
            len.pow(params.len().min(2) as u32)
        };
        for round in 0..rounds {
            check_round(&k, params, ret, op, inputs, round, |bits| {
                evaluate(action, input_type, output_type, bits, width).map_err(CallError::Trap)
            });
        }
    }
}

fn check_round(
    k: &ScalarKernel,
    params: &[Type],
    ret: Type,
    op: Op,
    inputs: fn(Lane) -> Vec<u128>,
    round: usize,
    reference: impl FnOnce(&[Bits; 3]) -> Result<Bits, CallError>,
) {
    let width = crate::test_target::width();
    let mut bits: [Bits; 3] = [[0; 64]; 3];
    let mut args = Vec::new();
    for (arg, &ty) in params.iter().enumerate() {
        let lane = ty.lane().unwrap();
        let values = inputs(lane);
        let shift = round / values.len().pow(arg as u32);
        let n = ty.bytes(width);
        for (i, p) in bits[arg][..n].chunks_exact_mut(lane.bytes()).enumerate() {
            let value = values[(i + shift) % values.len()];
            p.copy_from_slice(&value.to_le_bytes()[..p.len()]);
        }
        args.extend(
            bits[arg][..n.next_multiple_of(8)]
                .chunks_exact(8)
                .map(|p| u64::from_le_bytes(p.try_into().unwrap())),
        );
    }
    let n = ret.bytes(width);
    let mut out = vec![0; n.div_ceil(8)];
    let actual = k.call_words(&args, &mut out);
    let context = || format!("{params:?} {op:?} {:x?}", bits.map(|b| b[..16].to_vec()));
    let expected = match (reference(&bits), actual) {
        (Ok(expected), Ok(())) => expected,
        (expected, actual) => {
            assert_eq!(actual, expected.map(|_| ()), "{}", context());
            return;
        }
    };
    let actual: Vec<_> = out.into_iter().flat_map(u64::to_le_bytes).collect();
    let lane = ret.lane().unwrap();
    for (a, e) in actual[..n]
        .chunks_exact(lane.bytes())
        .zip(expected[..n].chunks_exact(lane.bytes()))
    {
        // NaN payloads and their sign may differ between hardware and software math.
        let nan = |x: &[u8]| match lane {
            Lane::F16 => u16::from_le_bytes(x.try_into().unwrap()) & 0x7fff > 0x7c00,
            Lane::BF16 => u16::from_le_bytes(x.try_into().unwrap()) & 0x7fff > 0x7f80,
            Lane::F32 => f32::from_le_bytes(x.try_into().unwrap()).is_nan(),
            Lane::F64 => f64::from_le_bytes(x.try_into().unwrap()).is_nan(),
            _ => false,
        };
        assert!(
            a == e || nan(a) && nan(e),
            "{}: got {a:x?}, expected {e:x?}",
            context()
        );
    }
}

#[test]
fn arithmetic_and_conversions_match_reference_for_all_lane_types() {
    use Binary as B;
    use Unary as U;
    let unary = [
        U::Neg,
        U::Abs,
        U::Not,
        U::LeadingZeros,
        U::TrailingZeros,
        U::Popcnt,
        U::Sqrt,
        U::Floor,
        U::Ceil,
        U::Trunc,
        U::Round,
        U::Nearest,
        U::Fract,
        U::Recip,
        U::Exp,
        U::Exp2,
        U::Ln,
        U::Log2,
        U::Log10,
        U::Sin,
        U::Cos,
        U::Tan,
        U::Asin,
        U::Acos,
        U::Atan,
        U::IsNan,
        U::IsInfinite,
        U::IsFinite,
        U::IsNormal,
        U::IsSubnormal,
        U::SignBit,
        U::Signum,
        U::NextUp,
        U::NextDown,
        U::Sinh,
        U::Cosh,
        U::Tanh,
        U::Asinh,
        U::Acosh,
        U::Atanh,
        U::ExpM1,
        U::Ln1p,
        U::Cbrt,
    ];
    let binary = [
        B::Add,
        B::Sub,
        B::Mul,
        B::Div,
        B::Rem,
        B::And,
        B::Or,
        B::Xor,
        B::Eq,
        B::Ne,
        B::Lt,
        B::Le,
        B::Gt,
        B::Ge,
        B::Shl,
        B::Shr,
        B::Min,
        B::Max,
        B::Ordered,
        B::Unordered,
        B::CopySign,
        B::Pow,
        B::Atan2,
        B::Hypot,
        B::RotateLeft,
        B::RotateRight,
        B::SaturatingAdd,
        B::SaturatingSub,
    ];
    for lane in LANES {
        for ty in [lane.scalar(), Type::Vector(lane)] {
            let packed = matches!(ty, Type::Vector(_));
            for op in unary.into_iter().filter(|op| op.supports(ty)) {
                differential(&[ty], op.result_type(ty), |p| Op::Unary(op, p[0]));
            }
            for op in binary.into_iter().filter(|op| op.supports(ty)) {
                differential(&[ty, ty], op.result_type(ty), |p| Op::Bin(op, p[0], p[1]));
            }
            for op in [Ternary::Select, Ternary::BitSelect, Ternary::MulAdd] {
                if op == Ternary::MulAdd && !lane.float() {
                    continue;
                }
                let first = if op == Ternary::Select {
                    ty.mask_type()
                } else {
                    ty
                };
                differential(&[first, ty, ty], ty, |p| Op::Ternary(op, p[0], p[1], p[2]));
            }
            for to in LANES {
                if !packed {
                    differential(&[ty], to.scalar(), |p| Op::Cast(to.scalar(), p[0]));
                } else if to.bytes() == lane.bytes() {
                    differential(&[ty], Type::Vector(to), |p| {
                        Op::Cast(Type::Vector(to), p[0])
                    });
                } else if to.bytes() == 2 * lane.bytes() {
                    for high in [false, true] {
                        differential(&[ty], Type::Vector(to), |p| {
                            Op::Simd(Simd::Widen {
                                to,
                                high,
                                value: p[0],
                            })
                        });
                    }
                } else if to.bytes() * 2 == lane.bytes() {
                    differential(&[ty, ty], Type::Vector(to), |p| {
                        Op::Simd(Simd::Narrow(to, p[0], p[1]))
                    });
                }
            }
        }
    }
}

#[test]
fn lane_permutations_and_reductions_match_reference_at_host_width() {
    for lane in LANES {
        let v = Type::Vector(lane);
        let m = v.mask_type();
        differential(&[lane.scalar()], v, |p| Op::Simd(Simd::Splat(p[0])));
        differential(&[v, m], v, |p| Op::Simd(Simd::Compress(p[0], p[1])));
        differential(&[v, m], v, |p| Op::Simd(Simd::Expand(p[0], p[1])));
        differential(&[v, v, m], v, |p| Op::Simd(Simd::Shuffle(p[0], p[1], p[2])));
        for op in [
            Binary::Add,
            Binary::Mul,
            Binary::And,
            Binary::Or,
            Binary::Xor,
            Binary::Min,
            Binary::Max,
        ] {
            differential(&[v], lane.scalar(), |p| Op::Simd(Simd::Reduce(op, p[0])));
        }
        if !lane.float() {
            differential(&[v], Type::I32, |p| Op::Simd(Simd::AnyTrue(p[0])));
            differential(&[v], Type::I32, |p| Op::Simd(Simd::AllTrue(p[0])));
        }
    }
}

#[test]
fn wide_integer_cast_resolves_ties_from_the_low_word() {
    // Adding one at these midpoints changes the correctly rounded result,
    // even though both neighboring integers have the same binary64 encoding.
    let base = 1u128 << 100;
    for (dest, midpoint, expected) in [
        (
            Type::F32,
            base + (1 << 76),
            u64::from((2f32.powi(100)).to_bits()),
        ),
        (Type::BF16, base + (1 << 92), 0x7180),
    ] {
        let mut f = Function::new(&[Type::U128], dest);
        let b = f.entry();
        let r = f.append(b, Op::Cast(dest, f.params(b)[0])).unwrap();
        f.set_terminator(b, Terminator::Return(r));
        let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        for (input, bits) in [
            (midpoint - 1, expected),
            (midpoint, expected),
            (midpoint + 1, expected + 1),
        ] {
            let args = [input as u64, (input >> 64) as u64];
            let mut out = [0];
            k.call_words(&args, &mut out).unwrap();
            assert_eq!(out[0], bits, "{input} as {dest:?}");
        }
    }
}

#[test]
fn overflowing_and_zero_divisors_trap_in_lane_order() {
    for op in [Binary::Div, Binary::Rem] {
        let v = Type::Vector(Lane::I64);
        let mut f = Function::new(&[v, v], v);
        let b = f.entry();
        let [lhs, rhs] = f.params(b).try_into().unwrap();
        let r = f.append(b, Op::Bin(op, lhs, rhs)).unwrap();
        f.set_terminator(b, Terminator::Return(r));
        let k = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
        let n = crate::test_target::width() / 8;
        for (overflow, zero, err) in [
            (0, 1, CallError::Trap(Trap::DivisionOverflow)),
            (1, 0, CallError::Trap(Trap::DivisionByZero)),
        ] {
            let mut args = vec![5; 2 * n];
            args[overflow] = i64::MIN as u64;
            args[n + overflow] = u64::MAX;
            args[n + zero] = 0;
            let mut out = vec![0x55; n];
            assert_eq!(k.call_words(&args, &mut out), Err(err));
        }
    }
}
