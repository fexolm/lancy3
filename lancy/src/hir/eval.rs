//! Portable numeric semantics used by the SSA reference and AArch64 libcalls.
use super::{
    integer::{evaluate as integer, extend, mask, signed},
    operation::Action,
};
use crate::support::float16::Format;
use crate::{
    CallError,
    hir::{Binary, Lane, Ternary, Type, Unary},
};
pub(crate) type Bits = [u8; 64];

pub(crate) fn read(bytes: &[u8]) -> u128 {
    let mut word = [0; 16];
    word[..bytes.len()].copy_from_slice(bytes);
    u128::from_le_bytes(word)
}

pub(crate) fn evaluate(
    action: Action,
    input: Type,
    output: Type,
    args: &[Bits; 3],
    width: usize,
) -> Result<Bits, CallError> {
    let lane = match input {
        // Pointer-to-word bitcasts preserve the address bits without an element type.
        Type::Ptr if matches!(action, Action::Bitcast) => return Ok(args[0]),
        ty => ty.lane().expect("numeric operand"),
    };
    let size = lane.bytes();
    let mut out = [0; 64];
    let [a, b, c] = args;
    match action {
        Action::Bitcast => return Ok(*a),
        Action::Iota => {
            for (i, elem) in out[..width].chunks_exact_mut(size).enumerate() {
                elem.copy_from_slice(&cast(Lane::Usize, lane, i as u128).to_le_bytes()[..size]);
            }
        }
        Action::AnyTrue | Action::AllTrue => {
            let mut lanes = a[..width].chunks_exact(size).map(|e| read(e) != 0);
            let yes = if matches!(action, Action::AnyTrue) {
                lanes.any(|x| x)
            } else {
                lanes.all(|x| x)
            };
            out[..4].copy_from_slice(&(-i32::from(yes)).to_le_bytes());
        }
        Action::Widen(_) | Action::Narrow => {
            let dst = output.lane().unwrap();
            for (i, elem) in out[..width].chunks_exact_mut(dst.bytes()).enumerate() {
                let start = i * size
                    + if matches!(action, Action::Widen(true)) {
                        width / 2
                    } else {
                        0
                    };
                let source = if start >= width { b } else { a };
                let offset = start % width;
                let n = cast(lane, dst, read(&source[offset..offset + size]));
                elem.copy_from_slice(&n.to_le_bytes()[..dst.bytes()]);
            }
        }
        Action::Ternary(Ternary::Select) if !matches!(input, Type::Vector(_)) => {
            out = if read(&a[..4]) != 0 { *b } else { *c };
        }
        Action::Compress | Action::Expand => {
            let mut pos = 0;
            for i in (0..width).step_by(size) {
                if b[i + size - 1] & 0x80 == 0 {
                    continue;
                }
                let (src, dst) = if matches!(action, Action::Compress) {
                    (i, pos)
                } else {
                    (pos, i)
                };
                out[dst..dst + size].copy_from_slice(&a[src..src + size]);
                pos += size;
            }
        }
        Action::Shuffle => {
            let count = width / size;
            for (index, elem) in c[..width]
                .chunks_exact(size)
                .zip(out[..width].chunks_exact_mut(size))
            {
                let i = read(index);
                if i < (2 * count) as u128 {
                    let src = if i < count as u128 { a } else { b };
                    let start = i as usize % count * size;
                    elem.copy_from_slice(&src[start..start + size]);
                }
            }
        }
        Action::Reduce(op) => {
            let scalar = lane.scalar();
            let mut pair = [[0; 64]; 3];
            let mut elems = a[..width].chunks_exact(size);
            pair[0][..size].copy_from_slice(elems.next().unwrap());
            for v in elems {
                pair[1][..size].copy_from_slice(v);
                pair[0] = evaluate(Action::Binary(op), scalar, scalar, &pair, width)?;
            }
            out = pair[0];
        }
        Action::Extract | Action::Insert => {
            let index = read(&b[..4]) as usize;
            if index >= width / size {
                return Err(CallError::OutOfBounds);
            }
            let range = index * size..(index + 1) * size;
            if matches!(action, Action::Insert) {
                out = *a;
                out[range].copy_from_slice(&c[..size]);
            } else {
                out[..size].copy_from_slice(&a[range]);
            }
        }
        Action::Splat => {
            for elem in out[..width].chunks_exact_mut(size) {
                elem.copy_from_slice(&a[..size]);
            }
        }
        action => {
            let dest = output.lane().unwrap();
            let result_size = dest.bytes();
            let mask = mask(lane);
            for (i, slot) in out[..output.bytes(width)]
                .chunks_exact_mut(result_size)
                .enumerate()
            {
                let range = i * size..(i + 1) * size;
                let x = read(&a[range.clone()]);
                let y = read(&b[range.clone()]);
                let z = read(&c[range]);
                let result = match action {
                    Action::Cast => cast(lane, dest, x),
                    Action::Ternary(Ternary::Select) => {
                        if x & (1 << (size * 8 - 1)) != 0 {
                            y
                        } else {
                            z
                        }
                    }
                    Action::Ternary(Ternary::BitSelect) => (x & y) | (!x & z),
                    Action::Unary(Unary::Not) => !x,
                    Action::Binary(Binary::And | Binary::Or | Binary::Xor) => {
                        integer(action, lane, x, y, mask)?
                    }
                    _ if lane.float() => match lane {
                        Lane::F16 | Lane::BF16 => {
                            float16(Format::for_lane(lane), action, x as u16, y as u16, z as u16)
                        }
                        Lane::F32 => float32(action, x as u32, y as u32, z as u32),
                        _ => float64(action, x as u64, y as u64, z as u64),
                    },
                    _ => integer(action, lane, x, y, mask)?,
                };
                slot.copy_from_slice(&result.to_le_bytes()[..result_size]);
            }
        }
    }
    Ok(out)
}

macro_rules! float_eval {
    ($name:ident,$float:ty,$bits:ty) => {
        pub(super) fn $name(action: Action, x: $bits, y: $bits, z: $bits) -> u128 {
            use Binary::*;
            let (a, b, c) = (
                <$float>::from_bits(x),
                <$float>::from_bits(y),
                <$float>::from_bits(z),
            );
            let n = match action {
                Action::Unary(op) => match op {
                    Unary::IsNan
                    | Unary::IsInfinite
                    | Unary::IsFinite
                    | Unary::IsNormal
                    | Unary::IsSubnormal
                    | Unary::SignBit => {
                        let pred = match op {
                            Unary::IsNan => a.is_nan(),
                            Unary::IsInfinite => a.is_infinite(),
                            Unary::IsFinite => a.is_finite(),
                            Unary::IsNormal => a.is_normal(),
                            Unary::IsSubnormal => a.is_subnormal(),
                            _ => a.is_sign_negative(),
                        };
                        return if pred { u128::MAX } else { 0 };
                    }
                    Unary::Signum => a.signum(),
                    Unary::NextUp => a.next_up(),
                    Unary::NextDown => a.next_down(),
                    Unary::Neg => -a,
                    Unary::Abs => a.abs(),
                    Unary::Sqrt => a.sqrt(),
                    Unary::Floor => a.floor(),
                    Unary::Ceil => a.ceil(),
                    Unary::Trunc => a.trunc(),
                    Unary::Round => a.round(),
                    Unary::Nearest => a.round_ties_even(),
                    Unary::Fract => a.fract(),
                    Unary::Recip => a.recip(),
                    Unary::Exp => a.exp(),
                    Unary::Exp2 => a.exp2(),
                    Unary::Ln => a.ln(),
                    Unary::Log2 => a.log2(),
                    Unary::Log10 => a.log10(),
                    Unary::Sin => a.sin(),
                    Unary::Cos => a.cos(),
                    Unary::Tan => a.tan(),
                    Unary::Asin => a.asin(),
                    Unary::Acos => a.acos(),
                    Unary::Atan => a.atan(),
                    Unary::Sinh => a.sinh(),
                    Unary::Cosh => a.cosh(),
                    Unary::Tanh => a.tanh(),
                    Unary::Asinh => a.asinh(),
                    Unary::Acosh => a.acosh(),
                    Unary::Atanh => a.atanh(),
                    Unary::ExpM1 => a.exp_m1(),
                    Unary::Ln1p => a.ln_1p(),
                    Unary::Cbrt => a.cbrt(),
                    _ => unreachable!("verified float unary"),
                },
                Action::Binary(op) => match op {
                    Add => a + b,
                    Sub => a - b,
                    Mul => a * b,
                    Div => a / b,
                    Rem => a % b,
                    // IEEE 754-2019 minimumNumber/maximumNumber: a NaN
                    // operand, quiet or signaling, yields the other one, and
                    // -0 orders below +0. Rust's min/max leave both open.
                    Min | Max => {
                        let min = op == Min;
                        if b.is_nan() {
                            a
                        } else if a.is_nan() {
                            b
                        } else if a == b {
                            if a.is_sign_negative() == min { a } else { b }
                        } else if (a < b) == min {
                            a
                        } else {
                            b
                        }
                    }
                    Pow => a.powf(b),
                    CopySign => a.copysign(b),
                    Hypot => a.hypot(b),
                    Atan2 => a.atan2(b),
                    Ordered | Unordered | Eq | Ne | Lt | Le | Gt | Ge => {
                        let yes = match op {
                            Ordered => !a.is_nan() && !b.is_nan(),
                            Unordered => a.is_nan() || b.is_nan(),
                            Eq => a == b,
                            Ne => a != b,
                            Lt => a < b,
                            Le => a <= b,
                            Gt => a > b,
                            _ => a >= b,
                        };
                        return if yes { u128::MAX } else { 0 };
                    }
                    _ => unreachable!("verified float binary"),
                },
                Action::Ternary(Ternary::MulAdd) => a.mul_add(b, c),
                _ => unreachable!("verified float operation"),
            };
            n.to_bits() as u128
        }
    };
}
float_eval!(float32, f32, u32);
float_eval!(float64, f64, u64);

fn float16(format: Format, action: Action, x: u16, y: u16, z: u16) -> u128 {
    let a = format.decode(x);
    let b = format.decode(y);
    let c = format.decode(z);
    let mag = x & 0x7fff;
    match action {
        Action::Unary(Unary::Neg) => return u128::from(x ^ 0x8000),
        Action::Unary(Unary::Abs) => return mag.into(),
        Action::Binary(Binary::CopySign) => return u128::from(mag | y & 0x8000),
        Action::Unary(Unary::IsNormal) => {
            return if mag >= 1 << format.fraction_bits() && mag < format.infinity() {
                u128::MAX
            } else {
                0
            };
        }
        Action::Unary(Unary::IsSubnormal) => {
            return if mag != 0 && mag < 1 << format.fraction_bits() {
                u128::MAX
            } else {
                0
            };
        }
        Action::Unary(op @ (Unary::NextUp | Unary::NextDown)) => {
            let up = op == Unary::NextUp;
            let result = if a.is_nan() || (a.is_infinite() && up == a.is_sign_positive()) {
                x
            } else if mag == 0 {
                if up { 1 } else { 0x8001 }
            } else if up == a.is_sign_positive() {
                x + 1
            } else {
                x - 1
            };
            return result.into();
        }
        Action::Ternary(Ternary::MulAdd) => {
            let product = a * b;
            let sum = product + c;
            // The significand product is exact in f64. FastTwoSum recovers the rounding residual.
            let residual = if product.abs() >= c.abs() {
                (product - sum) + c
            } else {
                (c - sum) + product
            };
            return format.encode(sum, residual).into();
        }
        _ => {}
    }
    let bits = float64(action, a.to_bits(), b.to_bits(), c.to_bits());
    match action {
        Action::Unary(op) if op.result_type(crate::hir::Type::F64) == crate::hir::Type::I32 => bits,
        Action::Binary(op) if op.result_type(crate::hir::Type::F64) == crate::hir::Type::I32 => {
            bits
        }
        _ => format.encode(f64::from_bits(bits as u64), 0.).into(),
    }
}

fn cast(from: Lane, to: Lane, x: u128) -> u128 {
    let a = signed(x, from.bytes() as u32 * 8);
    if from == to {
        return x;
    }
    let n = if matches!(from, Lane::F16 | Lane::BF16) {
        Format::for_lane(from).decode(x as u16)
    } else if from == Lane::F32 {
        f32::from_bits(x as u32) as f64
    } else {
        f64::from_bits(x as u64)
    };
    if matches!(to, Lane::F16 | Lane::BF16) {
        let format = Format::for_lane(to);
        if from.float() {
            return format.encode(n, 0.).into();
        }
        let negative = from.signed() && a < 0;
        let mag = if from.signed() { a.unsigned_abs() } else { x };
        let rounded = mag as f64;
        // Only the direction of the residual affects a half-precision tie.
        let residual = if rounded >= 2f64.powi(128) || (rounded as u128) > mag {
            -1.
        } else if (rounded as u128) < mag {
            1.
        } else {
            0.
        };
        return if negative {
            format.encode(-rounded, -residual)
        } else {
            format.encode(rounded, residual)
        }
        .into();
    }
    if to == Lane::F32 {
        let f = if from.float() {
            n as f32
        } else if from.signed() {
            a as f32
        } else {
            x as f32
        };
        return f.to_bits() as u128;
    }
    if to == Lane::F64 {
        let f = if from.float() {
            n
        } else if from.signed() {
            a as f64
        } else {
            x as f64
        };
        return f.to_bits() as u128;
    }
    if !from.float() {
        return extend(from, x);
    }
    let bits = to.bytes() as u32 * 8;
    if to.signed() {
        let min = signed(1 << (bits - 1), bits);
        (n as i128).clamp(min, !min) as u128
    } else {
        (n as u128).min(u128::MAX >> (128 - bits))
    }
}
