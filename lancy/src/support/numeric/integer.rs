//! Integer lane semantics shared by constant folding and the reference evaluator.
use super::Action;
use crate::Trap;
use crate::hir::{Binary, Lane, Unary};

pub(crate) fn mask(lane: Lane) -> u128 {
    u128::MAX >> (128 - lane.bytes() * 8)
}

pub(crate) fn signed(x: u128, bits: u32) -> i128 {
    ((x << (128 - bits)) as i128) >> (128 - bits)
}

pub(crate) fn extend(lane: Lane, x: u128) -> u128 {
    if lane.signed() {
        signed(x, lane.bytes() as u32 * 8) as u128
    } else {
        x
    }
}

/// Evaluate a verified integer binary operation on lane bit patterns.
pub(crate) fn binary(op: Binary, lane: Lane, x: u128, y: u128) -> Result<u128, Trap> {
    let mask = mask(lane);
    evaluate(Action::Binary(op), lane, x & mask, y & mask, mask)
}

pub(crate) fn evaluate(
    action: Action,
    lane: Lane,
    x: u128,
    y: u128,
    mask: u128,
) -> Result<u128, Trap> {
    use Binary::*;
    let bits = lane.bytes() as u32 * 8;
    let (a, b) = (signed(x, bits), signed(y, bits));
    let min = signed(1 << (bits - 1), bits);
    let max = !min;
    let shift = (y as u32) & (bits - 1);
    let sign = lane.signed();
    let order = if sign { a.cmp(&b) } else { x.cmp(&y) };
    let value = match action {
        Action::Unary(op) => match op {
            Unary::Not => !x,
            Unary::Neg => x.wrapping_neg(),
            Unary::Abs => {
                if sign {
                    a.unsigned_abs()
                } else {
                    x
                }
            }
            Unary::Popcnt => x.count_ones() as u128,
            Unary::LeadingZeros => (x.leading_zeros() - (128 - bits)) as u128,
            Unary::TrailingZeros => x.trailing_zeros().min(bits) as u128,
            _ => unreachable!("verified integer unary"),
        },
        Action::Binary(op) => match op {
            Add => x.wrapping_add(y),
            Sub => x.wrapping_sub(y),
            Mul => x.wrapping_mul(y),
            And => x & y,
            Or => x | y,
            Xor => x ^ y,
            Div | Rem => {
                if y == 0 {
                    return Err(Trap::DivisionByZero);
                }
                if sign && a == min && b == -1 {
                    return Err(Trap::DivisionOverflow);
                }
                match (sign, op) {
                    (true, Div) => (a / b) as u128,
                    (true, _) => (a % b) as u128,
                    (false, Div) => x / y,
                    _ => x % y,
                }
            }
            Shl => x << shift,
            Shr if sign => (a >> shift) as u128,
            Shr => x >> shift,
            RotateLeft => (x << shift) | (x >> ((bits - shift) & (bits - 1))),
            RotateRight => (x >> shift) | (x << ((bits - shift) & (bits - 1))),
            SaturatingAdd if sign => a.saturating_add(b).clamp(min, max) as u128,
            SaturatingSub if sign => a.saturating_sub(b).clamp(min, max) as u128,
            SaturatingAdd => x.saturating_add(y).min(mask),
            SaturatingSub => x.saturating_sub(y),
            Min => {
                if order.is_lt() {
                    x
                } else {
                    y
                }
            }
            Max => {
                if order.is_gt() {
                    x
                } else {
                    y
                }
            }
            Eq | Ne | Lt | Le | Gt | Ge => {
                let yes = match op {
                    Eq => order.is_eq(),
                    Ne => order.is_ne(),
                    Lt => order.is_lt(),
                    Le => order.is_le(),
                    Gt => order.is_gt(),
                    _ => order.is_ge(),
                };
                if yes { u128::MAX } else { 0 }
            }
            _ => unreachable!("verified integer binary"),
        },
        _ => unreachable!("verified integer operation"),
    };
    Ok(value)
}
