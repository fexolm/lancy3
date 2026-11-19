//! Lane semantics used for operations without a single native instruction.
mod half;
use crate::{
    CallError,
    hir::{Binary, Lane, Ternary, Type, Unary},
};
use half::Format;

pub(crate) type Bits = [u8; 64];

pub(crate) fn read(bytes: &[u8]) -> u128 {
    let mut word = [0; 16];
    word[..bytes.len()].copy_from_slice(bytes);
    u128::from_le_bytes(word)
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Action {
    Literal(u128),
    Unary(Unary),
    Binary(Binary),
    Ternary(Ternary),
    Cast,
    Bitcast,
    Splat,
    Extract,
    Insert,
    Compress,
    Expand,
    Iota,
    Shuffle,
    Reduce(Binary),
    AnyTrue,
    AllTrue,
    Widen(bool),
    Narrow,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Math {
    pub action: Action,
    pub input: Type,
    pub output: Type,
}

impl Math {
    pub fn from_op(
        op: crate::hir::Op,
        ty: impl Fn(crate::hir::Value) -> Type,
    ) -> Option<(Self, [Option<crate::hir::Value>; 3])> {
        use crate::hir::{Op, Simd};
        let (action, input, output, operands) = match op {
            Op::Simd(Simd::Compress(x, m)) => {
                (Action::Compress, ty(x), ty(x), [Some(x), Some(m), None])
            }
            Op::Simd(Simd::Expand(x, m)) => {
                (Action::Expand, ty(x), ty(x), [Some(x), Some(m), None])
            }
            Op::Simd(Simd::Iota(l)) => (Action::Iota, l.scalar(), Type::Vector(l), [None; 3]),
            Op::Simd(Simd::Shuffle(x, y, z)) => {
                (Action::Shuffle, ty(x), ty(x), [Some(x), Some(y), Some(z)])
            }
            Op::Simd(Simd::Reduce(op, x)) => (
                Action::Reduce(op),
                ty(x),
                ty(x).lane().unwrap().scalar(),
                [Some(x), None, None],
            ),
            Op::Simd(Simd::AnyTrue(x)) => {
                (Action::AnyTrue, ty(x), Type::I32, [Some(x), None, None])
            }
            Op::Simd(Simd::AllTrue(x)) => {
                (Action::AllTrue, ty(x), Type::I32, [Some(x), None, None])
            }
            Op::Simd(Simd::Widen { to, high, value }) => (
                Action::Widen(high),
                ty(value),
                Type::Vector(to),
                [Some(value), None, None],
            ),
            Op::Simd(Simd::Narrow(to, x, y)) => (
                Action::Narrow,
                ty(x),
                Type::Vector(to),
                [Some(x), Some(y), None],
            ),
            Op::Literal(l, x) => (Action::Literal(x), l.scalar(), l.scalar(), [None; 3]),
            Op::Unary(op, x) | Op::Simd(Simd::Unary(op, x)) => (
                Action::Unary(op),
                ty(x),
                op.result_type(ty(x)),
                [Some(x), None, None],
            ),
            Op::Bin(op, x, y) | Op::Simd(Simd::Bin(op, x, y)) => (
                Action::Binary(op),
                ty(x),
                op.result_type(ty(x)),
                [Some(x), Some(y), None],
            ),
            Op::Ternary(op, x, y, z) | Op::Simd(Simd::Ternary(op, x, y, z)) => (
                Action::Ternary(op),
                ty(y),
                ty(y),
                [Some(x), Some(y), Some(z)],
            ),
            Op::Simd(Simd::Cast(l, x)) => {
                (Action::Cast, ty(x), Type::Vector(l), [Some(x), None, None])
            }
            Op::Simd(Simd::Bitcast(l, x)) => (
                Action::Bitcast,
                ty(x),
                Type::Vector(l),
                [Some(x), None, None],
            ),
            Op::Cast(to, x) => (Action::Cast, ty(x), to, [Some(x), None, None]),
            Op::Bitcast(to, x) => (Action::Bitcast, ty(x), to, [Some(x), None, None]),
            Op::Simd(Simd::Splat(x)) => (
                Action::Splat,
                ty(x),
                Type::Vector(ty(x).lane().unwrap()),
                [Some(x), None, None],
            ),
            Op::Simd(Simd::ExtractLane(x, y)) => (
                Action::Extract,
                ty(x),
                ty(x).lane().unwrap().scalar(),
                [Some(x), Some(y), None],
            ),
            Op::Simd(Simd::InsertLane(x, y, z)) => {
                (Action::Insert, ty(x), ty(x), [Some(x), Some(y), Some(z)])
            }
            _ => return None,
        };
        Some((
            Self {
                action,
                input,
                output,
            },
            operands,
        ))
    }

    pub fn inputs(self) -> &'static [usize] {
        match self.action {
            Action::Iota | Action::Literal(_) => &[],
            Action::Reduce(_)
            | Action::AnyTrue
            | Action::AllTrue
            | Action::Widen(_)
            | Action::Unary(_)
            | Action::Cast
            | Action::Bitcast
            | Action::Splat => &[0],
            Action::Compress
            | Action::Expand
            | Action::Narrow
            | Action::Binary(_)
            | Action::Extract => &[0, 1],
            Action::Shuffle | Action::Ternary(_) | Action::Insert => &[0, 1, 2],
        }
    }
    pub fn operand_type(self, i: usize) -> Type {
        match (self.action, i) {
            (Action::Extract | Action::Insert, 1) => Type::I32,
            (Action::Ternary(Ternary::Select), 0) => self.input.mask_type(),
            (Action::Insert, 2) => self.input.lane().unwrap().scalar(),
            _ => self.input,
        }
    }
    pub fn eval(self, args: &[Bits; 3], width: usize) -> Result<Bits, CallError> {
        let lane = self.input.lane().unwrap();
        let size = lane.bytes();
        let mut out = [0; 64];
        let [a, b, c] = args;
        match self.action {
            Action::Bitcast => return Ok(*a),
            Action::Iota => {
                for (i, elem) in out[..width].chunks_exact_mut(size).enumerate() {
                    elem.copy_from_slice(&cast(Lane::Usize, lane, i as u128).to_le_bytes()[..size]);
                }
            }
            Action::AnyTrue | Action::AllTrue => {
                let mut lanes = a[..width].chunks_exact(size).map(|e| read(e) != 0);
                let yes = if matches!(self.action, Action::AnyTrue) {
                    lanes.any(|x| x)
                } else {
                    lanes.all(|x| x)
                };
                out[..4].copy_from_slice(&(-i32::from(yes)).to_le_bytes());
            }
            Action::Widen(_) | Action::Narrow => {
                let dst = self.output.lane().unwrap();
                for (i, elem) in out[..width].chunks_exact_mut(dst.bytes()).enumerate() {
                    let start = i * size
                        + if matches!(self.action, Action::Widen(true)) {
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
            Action::Ternary(Ternary::Select) if !matches!(self.input, Type::Vector(_)) => {
                out = if read(&a[..4]) != 0 { *b } else { *c };
            }
            Action::Compress | Action::Expand => {
                let mut pos = 0;
                for i in (0..width).step_by(size) {
                    if b[i + size - 1] & 0x80 == 0 {
                        continue;
                    }
                    let (src, dst) = if matches!(self.action, Action::Compress) {
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
                let scalar = Math {
                    action: Action::Binary(op),
                    input: lane.scalar(),
                    output: lane.scalar(),
                };
                let mut pair = [[0; 64]; 3];
                let mut elems = a[..width].chunks_exact(size);
                pair[0][..size].copy_from_slice(elems.next().unwrap());
                for v in elems {
                    pair[1][..size].copy_from_slice(v);
                    pair[0] = scalar.eval(&pair, width)?;
                }
                out = pair[0];
            }
            Action::Literal(bits) => {
                out[..size].copy_from_slice(&bits.to_le_bytes()[..size]);
            }
            Action::Extract | Action::Insert => {
                let index = read(&b[..4]) as usize;
                if index >= width / size {
                    return Err(CallError::OutOfBounds);
                }
                let range = index * size..(index + 1) * size;
                if matches!(self.action, Action::Insert) {
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
                let dest = self.output.lane().unwrap();
                let result_size = dest.bytes();
                let mask = u128::MAX >> (128 - size * 8);
                for (i, slot) in out[..self.output.bytes(width)]
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
                        Action::Binary(Binary::And) => x & y,
                        Action::Binary(Binary::Or) => x | y,
                        Action::Binary(Binary::Xor) => x ^ y,
                        _ if lane.float() => match lane {
                            Lane::F16 | Lane::BF16 => {
                                Format::for_lane(lane).eval(action, x as u16, y as u16, z as u16)
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
}

fn signed(x: u128, bits: u32) -> i128 {
    ((x << (128 - bits)) as i128) >> (128 - bits)
}

fn integer(action: Action, lane: Lane, x: u128, y: u128, mask: u128) -> Result<u128, CallError> {
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
            Div | Rem => {
                if y == 0 {
                    return Err(CallError::DivisionByZero);
                }
                if sign && a == min && b == -1 {
                    return Err(CallError::DivisionOverflow);
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

macro_rules! float_eval {
    ($name:ident,$float:ty,$bits:ty) => {
        fn $name(action: Action, x: $bits, y: $bits, z: $bits) -> u128 {
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
                    Min => a.min(b),
                    Max => a.max(b),
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
        return if from.signed() { a as u128 } else { x };
    }
    let bits = to.bytes() as u32 * 8;
    if to.signed() {
        let min = signed(1 << (bits - 1), bits);
        (n as i128).clamp(min, !min) as u128
    } else {
        (n as u128).min(u128::MAX >> (128 - bits))
    }
}
