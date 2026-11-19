//! Numeric operation signatures and reference semantics for the HIR interpreter.
pub(crate) mod half;
use crate::hir::{Binary, Ternary, Type, Unary};
#[cfg(test)]
mod reference;
#[cfg(test)]
pub(crate) use reference::read;

pub(crate) type Bits = [u8; 64];

#[derive(Clone, Copy, Debug)]
pub(crate) enum Action {
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
            Action::Iota => &[],
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
}
