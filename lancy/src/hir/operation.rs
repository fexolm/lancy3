//! Operation kinds and types shared by lowering and the HIR evaluator.
use super::{Binary, Ternary, Type, Unary};

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

pub(crate) fn operation(
    op: crate::hir::Op,
    ty: impl Fn(crate::hir::Value) -> Type,
) -> Option<(Action, Type, Type, [Option<crate::hir::Value>; 3])> {
    use crate::hir::{Op, Simd};
    let (action, input, output, operands) = match op {
        Op::Simd(Simd::Compress(x, m)) => {
            (Action::Compress, ty(x), ty(x), [Some(x), Some(m), None])
        }
        Op::Simd(Simd::Expand(x, m)) => (Action::Expand, ty(x), ty(x), [Some(x), Some(m), None]),
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
        Op::Simd(Simd::AnyTrue(x)) => (Action::AnyTrue, ty(x), Type::I32, [Some(x), None, None]),
        Op::Simd(Simd::AllTrue(x)) => (Action::AllTrue, ty(x), Type::I32, [Some(x), None, None]),
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
        Op::Unary(op, x) => (
            Action::Unary(op),
            ty(x),
            op.result_type(ty(x)),
            [Some(x), None, None],
        ),
        Op::Bin(op, x, y) => (
            Action::Binary(op),
            ty(x),
            op.result_type(ty(x)),
            [Some(x), Some(y), None],
        ),
        Op::Ternary(op, x, y, z) => (
            Action::Ternary(op),
            ty(y),
            ty(y),
            [Some(x), Some(y), Some(z)],
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
    Some((action, input, output, operands))
}
