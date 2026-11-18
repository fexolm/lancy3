//! Instruction-selected LIR. The same opcodes are first parameterized by SSA
//! values and later by machine homes. Calls and edge arguments are out of line.
use crate::hir::{Binary, Scale, Type, Value};
use crate::support::slotmap::DenseMap;

crate::slot_key!(pub(crate) CallSiteId);
crate::slot_key!(pub(crate) FunctionId);

#[derive(Clone, Copy, Debug)]
pub(super) enum Op<R> {
    Imm(R, i32),
    Move(R, R),
    Alu(R, Binary, R, R),
    BoundsCheck(R, R),
    PtrOffset(R, R, R, Scale),
    Load(R, Type, R),
    Store(Type, R, R),
    Call(R, CallSiteId),
}

// Keep the common pre-allocation stream at four 32-bit words per operation.
const _: () = assert!(std::mem::size_of::<Op<Value>>() <= 16);

impl<R: Copy> Op<R> {
    pub(super) fn map<S>(self, mut f: impl FnMut(R) -> S) -> Op<S> {
        match self {
            Self::Imm(d, n) => Op::Imm(f(d), n),
            Self::Move(d, x) => Op::Move(f(d), f(x)),
            Self::Alu(d, op, a, b) => Op::Alu(f(d), op, f(a), f(b)),
            Self::BoundsCheck(i, n) => Op::BoundsCheck(f(i), f(n)),
            Self::PtrOffset(d, b, i, s) => Op::PtrOffset(f(d), f(b), f(i), s),
            Self::Load(d, b, i) => Op::Load(f(d), b, f(i)),
            Self::Store(b, i, v) => Op::Store(b, f(i), f(v)),
            Self::Call(d, site) => Op::Call(f(d), site),
        }
    }
    pub(super) fn dst(self) -> Option<R> {
        match self {
            Self::Imm(d, _)
            | Self::Move(d, _)
            | Self::Alu(d, ..)
            | Self::PtrOffset(d, ..)
            | Self::Load(d, ..)
            | Self::Call(d, _) => Some(d),
            Self::Store(..) | Self::BoundsCheck(..) => None,
        }
    }
    pub(super) fn inputs(self, calls: &DenseMap<CallSiteId, CallSite<R>>, mut f: impl FnMut(R)) {
        match self {
            Self::Move(_, v) | Self::Load(_, _, v) => f(v),
            Self::Alu(_, _, a, b)
            | Self::PtrOffset(_, a, b, _)
            | Self::BoundsCheck(a, b)
            | Self::Store(_, a, b) => {
                f(a);
                f(b);
            }
            Self::Call(_, site) => calls[site].args.iter().copied().for_each(f),
            _ => {}
        }
    }
}

pub(super) struct CallSite<R> {
    pub target: FunctionId,
    pub args: Box<[R]>,
}
