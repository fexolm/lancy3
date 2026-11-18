//! Instruction-selected LIR. The same opcodes are first parameterized by SSA
//! values and later by machine homes. Calls and edge arguments are out of line.
use crate::hir::{
    Binary, Buffer,
    ssa::{self, Function, Value},
};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(super) enum Op<R> {
    Imm(R, i32),
    Move(R, R),
    Alu(R, Binary, R, R),
    Len(R, Buffer),
    Load(R, Buffer, R),
    Store(Buffer, R, R),
    Call(R, u32),
}

// Keep the common pre-allocation stream at four 32-bit words per operation.
const _: () = assert!(std::mem::size_of::<Op<Value>>() <= 16);

impl<R: Copy> Op<R> {
    pub(super) fn map<S>(self, mut f: impl FnMut(R) -> S) -> Op<S> {
        match self {
            Self::Imm(d, n) => Op::Imm(f(d), n),
            Self::Move(d, x) => Op::Move(f(d), f(x)),
            Self::Alu(d, op, a, b) => Op::Alu(f(d), op, f(a), f(b)),
            Self::Len(d, b) => Op::Len(f(d), b),
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
            | Self::Len(d, _)
            | Self::Load(d, ..)
            | Self::Call(d, _) => Some(d),
            Self::Store(..) => None,
        }
    }
    pub(super) fn inputs(self, calls: &[Site<R>], mut f: impl FnMut(R)) {
        match self {
            Self::Move(_, v) | Self::Load(_, _, v) => f(v),
            Self::Alu(_, _, a, b) | Self::Store(_, a, b) => {
                f(a);
                f(b);
            }
            Self::Call(_, i) => calls[i as usize].args.iter().copied().for_each(f),
            _ => {}
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Condition<R> {
    Nonzero(R),
    Compare(Binary, R, R),
    Mask { source: R, mask: i32, nonzero: bool },
}
impl<R: Copy> Condition<R> {
    pub(super) fn map<S>(self, mut f: impl FnMut(R) -> S) -> Condition<S> {
        match self {
            Self::Nonzero(r) => Condition::Nonzero(f(r)),
            Self::Compare(op, a, b) => Condition::Compare(op, f(a), f(b)),
            Self::Mask {
                source,
                mask,
                nonzero,
            } => Condition::Mask {
                source: f(source),
                mask,
                nonzero,
            },
        }
    }
}
#[derive(Clone, Copy)]
pub(super) enum Exit<R> {
    Return(R),
    Jump(usize),
    Branch(Condition<R>, usize, usize),
}
impl<R: Copy> Exit<R> {
    pub(super) fn successors(self) -> impl Iterator<Item = usize> {
        let (edges, count) = match self {
            Self::Return(_) => ([0, 0], 0),
            Self::Jump(edge) => ([edge, 0], 1),
            Self::Branch(_, a, b) => ([a, b], 2),
        };
        edges.into_iter().take(count)
    }
}
pub(super) struct Block<R> {
    pub ops: Range<usize>,
    pub exit: Exit<R>,
}
pub(super) struct Site<R> {
    pub target: usize,
    pub args: Box<[R]>,
    pub buffers: Box<[Buffer]>,
}
pub(super) struct Edge<'a> {
    pub target: usize,
    pub args: &'a [Value],
}

pub(super) struct SsaBody<'a> {
    pub blocks: Vec<Block<Value>>,
    pub ops: Vec<Op<Value>>,
    pub params: Vec<&'a [Value]>,
    pub edges: Vec<Edge<'a>>,
    pub calls: Vec<Site<Value>>,
    pub values: usize,
}

impl<'a> SsaBody<'a> {
    /// Select operations into a compact block layout. Fold unique flag-producing
    /// comparisons into branches before computing liveness, so their actual
    /// operands live up to the branch and no boolean register is necessary.
    pub fn lower(f: &'a Function, callees: &HashMap<*const Function, usize>) -> Self {
        let mut defs = vec![None; f.value_count()];
        let mut uses = vec![0; f.value_count()];
        let mut mapping = vec![0; f.block_capacity()];
        for (n, b) in f.blocks().enumerate() {
            mapping[b.index()] = n;
            for i in f.instructions(b) {
                let op = *f.op(i);
                if let Some(d) = f.result(i) {
                    defs[d.index()] = Some(op);
                }
                let mut used = |v: Value| uses[v.index()] += 1;
                match op {
                    ssa::Op::Copy(v) | ssa::Op::Load(_, v) => used(v),
                    ssa::Op::Bin(_, a, b) | ssa::Op::Store(_, a, b) => {
                        used(a);
                        used(b);
                    }
                    ssa::Op::Call(site) => f.call(site).1.iter().copied().for_each(used),
                    _ => {}
                }
            }
            let term = *f.terminator(b);
            match term {
                ssa::Terminator::Branch { cond, .. } | ssa::Terminator::Return(cond) => {
                    uses[cond.index()] += 1
                }
                _ => {}
            }
            for e in term.edges() {
                for v in f.edge(e).1 {
                    uses[v.index()] += 1;
                }
            }
        }
        let mut omitted = vec![false; uses.len()];
        let mut body = Self {
            blocks: Vec::new(),
            ops: Vec::new(),
            params: Vec::new(),
            edges: Vec::new(),
            calls: Vec::new(),
            values: uses.len(),
        };
        for b in f.blocks() {
            body.params.push(f.params(b));
            let mut edge = |e| {
                let (target, args) = f.edge(e);
                let id = body.edges.len();
                body.edges.push(Edge {
                    target: mapping[target.index()],
                    args,
                });
                id
            };
            let exit = match *f.terminator(b) {
                ssa::Terminator::Return(v) => Exit::Return(v),
                ssa::Terminator::Jump(e) => Exit::Jump(edge(e)),
                ssa::Terminator::Branch { cond, yes, no } => Exit::Branch(
                    condition(cond, &defs, &uses, &mut omitted),
                    edge(yes),
                    edge(no),
                ),
            };
            body.blocks.push(Block { ops: 0..0, exit });
        }
        for (id, b) in f.blocks().enumerate() {
            let start = body.ops.len();
            for i in f.instructions(b) {
                let dst = f.result(i);
                if dst.is_some_and(|v| omitted[v.index()]) {
                    continue;
                }
                body.ops.push(match *f.op(i) {
                    ssa::Op::Const(n) => Op::Imm(dst.unwrap(), n),
                    ssa::Op::Copy(v) => Op::Move(dst.unwrap(), v),
                    ssa::Op::Bin(op, a, b) => Op::Alu(dst.unwrap(), op, a, b),
                    ssa::Op::BufferLen(b) => Op::Len(dst.unwrap(), b),
                    ssa::Op::Load(b, v) => Op::Load(dst.unwrap(), b, v),
                    ssa::Op::Store(b, i, v) => Op::Store(b, i, v),
                    ssa::Op::Call(c) => {
                        let (child, args, buffers) = f.call(c);
                        // The number of sites is bounded by the verified u32 HIR instruction IDs.
                        let site = body.calls.len() as u32;
                        body.calls.push(Site {
                            target: callees[&Arc::as_ptr(child)],
                            args: args.into(),
                            buffers: buffers.into(),
                        });
                        Op::Call(dst.unwrap(), site)
                    }
                });
            }
            body.blocks[id].ops = start..body.ops.len();
        }
        body
    }
}

fn condition(
    v: Value,
    defs: &[Option<ssa::Op>],
    uses: &[usize],
    omitted: &mut [bool],
) -> Condition<Value> {
    use Binary::{And, Eq, Ge, Gt, Le, Lt, Ne};
    use ssa::Op::{Bin, Const};
    if uses[v.index()] != 1 {
        return Condition::Nonzero(v);
    }
    let Some(Bin(cmp @ (Eq | Ne | Lt | Le | Gt | Ge), a, b)) = defs[v.index()] else {
        return Condition::Nonzero(v);
    };
    omitted[v.index()] = true;
    if matches!(cmp, Eq | Ne) {
        for (test, zero) in [(a, b), (b, a)] {
            if uses[test.index()] != 1 || defs[zero.index()] != Some(Const(0)) {
                continue;
            }
            if let Some(Bin(And, x, y)) = defs[test.index()] {
                for (source, literal) in [(x, y), (y, x)] {
                    if let Some(Const(mask)) = defs[literal.index()] {
                        omitted[test.index()] = true;
                        return Condition::Mask {
                            source,
                            mask,
                            nonzero: cmp == Ne,
                        };
                    }
                }
            }
        }
    }
    Condition::Compare(cmp, a, b)
}
