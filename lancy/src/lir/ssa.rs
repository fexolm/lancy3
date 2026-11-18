//! Instruction-selected LIR. The same opcodes are first parameterized by SSA
//! values and later by machine homes. Calls and edge arguments are out of line.
use crate::hir::{self, Binary, Buffer, Function, Value};
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

#[derive(Clone, Copy)]
pub(super) enum Exit<R> {
    Return(R),
    Jump(usize),
    Branch(R, usize, usize),
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
    /// Copy verified HIR operands and terminators into dense LIR blocks.
    pub fn lower(f: &'a Function, callees: &HashMap<*const Function, usize>) -> Self {
        let mut mapping = vec![0; f.block_capacity()];
        for (index, block) in f.blocks().enumerate() {
            mapping[block.index()] = index;
        }
        let mut body = Self {
            blocks: Vec::new(),
            ops: Vec::new(),
            params: Vec::new(),
            edges: Vec::new(),
            calls: Vec::new(),
            values: f.value_count(),
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
                hir::Terminator::Return(v) => Exit::Return(v),
                hir::Terminator::Jump(e) => Exit::Jump(edge(e)),
                hir::Terminator::Branch { cond, yes, no } => {
                    Exit::Branch(cond, edge(yes), edge(no))
                }
            };

            let start = body.ops.len();
            for i in f.instructions(b) {
                let dst = f.result(i);
                body.ops.push(match *f.op(i) {
                    hir::Op::Const(n) => Op::Imm(dst.unwrap(), n),
                    hir::Op::Copy(v) => Op::Move(dst.unwrap(), v),
                    hir::Op::Bin(op, a, b) => Op::Alu(dst.unwrap(), op, a, b),
                    hir::Op::BufferLen(b) => Op::Len(dst.unwrap(), b),
                    hir::Op::Load(b, v) => Op::Load(dst.unwrap(), b, v),
                    hir::Op::Store(b, i, v) => Op::Store(b, i, v),
                    hir::Op::Call(c) => {
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
            body.blocks.push(Block {
                ops: start..body.ops.len(),
                exit,
            });
        }
        body
    }
}
