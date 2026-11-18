use crate::hir::{self, Function, Value};
use crate::lir::{
    BlockId, CallSiteId, EdgeId, FunctionId,
    block::{Block, Exit},
    instruction::{CallSite, Op},
};
use crate::support::slotmap::DenseMap;
use std::collections::HashMap;
use std::sync::Arc;

pub(super) struct Edge<'a> {
    pub target: BlockId,
    pub args: &'a [Value],
}

pub(super) struct SsaBody<'a> {
    pub entry: BlockId,
    pub blocks: DenseMap<BlockId, Block<Value, EdgeId>>,
    pub ops: Vec<Op<Value>>,
    pub params: DenseMap<BlockId, &'a [Value]>,
    pub edges: DenseMap<EdgeId, Edge<'a>>,
    pub calls: DenseMap<CallSiteId, CallSite<Value>>,
    pub values: usize,
}

impl<'a> SsaBody<'a> {
    /// Copy verified HIR operands and terminators into dense LIR blocks.
    pub fn lower(f: &'a Function, callees: &HashMap<*const Function, FunctionId>) -> Self {
        let mut mapping = DenseMap::filled(f.block_capacity(), BlockId::INVALID);
        let mut params = DenseMap::new();
        for block in f.blocks() {
            mapping[block] = params.insert(f.params(block));
        }
        let mut body = Self {
            entry: mapping[f.entry()],
            blocks: DenseMap::new(),
            ops: Vec::new(),
            params,
            edges: DenseMap::new(),
            calls: DenseMap::new(),
            values: f.value_count(),
        };
        for b in f.blocks() {
            let mut edge = |e| {
                let (target, args) = f.edge(e);
                body.edges.insert(Edge {
                    target: mapping[target],
                    args,
                })
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
                    hir::Op::BoundsCheck(i, n) => Op::BoundsCheck(i, n),
                    hir::Op::PtrOffset(base, i, scale) => {
                        Op::PtrOffset(dst.unwrap(), base, i, scale)
                    }
                    hir::Op::Load(ty, addr) => Op::Load(dst.unwrap(), ty, addr),
                    hir::Op::Store(addr, v) => Op::Store(f.value_type(v), addr, v),
                    hir::Op::Call(c) => {
                        let (child, args) = f.call(c);
                        let site = body.calls.insert(CallSite {
                            target: callees[&Arc::as_ptr(child)],
                            args: args.into(),
                        });
                        Op::Call(dst.unwrap(), site)
                    }
                });
            }
            let block = body.blocks.insert(Block {
                ops: start..body.ops.len(),
                exit,
            });
            debug_assert_eq!(block, mapping[b]);
        }
        body
    }
}
