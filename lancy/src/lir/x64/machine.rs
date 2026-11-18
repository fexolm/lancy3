//! Destroy SSA after allocation. Edge assignments are parallel: on branches
//! they run on the selected edge, and cycles use RCX as temporary storage.
mod parallel_moves;

use super::frame::{Frame, Home};
use super::regalloc::Allocation;
use crate::lir::{
    BlockId, CallSiteId, EdgeId,
    block::{Block, Exit},
    instruction::{CallSite, Op},
    ssa::SsaBody,
};
use crate::support::slotmap::DenseMap;
use parallel_moves::schedule;

pub(super) struct Machine {
    pub blocks: DenseMap<BlockId, Block<Home, BlockId>>,
    pub ops: Vec<Op<Home>>,
    pub calls: DenseMap<CallSiteId, CallSite<Home>>,
    pub frame: Frame,
}
impl Machine {
    pub fn new(body: SsaBody, allocation: Allocation) -> Self {
        let home = |v: crate::hir::Value| allocation.homes[v];
        let mut ops = Vec::with_capacity(body.ops.len());
        let calls = DenseMap::from_fn(body.calls.len(), |site| {
            let c = &body.calls[site];
            CallSite {
                target: c.target,
                args: c.args.iter().copied().map(home).collect(),
            }
        });
        let mut bridges = Vec::new();
        let mut moves = Vec::new();
        let mut blocks = DenseMap::from_fn(body.blocks.len(), |id| {
            let b = &body.blocks[id];
            let start = ops.len();
            ops.extend(body.ops[b.ops.clone()].iter().map(|op| op.map(home)));
            let end = ops.len();
            let mut edge = |e: EdgeId, conditional: bool| {
                let edge = &body.edges[e];
                moves.clear();
                for (&from, &to) in edge.args.iter().zip(body.params[edge.target]) {
                    let (src, dst) = (home(from), home(to));
                    if src != dst {
                        moves.push((dst, src));
                    }
                }
                if moves.is_empty() {
                    return edge.target;
                }
                let begin = ops.len();
                schedule(&mut moves, &mut ops);
                if conditional {
                    let id = BlockId::from_index(body.blocks.len() + bridges.len());
                    bridges.push(Block {
                        ops: begin..ops.len(),
                        exit: Exit::Jump(edge.target),
                    });
                    id
                } else {
                    edge.target
                }
            };
            let exit = match b.exit {
                Exit::Return(r) => Exit::Return(home(r)),
                Exit::Jump(e) => Exit::Jump(edge(e, false)),
                Exit::Branch(c, a, b) => Exit::Branch(home(c), edge(a, true), edge(b, true)),
            };
            Block {
                ops: start..if matches!(exit, Exit::Jump(_)) {
                    ops.len()
                } else {
                    end
                },
                exit,
            }
        });
        for bridge in bridges {
            blocks.insert(bridge);
        }
        Self {
            ops,
            blocks,
            calls,
            frame: allocation.frame,
        }
    }
}
