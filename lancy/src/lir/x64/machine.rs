//! Destroy SSA after allocation. Edge assignments are parallel: on branches
//! they run on the selected edge, and cycles use RCX as temporary storage.
use super::encode::ECX;
use super::frame::{Frame, Home};
use super::regalloc::Allocation;
use crate::lir::ssa::{Block, Exit, Op, Site, SsaBody};

pub(super) struct Machine {
    pub blocks: Vec<Block<Home>>,
    pub ops: Vec<Op<Home>>,
    pub calls: Vec<Site<Home>>,
    pub frame: Frame,
}
impl Machine {
    pub fn new(body: SsaBody, allocation: Allocation) -> Self {
        let home = |v: crate::hir::Value| allocation.homes[v.index()];
        let mut ops = Vec::with_capacity(body.ops.len());
        let calls = body
            .calls
            .iter()
            .map(|c| Site {
                target: c.target,
                args: c.args.iter().copied().map(home).collect(),
                buffers: c.buffers.clone(),
            })
            .collect();
        let mut blocks = Vec::with_capacity(body.blocks.len());
        let mut bridges = Vec::new();
        let mut moves = Vec::new();
        for b in &body.blocks {
            let start = ops.len();
            ops.extend(body.ops[b.ops.clone()].iter().map(|op| op.map(home)));
            let end = ops.len();
            let mut edge = |e: usize, conditional: bool| {
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
                    let index = body.blocks.len() + bridges.len();
                    bridges.push(Block {
                        ops: begin..ops.len(),
                        exit: Exit::Jump(edge.target),
                    });
                    index
                } else {
                    edge.target
                }
            };
            let exit = match b.exit {
                Exit::Return(r) => Exit::Return(home(r)),
                Exit::Jump(e) => Exit::Jump(edge(e, false)),
                Exit::Branch(c, a, b) => Exit::Branch(home(c), edge(a, true), edge(b, true)),
            };
            blocks.push(Block {
                ops: start..if matches!(exit, Exit::Jump(_)) {
                    ops.len()
                } else {
                    end
                },
                exit,
            });
        }
        blocks.extend(bridges);
        Self {
            ops,
            blocks,
            calls,
            frame: allocation.frame,
        }
    }
}
fn schedule(moves: &mut Vec<(Home, Home)>, ops: &mut Vec<Op<Home>>) {
    while !moves.is_empty() {
        if let Some(i) = moves
            .iter()
            .position(|(dst, _)| !moves.iter().any(|(_, src)| src == dst))
        {
            let (dst, src) = moves.swap_remove(i);
            ops.push(Op::Move(dst, src));
        } else {
            // All remaining moves participate in, or feed, a cycle. Preserve
            // an old destination before it is overwritten. Slot-to-slot MOV
            // uses EAX in the encoder, so ECX remains intact until consumed.
            let old = moves[0].0;
            let temp = Home::Register(ECX);
            ops.push(Op::Move(temp, old));
            for (_, src) in moves.iter_mut() {
                if *src == old {
                    *src = temp;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parallel_swap_supports_registers_slots_and_fanout() {
        let (a, b, c, d) = (
            Home::Register(8),
            Home::Slot(0),
            Home::Slot(1),
            Home::Register(9),
        );
        let mut ops = Vec::new();
        schedule(&mut vec![(a, b), (b, c), (c, a), (d, a)], &mut ops);
        let mut values = [(a, 1), (b, 2), (c, 3), (d, 4), (Home::Register(ECX), 0)];
        for op in ops {
            let Op::Move(dst, src) = op else {
                unreachable!()
            };
            let x = values.iter().find(|v| v.0 == src).unwrap().1;
            values.iter_mut().find(|v| v.0 == dst).unwrap().1 = x;
        }
        assert_eq!(values[..4], [(a, 2), (b, 3), (c, 1), (d, 1)]);
    }
}
