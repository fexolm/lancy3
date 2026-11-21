use crate::hir::{Block, Edge, Function, IrError};
use crate::support::slotmap::DenseMap;
use std::ops::Range;

/// Edges and reverse postorder of a function's reachable blocks.
/// Rebuild after changing blocks or terminators.
pub struct ControlFlowGraph {
    successors: DenseMap<Block, [Option<Block>; 2]>,
    incoming: DenseMap<Block, Range<usize>>,
    predecessors: Vec<Block>,
    edges: Vec<Edge>,
    reverse_postorder: Vec<Block>,
    has_cycle: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Visit {
    New,
    Active,
    Done,
}

impl ControlFlowGraph {
    /// Check the terminators, destinations and argument counts while building the graph.
    pub fn compute(function: &Function) -> Result<Self, IrError> {
        let capacity = function.block_capacity();
        let mut incoming = DenseMap::filled(capacity, 0..0);
        let mut successors = DenseMap::filled(capacity, [None; 2]);
        let mut seen = DenseMap::filled(capacity, Visit::New);
        let mut live_count = 0;
        for block in function.blocks() {
            let targets = function.checked_successors(block)?;
            for target in targets.into_iter().flatten() {
                incoming[target].end += 1;
            }
            successors[block] = targets;
            live_count += 1;
        }
        let (reverse_postorder, has_cycle) =
            reverse_postorder(function.entry(), &successors, &mut seen, live_count);
        if reverse_postorder.len() != live_count {
            let block = function.blocks().find(|&b| seen[b] == Visit::New).unwrap();
            return Err(IrError::UnreachableBlock { block });
        }

        // Reserve contiguous spans, then fill them in source and arm order.
        let mut total = 0;
        for range in incoming.values_mut() {
            let count = range.end;
            *range = total..total;
            total += count;
        }
        let mut predecessors = vec![Block::INVALID; total];
        let mut edges = vec![Edge::INVALID; total];
        for block in function.blocks() {
            let targets = successors[block].into_iter().flatten();
            for (target, edge) in targets.zip(function.terminator(block).edges()) {
                let span = &mut incoming[target];
                predecessors[span.end] = block;
                edges[span.end] = edge;
                span.end += 1;
            }
        }
        Ok(Self {
            successors,
            incoming,
            predecessors,
            edges,
            reverse_postorder,
            has_cycle,
        })
    }

    pub fn successors(&self, block: Block) -> impl Iterator<Item = Block> {
        self.successors[block].into_iter().flatten()
    }

    pub fn predecessors(&self, block: Block) -> &[Block] {
        &self.predecessors[self.incoming[block].clone()]
    }

    pub(crate) fn incoming_edges(&self, block: Block) -> impl Iterator<Item = (Block, Edge)> + '_ {
        self.predecessors(block)
            .iter()
            .copied()
            .zip(self.edges[self.incoming[block].clone()].iter().copied())
    }

    pub fn reverse_postorder(&self) -> &[Block] {
        &self.reverse_postorder
    }

    pub(crate) fn has_cycle(&self) -> bool {
        self.has_cycle
    }

    pub(super) fn block_capacity(&self) -> usize {
        self.successors.len()
    }
}

fn reverse_postorder(
    entry: Block,
    successors: &DenseMap<Block, [Option<Block>; 2]>,
    seen: &mut DenseMap<Block, Visit>,
    live_count: usize,
) -> (Vec<Block>, bool) {
    let mut pending = vec![(entry, false)];
    let mut order = Vec::with_capacity(live_count);
    let mut has_cycle = false;
    while let Some((block, exiting)) = pending.pop() {
        if exiting {
            seen[block] = Visit::Done;
            order.push(block);
        } else if seen[block] == Visit::New {
            seen[block] = Visit::Active;
            pending.push((block, true));
            pending.extend(
                successors[block]
                    .into_iter()
                    .rev()
                    .flatten()
                    .map(|b| (b, false)),
            );
        } else if seen[block] == Visit::Active {
            has_cycle = true;
        }
    }
    order.reverse();
    (order, has_cycle)
}
