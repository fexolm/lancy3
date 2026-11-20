use crate::hir::{Block, Function, IrError};
use crate::support::slotmap::DenseMap;

/// Edges and reverse postorder of a function's reachable blocks.
/// Rebuild after changing blocks or terminators.
pub struct ControlFlowGraph {
    successors: DenseMap<Block, [Option<Block>; 2]>,
    predecessors: DenseMap<Block, Vec<Block>>,
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
        let mut predecessors = DenseMap::from_fn(capacity, |_| Vec::new());
        let mut successors = DenseMap::filled(capacity, [None; 2]);
        let mut seen = DenseMap::filled(capacity, Visit::New);
        let mut live_count = 0;
        for block in function.blocks() {
            let targets = function.checked_successors(block)?;
            for target in targets.into_iter().flatten() {
                predecessors[target].push(block);
            }
            successors[block] = targets;
            live_count += 1;
        }
        let (reverse_postorder, has_cycle) =
            reverse_postorder(function.entry(), &successors, &mut seen);
        if reverse_postorder.len() != live_count {
            let block = function.blocks().find(|&b| seen[b] == Visit::New).unwrap();
            return Err(IrError::UnreachableBlock { block });
        }
        Ok(Self {
            successors,
            predecessors,
            reverse_postorder,
            has_cycle,
        })
    }

    pub fn successors(&self, block: Block) -> impl Iterator<Item = Block> {
        self.successors[block].into_iter().flatten()
    }

    pub fn predecessors(&self, block: Block) -> &[Block] {
        &self.predecessors[block]
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
) -> (Vec<Block>, bool) {
    let mut pending = vec![(entry, false)];
    let mut order = Vec::new();
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
