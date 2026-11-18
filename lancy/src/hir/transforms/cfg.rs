use crate::hir::{Block, Function, IrError};
use crate::support::slotmap::DenseMap;

/// Edges and reverse postorder of a function's reachable blocks.
/// Rebuild after changing blocks or terminators.
pub struct ControlFlowGraph {
    successors: DenseMap<Block, [Option<Block>; 2]>,
    predecessors: DenseMap<Block, Vec<Block>>,
    reverse_postorder: Vec<Block>,
}

impl ControlFlowGraph {
    /// Check the terminators, destinations and argument counts while building the graph.
    pub fn compute(function: &Function) -> Result<Self, IrError> {
        let capacity = function.block_capacity();
        let mut predecessors = DenseMap::from_fn(capacity, |_| Vec::new());
        let mut successors = DenseMap::filled(capacity, [None; 2]);
        let mut seen = DenseMap::filled(capacity, false);
        let mut live_count = 0;
        for block in function.blocks() {
            let targets = function.checked_successors(block)?;
            for target in targets.into_iter().flatten() {
                predecessors[target].push(block);
            }
            successors[block] = targets;
            live_count += 1;
        }
        let reverse_postorder = reverse_postorder(function.entry(), &successors, &mut seen);
        if reverse_postorder.len() != live_count {
            let block = function.blocks().find(|&b| !seen[b]).unwrap();
            return Err(IrError::UnreachableBlock { block });
        }
        Ok(Self {
            successors,
            predecessors,
            reverse_postorder,
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

    pub(super) fn block_capacity(&self) -> usize {
        self.successors.len()
    }
}

fn reverse_postorder(
    entry: Block,
    successors: &DenseMap<Block, [Option<Block>; 2]>,
    seen: &mut DenseMap<Block, bool>,
) -> Vec<Block> {
    let mut pending = vec![(entry, false)];
    let mut order = Vec::new();
    while let Some((block, exiting)) = pending.pop() {
        if exiting {
            order.push(block);
        } else if !std::mem::replace(&mut seen[block], true) {
            pending.push((block, true));
            pending.extend(
                successors[block]
                    .into_iter()
                    .rev()
                    .flatten()
                    .map(|b| (b, false)),
            );
        }
    }
    order.reverse();
    order
}
