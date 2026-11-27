use super::ControlFlowGraph;
use crate::hir::ssa::Block;
use crate::support::slotmap::DenseMap;
use std::ops::Range;

/// Block dominance for a snapshot of the control-flow graph.
#[derive(Clone)]
pub(crate) struct DomTree {
    // Each interval covers the block and all its descendants in preorder.
    intervals: DenseMap<Block, Range<usize>>,
    preorder: Vec<Block>,
}

impl DomTree {
    pub(crate) fn compute(cfg: &ControlFlowGraph) -> Self {
        let order = cfg.reverse_postorder();
        let parent = immediate_dominators(cfg);
        let mut children = DenseMap::from_fn(cfg.block_capacity(), |_| Vec::new());
        for &block in &order[1..] {
            children[parent[block]].push(block);
        }
        let mut intervals = DenseMap::filled(cfg.block_capacity(), 0..0);
        let mut preorder = Vec::with_capacity(order.len());
        let mut stack = vec![(order[0], false)];
        while let Some((block, leave)) = stack.pop() {
            let interval = &mut intervals[block];
            if leave {
                interval.end = preorder.len();
            } else {
                interval.start = preorder.len();
                preorder.push(block);
                stack.push((block, true));
                stack.extend(children[block].iter().map(|&child| (child, false)));
            }
        }
        Self {
            intervals,
            preorder,
        }
    }

    pub(crate) fn preorder(&self) -> &[Block] {
        &self.preorder
    }

    pub(crate) fn scope(&self, block: Block) -> Range<usize> {
        self.intervals[block].clone()
    }

    /// Whether every path from the entry to `block` passes through `dominator`.
    /// Returns false for IDs outside the captured graph or retired before capture.
    pub(crate) fn dominates(&self, dominator: Block, block: Block) -> bool {
        match (self.intervals.get(dominator), self.intervals.get(block)) {
            (Some(parent), Some(child)) => !child.is_empty() && parent.contains(&child.start),
            _ => false,
        }
    }
}

/// Cooper-Harvey-Kennedy iteration in reverse postorder.
fn immediate_dominators(cfg: &ControlFlowGraph) -> DenseMap<Block, Block> {
    let order = cfg.reverse_postorder();
    let mut rank = DenseMap::filled(cfg.block_capacity(), usize::MAX);
    let mut idom = DenseMap::filled(cfg.block_capacity(), Block::INVALID);
    for (i, &block) in order.iter().enumerate() {
        rank[block] = i;
    }
    let entry = order[0];
    idom[entry] = entry;
    loop {
        let mut changed = false;
        for &block in &order[1..] {
            let mut preds = cfg
                .predecessors(block)
                .iter()
                .copied()
                .filter(|&p| idom[p] != Block::INVALID);
            let mut common = preds
                .next()
                .expect("reachable block has processed predecessor in RPO");
            for mut other in preds {
                while common != other {
                    if rank[common] > rank[other] {
                        common = idom[common];
                    } else {
                        other = idom[other];
                    }
                }
            }
            changed |= idom[block] != common;
            idom[block] = common;
        }
        if !changed {
            return idom;
        }
    }
}
