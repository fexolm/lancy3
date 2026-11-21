use super::{ControlFlowGraph, DomTree};
use crate::hir::Block;
use crate::support::slotmap::DenseMap;

/// Loop nesting rooted at backedges whose target dominates their source.
pub(crate) struct NaturalLoops {
    pub(crate) headers: Vec<Block>,
    pub(crate) owner: DenseMap<Block, Block>,
    pub(crate) parent: DenseMap<Block, Block>,
    pub(crate) preheader: DenseMap<Block, Block>,
}

impl NaturalLoops {
    pub(crate) fn discover(cfg: &ControlFlowGraph, dom: &DomTree) -> Self {
        let capacity = cfg.block_capacity();
        let mut loops = Self {
            headers: Vec::new(),
            owner: DenseMap::filled(capacity, Block::INVALID),
            parent: DenseMap::filled(capacity, Block::INVALID),
            preheader: DenseMap::filled(capacity, Block::INVALID),
        };
        let mut representative = DenseMap::from_fn(capacity, |block| block);
        let mut pending = Vec::new();
        // Walk innermost headers first, collapsing each discovered region
        // to its header for the enclosing predecessor walks.
        for &header in cfg.reverse_postorder().iter().rev() {
            let mut entry = None;
            let mut multiple_entries = false;
            for &pred in cfg.predecessors(header) {
                if dom.dominates(header, pred) {
                    pending.push(pred);
                } else {
                    multiple_entries |= entry.replace(pred).is_some();
                }
            }
            if pending.is_empty() {
                continue;
            }
            loops.headers.push(header);
            loops.owner[header] = header;
            if let Some(entry) = entry
                && !multiple_entries
                && cfg.successors(entry).count() == 1
                && dom.dominates(entry, header)
            {
                loops.preheader[header] = entry;
            }
            while let Some(mut root) = pending.pop() {
                while representative[root] != root {
                    let parent = representative[root];
                    representative[root] = representative[parent];
                    root = representative[root];
                }
                if root == header || !dom.dominates(header, root) {
                    continue;
                }
                representative[root] = header;
                if loops.owner[root] == root {
                    loops.parent[root] = header;
                } else {
                    loops.owner[root] = header;
                }
                pending.extend_from_slice(cfg.predecessors(root));
            }
        }
        loops
    }

    pub(crate) fn header_depths(&self) -> DenseMap<Block, usize> {
        let mut depths = DenseMap::filled(self.owner.len(), 0);
        for &header in self.headers.iter().rev() {
            depths[header] = 1 + depths.get(self.parent[header]).copied().unwrap_or(0);
        }
        depths
    }

    pub(crate) fn contains(&self, header: Block, block: Block) -> bool {
        let mut owner = self.owner[block];
        while owner != Block::INVALID && owner != header {
            owner = self.parent[owner];
        }
        owner == header
    }

    /// A split edge is inside exactly the loops containing both endpoints.
    pub(crate) fn edge_depth(
        &self,
        depths: &DenseMap<Block, usize>,
        source: Block,
        target: Block,
    ) -> usize {
        let mut from = self.owner[source];
        let mut to = self.owner[target];
        let depth = |header| depths.get(header).copied().unwrap_or(0);
        while from != to {
            if depth(from) >= depth(to) {
                from = self.parent[from];
            } else {
                to = self.parent[to];
            }
        }
        depth(from)
    }
}
