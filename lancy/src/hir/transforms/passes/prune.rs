use crate::hir::{Function, IrError};
use crate::support::slotmap::DenseMap;

impl Function {
    /// Remove unreachable blocks without renumbering surviving instructions,
    /// blocks, or values. Only reachable blocks must have terminators.
    pub fn prune_unreachable(&mut self) -> Result<usize, IrError> {
        let mut pending = vec![self.entry()];
        let mut live = DenseMap::filled(self.block_capacity(), false);
        while let Some(b) = pending.pop() {
            if std::mem::replace(&mut live[b], true) {
                continue;
            }
            for edge in self.checked_terminator(b)?.edges() {
                pending.push(self.checked_target(b, edge)?);
            }
        }
        let mut count = 0;
        for (block, &reachable) in live.iter() {
            if !reachable && self.remove_block(block) {
                count += 1;
            }
        }
        Ok(count)
    }
}
