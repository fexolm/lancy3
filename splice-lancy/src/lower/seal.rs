use super::{Builder, Exit};
use lancy::hir::{Block, Function, Terminator, Value};
use splice::hir::Reg;

impl Builder<'_> {
    /// All predecessors are known now. Walk backwards through single-entry
    /// blocks, path-compressing live-through names. A merge gets a parameter
    /// immediately to break cycles, and its incoming operands join the worklist.
    fn reaching(&mut self, start: Block, r: Reg, preds: &[Vec<Block>]) -> Result<Value, String> {
        let mut path = Vec::new();
        let mut b = start;
        let value = loop {
            if let Some(&v) = self.blocks[b.index()].defs.get(&r) {
                break v;
            }
            match preds[b.index()].as_slice() {
                [] => return Err(format!("uninitialized {r:?} on path to {start:?}")),
                &[parent] => {
                    path.push(b);
                    b = parent;
                }
                _ => break self.parameter(b, r),
            }
        };
        for b in path {
            self.blocks[b.index()].defs.insert(r, value);
        }
        Ok(value)
    }

    pub(super) fn seal(mut self) -> Result<Function, String> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if self.labels.iter().any(|l| l.block.is_some() && !l.marked) {
            return Err("undefined kernel label".into());
        }
        let mut pending = vec![self.graph.entry()];
        let mut live = vec![false; self.blocks.len()];
        let mut preds = vec![Vec::new(); self.blocks.len()];
        while let Some(b) = pending.pop() {
            if std::mem::replace(&mut live[b.index()], true) {
                continue;
            }
            let term = self.blocks[b.index()]
                .exit
                .ok_or("reachable kernel path falls off the end")?;
            for next in term.successors() {
                if preds[next.index()].last() != Some(&b) {
                    preds[next.index()].push(b);
                }
                pending.push(next);
            }
        }
        while let Some((block, r)) = self.demands.pop() {
            if live[block.index()] {
                for &pred in &preds[block.index()] {
                    self.reaching(pred, r, &preds)?;
                }
            }
        }
        for (i, &reachable) in live.iter().enumerate() {
            if !reachable {
                continue;
            }
            let state = &self.blocks[i];
            let mut edge = |target: Block| {
                let args: Vec<_> = self.blocks[target.index()]
                    .incoming
                    .iter()
                    .map(|r| state.defs[r])
                    .collect();
                self.graph.add_edge(target, &args)
            };
            let term = match state.exit.unwrap() {
                Exit::Jump(b) => Terminator::Jump(edge(b)),
                Exit::Branch(cond, yes, no) => Terminator::Branch {
                    cond,
                    yes: edge(yes),
                    no: edge(no),
                },
                Exit::Return(v) => Terminator::Return(v),
            };
            self.graph.set_terminator(Block(i as u32), term);
        }
        if live.contains(&false) {
            self.graph.prune_unreachable()?;
        }
        self.graph.verify()?;
        Ok(self.graph)
    }
}
