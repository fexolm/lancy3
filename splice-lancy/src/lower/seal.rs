use super::{Builder, Exit};
use lancy::hir::{Block, Function, Op, Terminator, Value};
use lancy::support::slotmap::DenseMap;
use splice_core::hir::Reg;

impl Builder<'_> {
    /// All predecessors are known now. Walk backwards through single-entry
    /// blocks, path-compressing live-through names. A merge gets a parameter
    /// immediately to break cycles, and its incoming operands join the worklist.
    /// A path without an assignment is an error for a required value; a value
    /// that only source points observe gets a zero placeholder instead.
    fn reaching(
        &mut self,
        start: Block,
        r: Reg,
        required: bool,
        preds: &DenseMap<Block, Vec<Block>>,
    ) -> Result<Value, String> {
        let unassigned = || format!("uninitialized {r:?} on path to {start:?}");
        let mut path = Vec::new();
        let mut b = start;
        let value = loop {
            if let Some(&v) = self.blocks[b].defs.get(&r) {
                if required {
                    if self.unassigned.contains(&v) {
                        return Err(unassigned());
                    }
                    self.require(v);
                }
                break v;
            }
            match preds[b].as_slice() {
                [] if required => return Err(unassigned()),
                [] => {
                    let zero = self.graph.append(b, Op::Const(0)).unwrap();
                    self.unassigned.insert(zero);
                    self.blocks[b].defs.insert(r, zero);
                    break zero;
                }
                &[parent] => {
                    path.push(b);
                    b = parent;
                }
                _ => break self.parameter(b, r, required),
            }
        };
        for b in path {
            self.blocks[b].defs.insert(r, value);
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
        let mut live = DenseMap::filled(self.blocks.len(), false);
        let mut preds = DenseMap::from_fn(self.blocks.len(), |_| Vec::new());
        while let Some(b) = pending.pop() {
            if std::mem::replace(&mut live[b], true) {
                continue;
            }
            let term = self.blocks[b]
                .exit
                .ok_or("reachable kernel path falls off the end")?;
            for next in term.successors() {
                if preds[next].last() != Some(&b) {
                    preds[next].push(b);
                }
                pending.push(next);
            }
        }
        while let Some(demand) = self.demands.pop() {
            if live[demand.block] {
                for &pred in &preds[demand.block] {
                    self.reaching(pred, demand.reg, demand.required, &preds)?;
                }
            }
        }
        for (block, &reachable) in live.iter() {
            if !reachable {
                continue;
            }
            let state = &self.blocks[block];
            let mut edge = |target: Block| {
                let args: Vec<_> = self.blocks[target]
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
            self.graph.set_terminator(block, term);
        }
        if live.as_slice().contains(&false) {
            self.graph.prune_unreachable().map_err(|e| e.to_string())?;
        }
        self.graph.verify().map_err(|e| e.to_string())?;
        Ok(self.graph)
    }
}
