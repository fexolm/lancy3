use super::{Block, END, Function, Inst, Op, Terminator, Value};

impl Function {
    /// Remove unreachable blocks without renumbering surviving instructions,
    /// blocks, or values. Only reachable blocks must have terminators.
    pub fn prune_unreachable(&mut self) -> Result<usize, String> {
        let mut pending = vec![self.entry()];
        let mut live = vec![false; self.block_capacity()];
        while let Some(b) = pending.pop() {
            if std::mem::replace(&mut live[b.index()], true) {
                continue;
            }
            let term = self
                .block(b)
                .term
                .ok_or("reachable block has no terminator")?;
            for e in term.edges() {
                let target = self.edges.get(e.index()).ok_or("invalid SSA edge")?.0;
                self.blocks
                    .get(target.index())
                    .and_then(Option::as_ref)
                    .ok_or("invalid SSA target")?;
                pending.push(target);
            }
        }
        let mut count = 0;
        for (i, reachable) in live.into_iter().enumerate() {
            if reachable {
                continue;
            }
            if let Some(dead) = self.blocks[i].take() {
                count += 1;
                let mut cursor = dead.first;
                while cursor != END {
                    let index = cursor.index();
                    let inst = &mut self.insts[index];
                    // A self-link retires the instruction while keeping other IDs stable.
                    inst.op = Op::Const(0);
                    cursor = std::mem::replace(&mut inst.next, cursor);
                }
            }
        }
        Ok(count)
    }

    /// Replace an instruction and retain its position, ID and result ID.
    /// A rejected edit leaves the original instruction in place.
    pub fn replace_op(&mut self, inst: Inst, op: Op) -> Result<(), String> {
        self.replace_ops(&[(inst, op)])
    }

    /// Apply an entire pass worth of replacements in place with one verifier
    /// run. Repeated edits of the same instruction are replayed in order.
    pub fn replace_ops(&mut self, edits: &[(Inst, Op)]) -> Result<(), String> {
        for &(inst, op) in edits {
            let old = self
                .insts
                .get(inst.index())
                .ok_or("unknown SSA instruction")?;
            if old.next == inst {
                return Err("retired SSA instruction".into());
            }
            if old.op.produces_value() != op.produces_value() {
                return Err("replacement must preserve the SSA result count".into());
            }
        }
        let mut undo = Vec::with_capacity(edits.len());
        for &(inst, op) in edits {
            let old = std::mem::replace(&mut self.insts[inst.index()].op, op);
            undo.push((inst, old));
        }
        if let Err(err) = self.verify() {
            for (inst, previous) in undo.into_iter().rev() {
                self.insts[inst.index()].op = previous;
            }
            return Err(err);
        }
        Ok(())
    }

    /// Splice a successor with exactly one incoming edge into its predecessor.
    /// Live IDs are stable. The successor block and its block-parameter IDs
    /// are retired. Arguments substitute parameters simultaneously, including
    /// references in dominated blocks, calls and successor edges.
    pub fn merge_blocks(&mut self, pred: Block, succ: Block) -> Result<(), String> {
        self.verify()?;
        if pred == succ
            || self
                .blocks
                .get(succ.index())
                .and_then(Option::as_ref)
                .is_none()
        {
            return Err("expected a distinct live SSA successor".into());
        }
        let term = self
            .blocks
            .get(pred.index())
            .and_then(Option::as_ref)
            .and_then(|b| b.term);
        let Some(Terminator::Jump(edge)) = term else {
            return Err("SSA predecessor must end in an unconditional jump".into());
        };
        let (target, _) = self.edge(edge);
        if target != succ {
            return Err("SSA jump does not target the successor".into());
        }
        let incoming = self
            .blocks()
            .flat_map(|b| self.terminator(b).edges())
            .filter(|&e| self.edges[e.index()].0 == succ)
            .count();
        if incoming != 1 {
            return Err("SSA successor has multiple incoming edges".into());
        }

        let (_, args) = self.edges[edge.index()];
        let succ = self.blocks[succ.index()].take().unwrap();
        let last = self.block(pred).last;
        if succ.first != END {
            if last == END {
                self.block_mut(pred).first = succ.first;
            } else {
                self.insts[last.index()].next = succ.first;
            }
            self.block_mut(pred).last = succ.last;
        }
        self.block_mut(pred).term = succ.term;
        if !succ.params.is_empty() {
            let mut aliases: Vec<_> = (0..self.values).map(Value).collect();
            for (&param, &actual) in succ.params.iter().zip(&self.operands[args.range()]) {
                aliases[param.index()] = actual;
            }
            self.substitute_params(&aliases);
        }
        Ok(())
    }

    fn substitute_params(&mut self, aliases: &[Value]) {
        let map = |v: Value| aliases[v.index()];
        for inst in &mut self.insts {
            inst.op.map_operands(map);
        }
        for v in &mut self.operands {
            // Unreferenced call/edge records can hold speculative operands.
            if let Some(&alias) = aliases.get(v.index()) {
                *v = alias;
            }
        }
        for block in self.blocks.iter_mut().flatten() {
            if let Some(Terminator::Return(v) | Terminator::Branch { cond: v, .. }) =
                &mut block.term
            {
                *v = map(*v);
            }
        }
    }
}
