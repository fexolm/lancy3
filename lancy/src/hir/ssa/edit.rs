use super::{Block, Call, END, Edge, Function, Inst, Op, Terminator, Value};
use std::sync::Arc;

impl Function {
    /// Pass-internal replacement. The caller preserves result shape and
    /// verifies the completed pass as a unit.
    pub(crate) fn set_op_unchecked(&mut self, inst: Inst, op: Op) {
        let previous = std::mem::replace(&mut self.insts[inst.index()].op, op);
        debug_assert_eq!(previous.produces_value(), op.produces_value());
        self.fallible_ops -= usize::from(self.fallible(previous));
        self.fallible_ops += usize::from(self.fallible(op));
    }

    pub(crate) fn set_callee(&mut self, call: Call, callee: Arc<Self>) {
        let data = &mut self.calls[call.index()];
        let old = data.callee.needs_status();
        let new = callee.needs_status();
        data.callee = callee;
        if old != new {
            let uses = self.insts.iter().filter(|i| i.op == Op::Call(call)).count();
            if new {
                self.fallible_ops += uses;
            } else {
                self.fallible_ops -= uses;
            }
        }
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
            undo.push((inst, *self.op(inst)));
            self.set_op_unchecked(inst, op);
        }
        if let Err(err) = self.verify() {
            for (inst, previous) in undo.into_iter().rev() {
                self.set_op_unchecked(inst, previous);
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

        let mut aliases = Vec::new();
        self.splice(pred, edge, &mut aliases);
        self.substitute_params(aliases);
        Ok(())
    }

    /// Merge all chains of unconditional edges whose targets have a single
    /// predecessor. Count incoming edges once, link instruction lists without
    /// relocating them, then substitute all retired parameters in one pass.
    pub fn merge_linear_blocks(&mut self) -> Result<usize, String> {
        self.verify()?;
        Ok(self.merge_verified_linear_blocks())
    }

    pub(crate) fn merge_verified_linear_blocks(&mut self) -> usize {
        if self.blocks.len() == 1 {
            return 0;
        }
        let mut incoming = vec![0usize; self.block_capacity()];
        for block in self.blocks() {
            for edge in self.terminator(block).edges() {
                incoming[self.edges[edge.index()].0.index()] += 1;
            }
        }
        let mut aliases = Vec::new();
        let mut merged = 0;
        for i in 0..self.blocks.len() {
            let pred = Block(i as u32);
            while let Some(Terminator::Jump(edge)) = self.blocks[i].as_ref().and_then(|b| b.term) {
                let target = self.edges[edge.index()].0;
                if target == pred || incoming[target.index()] != 1 {
                    break;
                }
                self.splice(pred, edge, &mut aliases);
                merged += 1;
            }
        }
        self.substitute_params(aliases);
        merged
    }

    fn splice(&mut self, pred: Block, edge: Edge, aliases: &mut Vec<Value>) {
        let (succ, args) = self.edges[edge.index()];
        let succ = self.blocks[succ.index()].take().unwrap();
        if !succ.params.is_empty() {
            if aliases.is_empty() {
                aliases.extend((0..self.values).map(Value));
            }
            for (&param, &arg) in succ.params.iter().zip(&self.operands[args.range()]) {
                aliases[param.index()] = arg;
            }
        }
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
    }

    pub(super) fn substitute_params(&mut self, mut aliases: Vec<Value>) {
        if aliases.is_empty() {
            return;
        }
        // Edges can themselves pass parameters of previously merged blocks.
        // Compress each chain, then visits to already-resolved chains are O(1).
        for i in 0..aliases.len() {
            let mut root = aliases[i];
            while aliases[root.index()] != root {
                root = aliases[root.index()];
            }
            let mut cursor = Value(i as u32);
            while cursor != root {
                cursor = std::mem::replace(&mut aliases[cursor.index()], root);
            }
        }
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
