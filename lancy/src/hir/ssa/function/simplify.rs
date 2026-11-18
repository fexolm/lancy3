use super::*;
use std::collections::HashMap;

impl Function {
    /// Canonicalize scalar dataflow before instruction selection. Division,
    /// memory operations and checks remain ordered on every executable path.
    pub(crate) fn simplify(&mut self) -> Result<(), IrError> {
        let mut cfg = ControlFlowGraph::compute(self)?;
        let mut aliases = DenseMap::from_fn(self.value_count(), |v| v);
        let mut constants = DenseMap::filled(self.value_count(), None);
        let mut changed_cfg = false;
        for &b in cfg.reverse_postorder() {
            let mut i = self.blocks[b].first;
            while i != Inst::INVALID {
                let inst = &mut self.insts[i];
                inst.op.map_operands(|v| aliases[v]);
                match inst.op {
                    Op::Copy(v) => aliases[inst.result] = aliases[v],
                    Op::Const(n) => constants[inst.result] = Some(n),
                    Op::Bin(op, x, y) => {
                        if let (Some(x), Some(y)) = (constants[x], constants[y])
                            && let Some(n) = op.checked_eval(x, y)
                        {
                            inst.op = Op::Const(n);
                            constants[inst.result] = Some(n);
                        }
                    }
                    _ => {}
                }
                i = inst.next;
            }
            if let Terminator::Branch { cond, yes, no } = *self.terminator(b)
                && let Some(n) = constants[aliases[cond]]
            {
                self.set_terminator(b, Terminator::Jump(if n != 0 { yes } else { no }));
                changed_cfg = true;
            }
        }
        self.substitute_params(&aliases);
        if changed_cfg {
            self.prune_unreachable()?;
            cfg = ControlFlowGraph::compute(self)?;
        }
        self.remove_redundant_checks(&DomTree::compute(&cfg));
        self.remove_dead_computations(&cfg);
        Ok(())
    }

    fn remove_redundant_checks(&mut self, dom: &DomTree) {
        let mut checks = HashMap::new();
        for &b in dom.preorder() {
            self.retain_instructions(b, |op, _| {
                if let Op::BoundsCheck(i, n) = op {
                    let previous = checks.entry((i, n)).or_insert(Block::INVALID);
                    if dom.dominates(*previous, b) {
                        return false;
                    }
                    *previous = b;
                }
                true
            });
        }
    }

    fn remove_dead_computations(&mut self, cfg: &ControlFlowGraph) {
        let mut live = DenseMap::filled(self.value_count(), false);
        for &b in cfg.reverse_postorder() {
            let term = *self.terminator(b);
            if let Terminator::Branch { cond: v, .. } | Terminator::Return(v) = term {
                live[v] = true;
            }
            for e in term.edges() {
                for &v in self.edge(e).1 {
                    live[v] = true;
                }
            }
        }
        let mut instructions = Vec::new();
        for &b in cfg.reverse_postorder().iter().rev() {
            instructions.extend(self.instructions(b));
            for i in instructions.drain(..).rev() {
                let op = *self.op(i);
                if op.pure() && !live[self.insts[i].result] {
                    continue;
                }
                self.visit_operands(i, |v| live[v] = true);
            }
            self.retain_instructions(b, |op, result| !op.pure() || live[result]);
        }
    }

    fn retain_instructions(&mut self, b: Block, mut keep: impl FnMut(Op, Value) -> bool) {
        let block = &mut self.blocks[b];
        let mut prev = Inst::INVALID;
        let mut i = block.first;
        while i != Inst::INVALID {
            let inst = &self.insts[i];
            let next = inst.next;
            if keep(inst.op, inst.result) {
                prev = i;
            } else {
                if prev == Inst::INVALID {
                    block.first = next;
                } else {
                    self.insts[prev].next = next;
                }
                self.insts.remove(i);
            }
            i = next;
        }
        block.last = prev;
    }
}

impl Module<VerifiedBody<'_>> {
    /// Rebind calls to optimized callees when the caller explicitly keeps them out of line.
    pub(crate) fn simplify_bodies(&self, ids: &FunctionIds<Function>) -> Result<Function, IrError> {
        let mut bodies = DenseMap::with_capacity(self.functions.len());
        for (id, source) in self.functions.iter() {
            let mut body = source.function.clone();
            body.simplify()?;
            for site in body.calls.values_mut() {
                if let Some(&child) = ids.get(&Arc::as_ptr(&site.callee))
                    && let Some(callee) = bodies.get(child)
                {
                    site.callee = Arc::clone(callee);
                }
            }
            if id == self.call_graph.root() {
                return Ok(body);
            }
            bodies.insert(Arc::new(body));
        }
        unreachable!("module has a root")
    }
}
