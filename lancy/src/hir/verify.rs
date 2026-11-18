use super::{Block, Function, Op, Terminator, Value};
use crate::hir::{BufferKind, check_forwarding};

/// Dominator tree preorder and subtree end. Same-block uses additionally
/// require an earlier definition, including uses on the outgoing edges.
struct Dominance {
    start: Vec<usize>,
    end: Vec<usize>,
}

impl Dominance {
    fn compute(f: &Function) -> Result<Self, String> {
        let count = f.block_capacity();
        let mut predecessors = vec![Vec::new(); count];
        let mut successors = vec![[None; 2]; count];
        for block in f.blocks() {
            let term = f
                .block(block)
                .term
                .ok_or_else(|| format!("unterminated {block:?}"))?;
            for (i, edge) in term.edges().enumerate() {
                let &(target, args) = f.edges.get(edge.index()).ok_or("unknown SSA edge")?;
                let target_block = f
                    .blocks
                    .get(target.index())
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("unknown target {target:?}"))?;
                if target == f.entry() {
                    return Err("branch into SSA ABI entry block".into());
                }
                if target_block.params.len() != args.len as usize {
                    return Err(format!("edge from {block:?} to {target:?} has wrong arity"));
                }
                predecessors[target.index()].push(block.index());
                successors[block.index()][i] = Some(target.index());
            }
        }

        // Reverse postorder keeps the common reducible case to a small number
        // of passes and avoids a quadratic per-block matrix of dominator sets.
        let entry = f.entry().index();
        let order = super::dominance::reverse_postorder(entry, &successors);
        if order.len() != f.blocks().count() {
            return Err("unreachable SSA block".into());
        }
        let parent = super::dominance::immediate(&order, &predecessors);
        let mut children = vec![Vec::new(); count];
        for &block in &order[1..] {
            children[parent[block]].push(block);
        }
        let mut start = vec![0; count];
        let mut end = vec![0; count];
        let mut serial = 0;
        let mut stack = vec![(entry, false)];
        while let Some((block, leave)) = stack.pop() {
            if leave {
                end[block] = serial;
            } else {
                start[block] = serial;
                serial += 1;
                stack.push((block, true));
                stack.extend(children[block].iter().map(|&child| (child, false)));
            }
        }
        Ok(Self { start, end })
    }

    fn dominates(&self, a: Block, b: Block) -> bool {
        self.start[a.index()] <= self.start[b.index()]
            && self.start[b.index()] < self.end[a.index()]
    }
}

impl Function {
    /// Validate this function and each distinct immutable callee once. Bodies
    /// can form a shared call DAG of arbitrary depth.
    #[cfg(test)]
    pub(super) fn verify_graph(&self) -> Result<(), String> {
        let mut pending = vec![self];
        let mut seen = std::collections::HashSet::new();
        while let Some(body) = pending.pop() {
            if !seen.insert(std::ptr::from_ref(body)) {
                continue;
            }
            body.verify()?;
            for inst in &body.insts {
                if let Op::Call(call) = inst.op {
                    pending.push(body.calls[call.index()].callee.as_ref());
                }
            }
        }
        Ok(())
    }

    /// Verify this body. Callee signatures participate in the check; code
    /// generation verifies every distinct callee body in the reachable graph.
    pub fn verify(&self) -> Result<(), String> {
        if self.params(self.entry()).len() != self.arity {
            return Err("SSA entry parameters disagree with function arity".into());
        }
        let dom = Dominance::compute(self)?;
        let mut defs = vec![None; self.value_count()];
        for block in self.blocks() {
            for &param in self.params(block) {
                defs[param.index()] = Some((block, 0));
            }
            for (i, inst) in self.instructions(block).enumerate() {
                if let Some(value) = self.result(inst) {
                    defs[value.index()] = Some((block, i + 1));
                }
            }
        }
        for block in self.blocks() {
            let used = |val: Value, index: usize| -> Result<(), String> {
                let &(def, position) = defs
                    .get(val.index())
                    .and_then(Option::as_ref)
                    .ok_or_else(|| format!("undefined {val:?} used in {block:?}"))?;
                if (block == def && position >= index) || !dom.dominates(def, block) {
                    return Err(format!("{val:?} does not dominate its use in {block:?}"));
                }
                Ok(())
            };
            for (i, inst) in self.instructions(block).enumerate() {
                let used = |v| used(v, i + 1);
                let buffer = |b: crate::hir::Buffer| {
                    self.buffers
                        .get(b.index())
                        .ok_or_else(|| format!("unknown buffer {b:?} in {inst:?}"))
                };
                match *self.op(inst) {
                    Op::Const(_) => {}
                    Op::Copy(v) => used(v)?,
                    Op::Bin(_, a, b) => {
                        used(a)?;
                        used(b)?;
                    }
                    Op::BufferLen(b) => {
                        buffer(b)?;
                    }
                    Op::Load(b, idx) => {
                        buffer(b)?;
                        used(idx)?;
                    }
                    Op::Store(b, idx, value) => {
                        if *buffer(b)? != BufferKind::Write {
                            return Err("store to read-only SSA buffer".into());
                        }
                        used(idx)?;
                        used(value)?;
                    }
                    Op::Call(id) => {
                        if id.index() >= self.calls.len() {
                            return Err("unknown SSA call".into());
                        }
                        let (callee, args, buffers) = self.call(id);
                        if args.len() != callee.arity() {
                            return Err("wrong scalar arity at SSA call".into());
                        }
                        check_forwarding(&self.buffers, callee.buffer_kinds(), buffers)?;
                        args.iter().copied().try_for_each(used)?;
                    }
                }
            }
            let term = *self.terminator(block);
            match term {
                Terminator::Branch { cond, .. } | Terminator::Return(cond) => {
                    used(cond, usize::MAX)?
                }
                Terminator::Jump(_) => {}
            }
            for edge in term.edges() {
                for &v in self.edge(edge).1 {
                    used(v, usize::MAX)?;
                }
            }
        }
        Ok(())
    }
}
