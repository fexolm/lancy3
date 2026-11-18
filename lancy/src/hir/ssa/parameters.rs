use super::{END, Function, Op, Span, Value};

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
                    self.fallible_ops -= usize::from(self.fallible(self.insts[index].op));
                    let inst = &mut self.insts[index];
                    // A self-link is a tombstone. Retired operations carry no
                    // operands, fallibility, or references to a callee.
                    inst.op = Op::Const(0);
                    cursor = std::mem::replace(&mut inst.next, cursor);
                }
            }
        }
        Ok(count)
    }

    /// Coalesce block parameters whose incoming values agree, disregarding a
    /// parameter forwarding itself on a backedge. Propagate through other
    /// parameters and rewrite the graph once after reaching the fixed point.
    pub fn simplify_block_params(&mut self) -> Result<usize, String> {
        self.verify()?;
        if self.blocks().skip(1).all(|b| self.params(b).is_empty()) {
            return Ok(0);
        }
        let n = self.value_count();
        let mut aliases: Vec<_> = (0..n as u32).map(Value).collect();
        let mut sources = vec![Vec::new(); n];
        let mut users = vec![Vec::new(); n];
        let mut work = Vec::new();
        for block in self.blocks() {
            if block != self.entry() {
                work.extend_from_slice(self.params(block));
            }
            for edge in self.terminator(block).edges() {
                let (target, args) = self.edge(edge);
                for (&p, &arg) in self.params(target).iter().zip(args) {
                    sources[p.index()].push(arg);
                    if arg != p {
                        users[arg.index()].push(p);
                    }
                }
            }
        }
        let mut changed = 0;
        while let Some(p) = work.pop() {
            if aliases[p.index()] != p {
                continue;
            }
            let mut candidate = None;
            for &v in &sources[p.index()] {
                let root = representative(&mut aliases, v);
                if root != p {
                    if candidate.is_some_and(|prev| prev != root) {
                        candidate = None;
                        break;
                    }
                    candidate = Some(root);
                }
            }
            if let Some(root) = candidate {
                aliases[p.index()] = root;
                changed += 1;
                for user in std::mem::take(&mut users[p.index()]) {
                    if aliases[user.index()] == user {
                        work.push(user);
                        if !sources[root.index()].is_empty() {
                            users[root.index()].push(user);
                        }
                    }
                }
            }
        }
        if changed == 0 {
            return Ok(0);
        }
        // Edge records may be shared between terminators. Compact each
        // referenced operand span once, before shortening the parameter lists.
        let mut visited = vec![false; self.edges.len()];
        for i in 0..self.block_capacity() {
            let Some(b) = &self.blocks[i] else {
                continue;
            };
            for e in b.term.unwrap().edges() {
                if std::mem::replace(&mut visited[e.index()], true) {
                    continue;
                }
                let (dest, span) = self.edges[e.index()];
                let mut write = span.start as usize;
                for (offset, &p) in self.blocks[dest.index()]
                    .as_ref()
                    .unwrap()
                    .params
                    .iter()
                    .enumerate()
                {
                    if aliases[p.index()] == p {
                        self.operands[write] = self.operands[span.start as usize + offset];
                        write += 1;
                    }
                }
                self.edges[e.index()].1 = Span {
                    start: span.start,
                    len: write as u32 - span.start,
                };
            }
        }
        for block in self.blocks.iter_mut().flatten() {
            block.params.retain(|v| aliases[v.index()] == *v);
        }
        self.substitute_params(aliases);
        Ok(changed)
    }
}

fn representative(parents: &mut [Value], mut v: Value) -> Value {
    while parents[v.index()] != v {
        let next = parents[v.index()];
        parents[v.index()] = parents[next.index()];
        v = next;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{Binary, Terminator};

    #[test]
    fn pruning_retires_fallible_instructions_and_keeps_surviving_ids() {
        let mut graph = Function::new(1, vec![]);
        let entry = graph.entry();
        let arg = graph.params(entry)[0];
        graph.set_terminator(entry, Terminator::Return(arg));
        let dead = graph.create_block();
        let div = graph.append_inst(dead, Op::Bin(Binary::Div, arg, arg));
        assert!(graph.needs_status());
        assert_eq!(graph.prune_unreachable(), Ok(1));
        assert!(!graph.needs_status());
        assert!(graph.replace_op(div, Op::Const(7)).is_err());
        assert_eq!(graph.interpret(&[91]), Ok(91));
    }

    #[test]
    fn trivial_phi_folds_through_loop_and_shared_edges() {
        let mut graph = Function::new(1, vec![]);
        let entry = graph.entry();
        let arg = graph.params(entry)[0];
        let head = graph.create_block();
        let exit = graph.create_block();
        let invariant = graph.append_param(head);
        let flag = graph.append_param(head);
        let output = graph.append_param(exit);
        let enter = graph.add_edge(head, &[arg, arg]);
        graph.set_terminator(
            entry,
            Terminator::Branch {
                cond: arg,
                yes: enter,
                no: enter,
            },
        );
        let zero = graph.append(head, Op::Const(0)).unwrap();
        let again = graph.add_edge(head, &[invariant, zero]);
        let leave = graph.add_edge(exit, &[invariant]);
        graph.set_terminator(
            head,
            Terminator::Branch {
                cond: flag,
                yes: again,
                no: leave,
            },
        );
        graph.set_terminator(exit, Terminator::Return(output));
        assert_eq!(graph.simplify_block_params(), Ok(2));
        assert_eq!(graph.params(head), &[flag]);
        assert_eq!(graph.params(exit), &[]);
        assert_eq!(graph.terminator(exit), &Terminator::Return(arg));
        assert_eq!(graph.interpret(&[42]), Ok(42));
    }
}
