//! Propagate constants along reachable edges and rewrite scalar instructions.

use super::super::{Binary, Function, Instr, Label, Reg};
use super::{KnownArgs, const_eval::ConstEvaluator};
use std::collections::{HashMap, VecDeque};

type Facts = HashMap<Reg, i32>;
const MAX_MIXED_ARITY: usize = 32;

struct Blocks {
    start: Vec<usize>,
    by_label: Vec<usize>,
}

impl Blocks {
    fn new(f: &Function) -> Self {
        let mut start = vec![0];
        let mut by_label = vec![usize::MAX; f.targets.len()];
        let mut after_terminator = false;
        for (i, op) in f.code.iter().enumerate() {
            if i > 0 && (after_terminator || matches!(op, Instr::Label(_))) {
                start.push(i);
            }
            if let Instr::Label(l) = op {
                by_label[l.0 as usize] = start.len() - 1;
            }
            after_terminator =
                matches!(op, Instr::Return(_) | Instr::Branch { .. } | Instr::Jump(_));
        }
        start.push(f.code.len());
        Self { start, by_label }
    }

    fn range(&self, block: usize) -> std::ops::Range<usize> {
        self.start[block]..self.start[block + 1]
    }

    fn block(&self, label: Label) -> usize {
        self.by_label[label.0 as usize]
    }
}

// Dominating literal registers have the same value at every initialized
// use. Store these once instead of copying them into every block's entry.
fn fixed_literals(f: &Function) -> Vec<Option<i32>> {
    let mut assigned = vec![false; f.registers];
    assigned[..f.arity].fill(true);
    let mut fixed = vec![None; f.registers];
    for op in &f.code {
        if let Some(r) = op.written() {
            let index = r.0 as usize;
            fixed[index] = match op {
                Instr::Const(_, n) if !assigned[index] => Some(*n),
                _ => None,
            };
            assigned[index] = true;
        }
    }
    fixed
}

struct Transfer<'a> {
    fixed: Vec<Option<i32>>,
    evaluator: &'a mut ConstEvaluator,
}

impl Transfer<'_> {
    fn value(&self, facts: &Facts, r: Reg) -> Option<i32> {
        self.fixed[r.0 as usize].or_else(|| facts.get(&r).copied())
    }

    fn mixed_args(&self, args: &[Reg], has_buffers: bool, facts: &Facts) -> KnownArgs {
        let min_arity = if has_buffers { 1 } else { 2 };
        if args.len() < min_arity || args.len() > MAX_MIXED_ARITY {
            return Vec::new();
        }
        let known: KnownArgs = args
            .iter()
            .enumerate()
            .filter_map(|(i, r)| self.value(facts, *r).map(|n| (Reg(i as u32), n)))
            .collect();
        // An all-known call is handled by the bounded evaluator. If it runs
        // out of fuel, keep the generic call and its termination semantics.
        if !has_buffers && known.len() == args.len() {
            Vec::new()
        } else {
            known
        }
    }

    fn apply(&mut self, op: &Instr, facts: &mut Facts) -> Option<i32> {
        let dst = op.written()?;
        if let Some(n) = self.fixed[dst.0 as usize] {
            return Some(n);
        }
        let read = |r: Reg| self.value(facts, r);
        let value = match op {
            Instr::Const(_, n) => Some(*n),
            Instr::Copy(_, s) => read(*s),
            Instr::BufferLen { .. } | Instr::Load { .. } => None,
            Instr::Bin(_, op, a, b) => match (read(*a), read(*b)) {
                (Some(x), Some(y)) => op.checked_eval(x, y),
                (Some(0), _) | (_, Some(0)) if matches!(op, Binary::And | Binary::Mul) => Some(0),
                (Some(-1), _) | (_, Some(-1)) if matches!(op, Binary::Or) => Some(-1),
                _ if a == b
                    && matches!(
                        op,
                        Binary::Sub | Binary::Xor | Binary::Ne | Binary::Lt | Binary::Gt
                    ) =>
                {
                    Some(0)
                }
                _ if a == b && matches!(op, Binary::Eq | Binary::Ge | Binary::Le) => Some(-1),
                _ => None,
            },
            Instr::Call {
                function,
                args,
                buffers,
                ..
            } if buffers.is_empty() => {
                let actual = args.iter().map(|r| read(*r)).collect::<Option<Vec<_>>>();
                actual.and_then(|args| self.evaluator.evaluate(function, args))
            }
            Instr::Call { .. } => None,
            _ => unreachable!(),
        };
        if let Some(n) = value {
            facts.insert(dst, n);
        } else {
            facts.remove(&dst);
        }
        value
    }
}

fn entries(
    f: &Function,
    blocks: &Blocks,
    transfer: &mut Transfer<'_>,
    known: &[(Reg, i32)],
) -> Vec<Option<Facts>> {
    let count = blocks.start.len() - 1;
    let mut entry = vec![None; count];
    // Known only on the call edge. Assignments kill these facts and a
    // backedge to the first block can weaken them like any other join.
    entry[0] = Some(known.iter().copied().collect::<Facts>());
    let mut queue = VecDeque::from([0]);
    let mut queued = vec![false; count];
    queued[0] = true;
    while let Some(id) = queue.pop_front() {
        queued[id] = false;
        let mut facts = entry[id].as_ref().unwrap().clone();
        let code = &f.code[blocks.range(id)];
        for op in code {
            transfer.apply(op, &mut facts);
        }
        let mut propagate = |block: usize| {
            let changed = if let Some(existing) = &mut entry[block] {
                let before = existing.len();
                existing.retain(|r, n| facts.get(r) == Some(n));
                before != existing.len()
            } else {
                entry[block] = Some(facts.clone());
                true
            };
            if changed && !queued[block] {
                queued[block] = true;
                queue.push_back(block);
            }
        };
        match code.last().unwrap() {
            Instr::Return(_) => {}
            Instr::Jump(l) => propagate(blocks.block(*l)),
            Instr::Branch { cond, yes, no } => match transfer.value(&facts, *cond) {
                Some(n) => propagate(blocks.block(if n != 0 { *yes } else { *no })),
                None => {
                    propagate(blocks.block(*yes));
                    propagate(blocks.block(*no));
                }
            },
            _ => propagate(id + 1), // a verified reachable block cannot fall off the end
        }
    }
    entry
}

pub(super) fn fold_constants(
    f: &Function,
    evaluator: &mut ConstEvaluator,
    known: &[(Reg, i32)],
) -> (Vec<Instr>, Vec<KnownArgs>) {
    let blocks = Blocks::new(f);
    let mut transfer = Transfer {
        fixed: fixed_literals(f),
        evaluator,
    };
    let mut code = Vec::with_capacity(f.code.len());
    let mut call_facts = Vec::new();
    for (id, facts) in entries(f, &blocks, &mut transfer, known)
        .into_iter()
        .enumerate()
    {
        let Some(mut facts) = facts else { continue };
        for op in &f.code[blocks.range(id)] {
            // Snapshot before transfer. The destination can alias an argument.
            let actual = match op {
                Instr::Call { args, buffers, .. } => {
                    transfer.mixed_args(args, !buffers.is_empty(), &facts)
                }
                _ => Vec::new(),
            };
            let value = transfer.apply(op, &mut facts);
            if matches!(op, Instr::Call { .. }) && value.is_none() {
                call_facts.push(actual);
            }
            let replacement = match op {
                Instr::Branch { cond, yes, no } => {
                    let to = if yes == no {
                        Some(*yes)
                    } else {
                        transfer
                            .value(&facts, *cond)
                            .map(|n| if n != 0 { *yes } else { *no })
                    };
                    to.map(Instr::Jump)
                }
                Instr::Copy(dst, _) | Instr::Bin(dst, ..) | Instr::Call { dst, .. } => {
                    value.map(|n| Instr::Const(*dst, n))
                }
                _ => None,
            };
            code.push(replacement.unwrap_or_else(|| op.clone()));
        }
    }
    (code, call_facts)
}
