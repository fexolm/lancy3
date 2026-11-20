use super::{CountedLoop, Facts, NaturalLoops};
use crate::hir::transforms::{ControlFlowGraph, DomTree};
use crate::hir::{Binary, Block, Function, Op, Scale, Terminator, Type, Value};
use crate::support::slotmap::DenseMap;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Operand {
    Constant(i32),
    Ssa(Value),
}

/// Bound the work per query even on CFGs that repeat an equivalent guard.
fn record<K: Eq + std::hash::Hash>(table: &mut HashMap<K, Vec<Block>>, key: K, block: Block) {
    let witnesses = table.entry(key).or_default();
    if witnesses.len() < 4 {
        witnesses.push(block);
    }
}

fn unsigned_slice_length(f: &Function, facts: &Facts, value: Value) -> Option<Value> {
    let Op::Bin(Binary::Xor, a, b) = facts.op(f, value)? else {
        return None;
    };
    [(a, b), (b, a)].into_iter().find_map(|(len, bias)| {
        let len = facts.root(len);
        (facts.constant(f, bias) == Some(i32::MIN) && f.is_slice_length(len)).then_some(len)
    })
}

pub(super) fn optimize(
    f: &mut Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    loops: &NaturalLoops,
    facts: &mut Facts,
    counted: &DenseMap<Block, Option<CountedLoop>>,
) {
    let readonly = f.has_slice_params();
    if !readonly && !counted.values().any(Option::is_some) {
        return;
    }
    let operand = |f: &Function, value| {
        let v = facts.root(value);
        facts
            .constant(f, v)
            .map_or(Operand::Ssa(v), Operand::Constant)
    };
    let mut before = HashMap::new();
    let mut checked = HashMap::new();
    let mut minimum_lengths: HashMap<Value, Vec<(i32, Block)>> = HashMap::new();
    for &block in cfg.reverse_postorder() {
        if readonly {
            for inst in f.instructions(block) {
                if let Op::BoundsCheck(index, len) = *f.op(inst) {
                    record(&mut checked, (operand(f, index), facts.root(len)), block);
                }
            }
        }
        let Terminator::Branch { cond, yes, no } = *f.terminator(block) else {
            continue;
        };
        let yes = f.edge(yes).0;
        let no = f.edge(no).0;
        if yes == no {
            continue;
        }
        let Some(Op::Bin(kind, a, b)) = facts.op(f, cond) else {
            continue;
        };
        let a = facts.root(a);
        let b = facts.root(b);
        if f.value_type(a) != Type::I32 {
            continue;
        }
        // Each fact holds only after a specific edge. A single-predecessor
        // destination makes block dominance sufficient to use that fact.
        let relations = match kind {
            Binary::Lt | Binary::Le => [(a, b, yes), (b, a, no)],
            Binary::Gt | Binary::Ge => [(b, a, yes), (a, b, no)],
            _ => continue,
        };
        for (low, high, scope) in relations {
            if cfg.predecessors(scope) != [block] {
                continue;
            }
            record(&mut before, (low, high), scope);
            if readonly && let Some(min) = facts.constant(f, low) {
                // Splice lowers usize comparisons to signed comparisons
                // after flipping the sign bit. Only decode this guard for
                // declared slice lengths and thresholds in the i32 range.
                let (len, min) = if min < 0 {
                    let Some(len) = unsigned_slice_length(f, facts, high) else {
                        continue;
                    };
                    (len, min ^ i32::MIN)
                } else {
                    (high, min)
                };
                let entries = minimum_lengths.entry(len).or_default();
                if entries.len() < 4 {
                    entries.push((min, scope));
                }
            }
        }
    }
    let dominates = |witnesses: Option<&Vec<Block>>, at| {
        witnesses.is_some_and(|blocks| blocks.iter().any(|&b| dom.dominates(b, at)))
    };
    let covered = |index, len, at| {
        dominates(checked.get(&(index, len)), at)
            || matches!(index, Operand::Constant(i) if i >= 0 &&
                minimum_lengths.get(&len).is_some_and(|bounds| bounds.iter().any(
                    |&(min, block)| i < min && dom.dominates(block, at))))
    };

    // A check in the selected body is redundant when the signed induction
    // variable starts nonnegative and is below an upper bound <= length.
    for &block in cfg.reverse_postorder() {
        let Some(lp) = counted.get(loops.owner[block]).copied().flatten() else {
            continue;
        };
        let body = f.edge(lp.body).0;
        if facts.constant(f, lp.start).is_none_or(|n| n < 0)
            || body == lp.header
            || cfg.predecessors(body) != [lp.header]
            || !dom.dominates(body, block)
        {
            continue;
        }
        f.rewrite_instructions(block, |op, _, _| {
            if let Op::BoundsCheck(index, len) = op
                && facts.root(index) == lp.iv
            {
                let len = facts.root(len);
                if lp.bound == len || dominates(before.get(&(lp.bound, len)), lp.preheader) {
                    return None;
                }
            }
            Some(op)
        });
    }

    if !readonly {
        return;
    }
    let mut moves = Vec::new();
    let mut redundant = Vec::new();
    for &block in cfg.reverse_postorder() {
        let Some(&target) = loops.preheader.get(loops.owner[block]) else {
            continue;
        };
        if target == Block::INVALID {
            continue;
        }
        redundant.clear();
        for inst in f.instructions(block) {
            if let Op::BoundsCheck(index, len) = *f.op(inst)
                && covered(operand(f, index), facts.root(len), target)
            {
                redundant.push((index, len));
            }
            let Op::Load(Type::I32, ptr) = *f.op(inst) else {
                continue;
            };
            let Some(Op::PtrOffset(base, index, Scale::Four)) = facts.op(f, ptr) else {
                continue;
            };
            let Some(base) = facts.available(base, target, dom) else {
                continue;
            };
            let Some((len, 4, false)) = f.slice_param(base) else {
                continue;
            };
            let key = operand(f, index);
            if covered(key, len, target)
                && (matches!(key, Operand::Constant(_))
                    || facts.available(index, target, dom).is_some())
            {
                moves.push((target, inst, base, key));
            }
        }
        if !redundant.is_empty() {
            redundant.sort_unstable();
            f.rewrite_instructions(block, |op, _, _| match op {
                Op::BoundsCheck(index, len) if redundant.binary_search(&(index, len)).is_ok() => {
                    None
                }
                _ => Some(op),
            });
        }
    }
    let mut hoisted = HashMap::new();
    for (target, inst, base, key) in moves {
        let value = *hoisted.entry((target, base, key)).or_insert_with(|| {
            let index = match key {
                Operand::Constant(n) => facts.append(f, target, Op::Const(n)),
                Operand::Ssa(v) => v,
            };
            let addr = facts.append(f, target, Op::PtrOffset(base, index, Scale::Four));
            facts.append(f, target, Op::Load(Type::I32, addr))
        });
        f.replace_op_preserving_type(inst, Op::Copy(value));
    }
}
