use super::{Count, CountedLoop, Facts, NaturalLoops, at_least};
use crate::hir::transforms::{ControlFlowGraph, DomTree};
use crate::hir::{Binary, Block, Function, Op, Simd, Terminator, Type, Value};
use crate::support::slotmap::DenseMap;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Index {
    Constant(i32),
    Ssa(Value),
}

fn index(f: &Function, facts: &Facts, value: Value) -> Index {
    facts
        .constant(f, value)
        .map_or(Index::Ssa(facts.root(value)), Index::Constant)
}

fn remaining(f: &Function, facts: &Facts, value: Value) -> (Value, Index) {
    match facts.op(f, value) {
        Some(Op::Bin(Binary::Sub, len, i)) => (facts.root(len), index(f, facts, i)),
        _ => (facts.root(value), Index::Constant(0)),
    }
}

fn last_lane(f: &Function, facts: &Facts, value: Value) -> Option<Count> {
    match facts.op(f, value)? {
        Op::Bin(Binary::Sub, count, one) if facts.constant(f, one) == Some(1) => {
            Count::of(f, facts, count)
        }
        Op::Const(n) if n >= 0 => Some(Count::Fixed(n.checked_add(1)?)),
        _ => None,
    }
}

/// Each query only needs a small number of dominating witnesses.
fn record<K: Eq + std::hash::Hash, V>(table: &mut HashMap<K, Vec<V>>, key: K, witness: V) {
    let witnesses = table.entry(key).or_default();
    if witnesses.len() < 4 {
        witnesses.push(witness);
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

#[derive(Default)]
pub(super) struct Bounds {
    before: HashMap<(Value, Value), Vec<Block>>,
    checked: HashMap<(Index, Value), Vec<Block>>,
    minimum_lengths: HashMap<Value, Vec<(i32, Block)>>,
    packets: HashMap<(Value, Index), Vec<(Count, Block)>>,
    checked_packets: HashMap<(Value, Index), Vec<(Count, Block)>>,
}

impl Bounds {
    pub(super) fn new(f: &Function, cfg: &ControlFlowGraph, facts: &Facts) -> Self {
        let mut bounds = Self::default();
        let declared = f.has_slice_params();
        for &block in cfg.reverse_postorder() {
            if declared {
                for inst in f.instructions(block) {
                    if let Op::BoundsCheck(i, len) = *f.op(inst) {
                        record(
                            &mut bounds.checked,
                            (index(f, facts, i), facts.root(len)),
                            block,
                        );
                        if let Some(count) = last_lane(f, facts, i) {
                            record(
                                &mut bounds.checked_packets,
                                remaining(f, facts, len),
                                (count, block),
                            );
                        }
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
            let Some((high, low, on_yes)) = at_least(kind, a, b) else {
                continue;
            };
            let (enough, short) = if on_yes { (yes, no) } else { (no, yes) };
            // An edge is a witness only when its destination has no other
            // predecessor. The other outcome also implies high <= low.
            for (lower, upper, scope) in [(low, high, enough), (high, low, short)] {
                if cfg.predecessors(scope) != [block] {
                    continue;
                }
                record(&mut bounds.before, (lower, upper), scope);
                if declared && let Some(minimum) = facts.constant(f, lower) {
                    let (length, minimum) = if minimum < 0 {
                        let Some(length) = unsigned_slice_length(f, facts, upper) else {
                            continue;
                        };
                        (length, minimum ^ i32::MIN)
                    } else {
                        (upper, minimum)
                    };
                    record(&mut bounds.minimum_lengths, length, (minimum, scope));
                }
            }
            if declared
                && cfg.predecessors(enough) == [block]
                && let Some(count) = Count::of(f, facts, low)
            {
                record(
                    &mut bounds.packets,
                    remaining(f, facts, high),
                    (count, enough),
                );
            }
        }
        bounds
    }

    pub(super) fn loop_covers(
        &self,
        f: &Function,
        facts: &Facts,
        dom: &DomTree,
        lp: CountedLoop,
        len: Value,
        count: Count,
    ) -> bool {
        facts.constant(f, lp.start).is_some_and(|n| n >= 0)
            && count.at_most(lp.packet_size().unwrap_or(Count::Fixed(1)))
            && (lp.bound == len
                || self.before.get(&(lp.bound, len)).is_some_and(|blocks| {
                    blocks
                        .iter()
                        .any(|&block| dom.dominates(block, lp.preheader))
                }))
    }

    // Only declared slice lengths are used here. They are nonnegative, so
    // len - i cannot wrap if i is nonnegative. The unsigned last-lane check
    // also needs a separate proof that the starting index fits.
    fn invariant_covers(
        &self,
        dom: &DomTree,
        i: Index,
        len: Value,
        count: Count,
        at: Block,
    ) -> bool {
        let dominates = |block| dom.dominates(block, at);
        if let Index::Constant(n) = i
            && n >= 0
            && self.minimum_lengths.get(&len).is_some_and(|bounds| {
                bounds.iter().any(|&(min, block)| {
                    i64::from(n) + i64::from(count.at_width(64)) <= i64::from(min)
                        && dominates(block)
                })
            })
        {
            return true;
        }
        let first = self
            .checked
            .get(&(i, len))
            .is_some_and(|blocks| blocks.iter().any(|&block| dominates(block)))
            || matches!(i, Index::Constant(n) if n >= 0 &&
            self.minimum_lengths.get(&len).is_some_and(|bounds| {
                bounds.iter().any(|&(min, block)| n < min && dominates(block))
            }));
        let applies = |witnesses: Option<&Vec<(Count, Block)>>| {
            witnesses.is_some_and(|entries| {
                entries
                    .iter()
                    .any(|&(width, block)| count.at_most(width) && dominates(block))
            })
        };
        (first && (count == Count::Fixed(1) || applies(self.checked_packets.get(&(len, i)))))
            || ((first || matches!(i, Index::Constant(n) if n >= 0))
                && applies(self.packets.get(&(len, i))))
    }

    pub(super) fn optimize(
        &self,
        f: &mut Function,
        cfg: &ControlFlowGraph,
        dom: &DomTree,
        loops: &NaturalLoops,
        facts: &mut Facts,
        counted: &DenseMap<Block, Option<CountedLoop>>,
    ) -> bool {
        let declared = f.has_slice_params();
        let mut changed_cfg = false;
        let mut moves = Vec::new();
        let mut redundant = Vec::new();
        for &block in cfg.reverse_postorder() {
            let owner = loops.owner[block];
            let Some(&preheader) = loops.preheader.get(owner) else {
                continue;
            };
            if preheader == Block::INVALID {
                continue;
            }
            let active = counted[owner].filter(|lp| {
                let body = f.edge(lp.body).0;
                body != lp.header
                    && cfg.predecessors(body) == [lp.header]
                    && dom.dominates(body, block)
            });
            let covered = |f: &Function, i, len, count| {
                active.is_some_and(|lp| {
                    i == Index::Ssa(lp.iv) && self.loop_covers(f, facts, dom, lp, len, count)
                }) || (declared
                    && f.is_slice_length(len)
                    && self.invariant_covers(dom, i, len, count, preheader))
            };
            redundant.clear();
            for inst in f.instructions(block) {
                let (ty, ptr) = match *f.op(inst) {
                    Op::BoundsCheck(i, len) => {
                        if covered(f, index(f, facts, i), facts.root(len), Count::Fixed(1))
                            || last_lane(f, facts, i).is_some_and(|count| {
                                let (len, i) = remaining(f, facts, len);
                                covered(f, i, len, count)
                            })
                        {
                            redundant.push((i, len));
                        }
                        continue;
                    }
                    Op::Load(ty, ptr) if declared => (ty, ptr),
                    Op::Simd(Simd::Load(lane, ptr)) if declared => (Type::Vector(lane), ptr),
                    _ => continue,
                };
                let Some(Op::PtrOffset(base, i, scale)) = facts.op(f, ptr) else {
                    continue;
                };
                let Some(base) = facts.available(base, preheader, dom) else {
                    continue;
                };
                let Some((len, element, false)) = f.slice_param(base) else {
                    continue;
                };
                let stride = 1 << scale as u8;
                let i = index(f, facts, i);
                if stride == usize::from(element)
                    && self.invariant_covers(dom, i, len, Count::access(ty, stride), preheader)
                    && (matches!(i, Index::Constant(_))
                        || matches!(i, Index::Ssa(v) if facts.available(v, preheader, dom).is_some()))
                {
                    moves.push((preheader, inst, base, i, scale, ty));
                }
            }
            if !redundant.is_empty() {
                redundant.sort_unstable();
                f.rewrite_instructions(block, |op, _, _| match op {
                    Op::BoundsCheck(i, n) if redundant.binary_search(&(i, n)).is_ok() => None,
                    _ => Some(op),
                });
            }
            // A full packet loop already guarantees the full branch of a
            // checked SIMD store. Its short-store path is unreachable.
            if let Some(lp) = active
                && let Terminator::Branch { cond, yes, no } = *f.terminator(block)
                && let Some(Op::Bin(kind, a, b)) = facts.op(f, cond)
                && f.value_type(a) == Type::I32
                && let Some((left, right, full_on_yes)) = at_least(kind, a, b)
                && let Some(count) = Count::of(f, facts, right)
                && let (len, Index::Ssa(i)) = remaining(f, facts, left)
                && i == lp.iv
                && self.loop_covers(f, facts, dom, lp, len, count)
            {
                f.set_terminator(block, Terminator::Jump(if full_on_yes { yes } else { no }));
                changed_cfg = true;
            }
        }

        let mut hoisted = HashMap::new();
        for (target, inst, base, i, scale, ty) in moves {
            let value = *hoisted
                .entry((target, base, i, scale, ty))
                .or_insert_with(|| {
                    let i = match i {
                        Index::Constant(n) => facts.append(f, target, Op::Const(n)),
                        Index::Ssa(v) => v,
                    };
                    let addr = facts.append(f, target, Op::PtrOffset(base, i, scale));
                    let op = match ty {
                        Type::Vector(lane) => Op::Simd(Simd::Load(lane, addr)),
                        _ => Op::Load(ty, addr),
                    };
                    facts.append(f, target, op)
                });
            f.replace_op_preserving_type(inst, Op::Copy(value));
        }
        changed_cfg
    }
}
