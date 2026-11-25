//! Bounds-check elimination and invariant loads in natural loops.
use super::super::bounds::{
    Bounds, BoundsPoint, Count, Index, at_least, index, last_lane, remaining,
};
use super::{CountedLoop, Facts, NaturalLoops};
use crate::hir::transforms::{ControlFlowGraph, DomTree};
use crate::hir::{Block, Function, Op, SliceParam, Terminator, Type, Value};
use crate::support::slotmap::DenseMap;
use std::collections::HashMap;

impl Bounds {
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
            && (lp.bound == len || self.before(dom, lp.bound, len, lp.preheader))
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
                    && self.covers(dom, i, len, count, BoundsPoint::End(preheader)))
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
                    _ => continue,
                };
                let Some(Op::PtrOffset(base, i, scale)) = facts.op(f, ptr) else {
                    continue;
                };
                let Some(base) = facts.available(base, preheader, dom) else {
                    continue;
                };
                let Some(SliceParam {
                    len,
                    element,
                    writable: false,
                    ..
                }) = f.slice_param(base)
                else {
                    continue;
                };
                let stride = 1 << scale as u8;
                let i = index(f, facts, i);
                if stride == usize::from(element)
                    && self.covers(
                        dom,
                        i,
                        len,
                        Count::access(ty, stride),
                        BoundsPoint::End(preheader),
                    )
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
                    facts.append(f, target, Op::Load(ty, addr))
                });
            f.replace_op_preserving_type(inst, Op::Copy(value));
        }
        changed_cfg
    }
}
