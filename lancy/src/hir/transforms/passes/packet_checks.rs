//! Prove the last-lane check on dominated full-packet edges.
use crate::hir::{
    Binary, Function, Op, Simd, Terminator,
    transforms::{ControlFlowGraph, DomTree},
};
use crate::support::slotmap::DenseMap;
use std::collections::HashMap;

pub(super) fn remove_full_packet_checks(f: &mut Function, cfg: &ControlFlowGraph, dom: &DomTree) {
    let mut definitions = DenseMap::filled(f.value_count(), None);
    let mut has_width = false;
    for &b in cfg.reverse_postorder() {
        for i in f.instructions(b) {
            if let Some(v) = f.result(i) {
                let op = *f.op(i);
                has_width |= matches!(op, Op::Simd(Simd::LaneCount(_)));
                definitions[v] = Some(op);
            }
        }
    }
    if !has_width {
        return;
    }
    let mut full_packets: HashMap<_, Vec<_>> = HashMap::new();
    let remaining_key = |value| match definitions[value] {
        Some(Op::Bin(Binary::Sub, len, offset)) => Ok((len, offset)),
        _ => Err(value),
    };
    for &b in dom.preorder() {
        if let Terminator::Branch { cond, yes, no } = *f.terminator(b)
            && let Some(Op::Bin(kind, a, c)) = definitions[cond]
        {
            let proof = match kind {
                Binary::Ge => Some((yes, a, c)),
                Binary::Le => Some((yes, c, a)),
                Binary::Lt => Some((no, a, c)),
                Binary::Gt => Some((no, c, a)),
                _ => None,
            };
            if let Some((edge, remaining, count)) = proof
                && let Some(Op::Simd(Simd::LaneCount(lane))) = definitions[count]
            {
                let target = f.edge(edge).0;
                if cfg.predecessors(target) == [b] {
                    full_packets
                        .entry((remaining_key(remaining), lane.bytes()))
                        .or_default()
                        .push(target);
                }
            }
        }
        f.rewrite_instructions(b, |op, _, _| {
            if let Op::BoundsCheck(i, n) = op {
                // A positive lane count makes the signed guard sufficient
                // for the unsigned check of the last lane.
                if let Some(Op::Bin(Binary::Sub, count, one)) = definitions[i]
                    && definitions[one] == Some(Op::Const(1))
                    && let Some(Op::Simd(Simd::LaneCount(lane))) = definitions[count]
                    && full_packets
                        .get(&(remaining_key(n), lane.bytes()))
                        .is_some_and(|guards| guards.iter().any(|&guard| dom.dominates(guard, b)))
                {
                    return None;
                }
            }
            Some(op)
        });
    }
}
