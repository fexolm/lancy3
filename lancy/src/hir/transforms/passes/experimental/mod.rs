//! Opt-in HIR rewrites for explicitly vectorized kernels.
use super::sccp::Fact;
use crate::hir::{
    Binary, Function, Op, Simd, Terminator, Ternary, Type, Value,
    transforms::{ControlFlowGraph, DomTree},
};
use crate::support::slotmap::DenseMap;
use std::collections::HashMap;

pub(super) fn simplify_simd(
    op: Op,
    result: Option<Value>,
    types: &DenseMap<Value, Type>,
    facts: &DenseMap<Value, Fact>,
    aliases: &DenseMap<Value, Value>,
    definitions: &mut HashMap<Value, Op>,
    vector_constants: &mut HashMap<Result<u8, Value>, Value>,
) -> Option<Op> {
    let definition = |v| definitions.get(&aliases[v]).copied();
    let first_lane = |v| facts.get(v).and_then(|fact| fact.bits()) == Some(0);
    let splat_index = |v| match definition(v) {
        Some(Op::Simd(Simd::Splat(i))) => Some(i),
        _ => None,
    };
    let mask = |v| {
        let Some(Op::Simd(Simd::Splat(scalar))) = definition(v) else {
            return None;
        };
        if let Some(bits) = facts.get(scalar).and_then(|fact| fact.bits()) {
            let lane = types[scalar].lane()?;
            return Some((bits >> (lane.bytes() * 8 - 1)) & 1 != 0);
        }
        let Some(Op::Literal(lane, bits)) = definition(scalar) else {
            return None;
        };
        Some((bits >> (lane.bytes() * 8 - 1)) & 1 != 0)
    };
    let op = match op {
        Op::Simd(Simd::Shuffle(x, _, indices))
            if matches!(definition(indices), Some(Op::Simd(Simd::Iota(_)))) =>
        {
            Op::Copy(x)
        }
        // An inner broadcast already has identical lanes, even if
        // its index was out of range and all those lanes are zero.
        Op::Simd(Simd::Shuffle(x, _, indices))
            if splat_index(indices).is_some_and(first_lane)
                && matches!(definition(x), Some(Op::Simd(Simd::Shuffle(_, _, inner)))
                    if splat_index(inner).is_some()) =>
        {
            Op::Copy(x)
        }
        Op::Simd(Simd::ExtractLane(vector, index)) if first_lane(index) => {
            match definition(vector) {
                Some(Op::Simd(Simd::Splat(scalar))) => Op::Copy(scalar),
                Some(Op::Simd(Simd::InsertLane(_, at, scalar))) if first_lane(at) => {
                    Op::Copy(scalar)
                }
                _ => op,
            }
        }
        Op::Simd(Simd::Ternary(Ternary::Select, m, yes, no)) => match mask(m) {
            Some(true) => Op::Copy(yes),
            Some(false) => Op::Copy(no),
            None if aliases[yes] == aliases[no] => Op::Copy(yes),
            None => op,
        },
        Op::Simd(Simd::MaskedLoad {
            address,
            mask: m,
            passthrough,
        }) => match mask(m) {
            Some(false) => Op::Copy(passthrough),
            Some(true) if address.indices.is_none() => {
                Op::Simd(Simd::Load(types[passthrough].lane().unwrap(), address.base))
            }
            _ => op,
        },
        Op::Simd(Simd::MaskedStore {
            address,
            mask: m,
            value,
        }) => match mask(m) {
            Some(false) => return None,
            Some(true) if address.indices.is_none() => Op::Simd(Simd::Store(address.base, value)),
            _ => op,
        },
        _ => op,
    };
    if let Some(dst) = result {
        let key = match op {
            Op::Simd(Simd::Iota(lane)) => Some(Ok(lane as u8)),
            Op::Simd(Simd::Splat(scalar)) => Some(Err(aliases[scalar])),
            _ => None,
        };
        if let Some(key) = key {
            if let Some(&existing) = vector_constants.get(&key) {
                return Some(Op::Copy(existing));
            }
            vector_constants.insert(key, dst);
        }
        if matches!(
            op,
            Op::Literal(..)
                | Op::Simd(
                    Simd::Splat(_) | Simd::Iota(_) | Simd::InsertLane(..) | Simd::Shuffle(..)
                )
        ) {
            definitions.insert(dst, op);
        }
    }
    Some(op)
}

pub(super) fn remove_full_packet_checks(f: &mut Function, cfg: &ControlFlowGraph, dom: &DomTree) {
    let mut definitions = DenseMap::filled(f.value_count(), None);
    for &b in cfg.reverse_postorder() {
        for i in f.instructions(b) {
            if let Some(v) = f.result(i) {
                definitions[v] = Some(*f.op(i));
            }
        }
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
                        .entry((remaining_key(remaining), lane as u8))
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
                        .get(&(remaining_key(n), lane as u8))
                        .is_some_and(|guards| guards.iter().any(|&guard| dom.dominates(guard, b)))
                {
                    return None;
                }
            }
            Some(op)
        });
    }
}

pub(super) fn zero_constants(f: &Function, cfg: &ControlFlowGraph) -> DenseMap<Value, bool> {
    let mut zero = DenseMap::filled(f.value_count(), false);
    for &b in cfg.reverse_postorder() {
        for i in f.instructions(b) {
            if let (Op::Const(0), Some(v)) = (*f.op(i), f.result(i)) {
                zero[v] = true;
            }
        }
    }
    zero
}

pub(super) fn accesses_first_lane(op: Op, zero: &DenseMap<Value, bool>) -> bool {
    // Every supported vector has a lane zero. Other lane accesses can trap.
    match op {
        Op::Simd(Simd::ExtractLane(_, i) | Simd::InsertLane(_, i, _)) => zero[i],
        _ => false,
    }
}
