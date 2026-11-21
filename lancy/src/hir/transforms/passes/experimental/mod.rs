//! Opt-in HIR rewrites for explicitly vectorized kernels.
use super::sccp::Fact;
use crate::hir::{Op, Simd, Ternary, Type, Value};
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
