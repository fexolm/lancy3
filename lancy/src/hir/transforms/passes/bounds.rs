//! Bounds established by guards and successfully executed checks.
mod witnesses;

use super::facts::Facts;
use crate::hir::{
    Binary, Block, Function, MAX_VECTOR_BYTES, MIN_VECTOR_BYTES, Op, Simd, Terminator, Type, Value,
    transforms::{ControlFlowGraph, DomTree},
};
use std::collections::{HashMap, HashSet};
use witnesses::{ScopedFacts, Witnesses};

/// Positive counts that can be compared without fixing the backend's SIMD width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Count {
    Fixed(i32),
    Lanes(usize),
}

impl Count {
    pub(super) fn of(f: &Function, facts: &Facts, value: Value) -> Option<Self> {
        match facts.op(f, value)? {
            Op::Const(n) if n > 0 => Some(Self::Fixed(n)),
            Op::Simd(Simd::LaneCount(lane)) => Some(Self::Lanes(lane.bytes())),
            _ => None,
        }
    }

    pub(super) fn access(ty: Type, scale: usize) -> Self {
        match ty {
            Type::Vector(_) => Self::Lanes(scale),
            _ => Self::Fixed(ty.bytes(0).div_ceil(scale) as i32),
        }
    }

    pub(super) fn at_width(self, bytes: usize) -> i32 {
        match self {
            Self::Fixed(n) => n,
            Self::Lanes(element) => bytes.div_ceil(element) as i32,
        }
    }

    pub(super) fn at_most(self, other: Self) -> bool {
        // Counts are constant or linear in the supported vector widths.
        [MIN_VECTOR_BYTES, MAX_VECTOR_BYTES]
            .into_iter()
            .all(|bytes| self.at_width(bytes) <= other.at_width(bytes))
    }
}

/// Normalize an edge that guarantees a signed lower bound.
pub(super) fn at_least(kind: Binary, a: Value, b: Value) -> Option<(Value, Value, bool)> {
    match kind {
        Binary::Ge => Some((a, b, true)),
        Binary::Le => Some((b, a, true)),
        Binary::Lt => Some((a, b, false)),
        Binary::Gt => Some((b, a, false)),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Index {
    Constant(i32),
    Ssa(Value),
}

pub(super) fn index(f: &Function, facts: &Facts, value: Value) -> Index {
    facts
        .constant(f, value)
        .map_or(Index::Ssa(facts.root(value)), Index::Constant)
}

pub(super) fn remaining(f: &Function, facts: &Facts, value: Value) -> (Value, Index) {
    match facts.op(f, value) {
        Some(Op::Bin(Binary::Sub, len, i)) => (facts.root(len), index(f, facts, i)),
        _ => (facts.root(value), Index::Constant(0)),
    }
}

pub(super) fn last_lane(f: &Function, facts: &Facts, value: Value) -> Option<Count> {
    match facts.op(f, value)? {
        Op::Bin(Binary::Sub, count, one) if facts.constant(f, one) == Some(1) => {
            Count::of(f, facts, count)
        }
        Op::Const(n) if n >= 0 => Some(Count::Fixed(n.checked_add(1)?)),
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct PacketCounts {
    fixed: Option<i32>,
    lane_bytes: Option<usize>,
}

impl From<Count> for PacketCounts {
    fn from(count: Count) -> Self {
        match count {
            Count::Fixed(n) => Self {
                fixed: Some(n),
                lane_bytes: None,
            },
            Count::Lanes(bytes) => Self {
                fixed: None,
                lane_bytes: Some(bytes),
            },
        }
    }
}

impl PacketCounts {
    fn combine(self, other: Self) -> Self {
        Self {
            fixed: self.fixed.max(other.fixed),
            lane_bytes: match (self.lane_bytes, other.lane_bytes) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
        }
    }

    fn covers(self, count: Count) -> bool {
        self.fixed.is_some_and(|n| count.at_most(Count::Fixed(n)))
            || self
                .lane_bytes
                .is_some_and(|bytes| count.at_most(Count::Lanes(bytes)))
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

pub(super) struct Bounds {
    before: ScopedFacts<(Value, Value), ()>,
    checked: ScopedFacts<(Index, Value), ()>,
    minimum_lengths: ScopedFacts<Value, i32>,
    packets: ScopedFacts<(Value, Index), PacketCounts>,
    checked_packets: ScopedFacts<(Value, Index), PacketCounts>,
}

#[derive(Default)]
pub(super) struct LocalChecks {
    elements: HashSet<(Index, Value)>,
    packets: HashMap<(Value, Index), PacketCounts>,
}

impl LocalChecks {
    pub(super) fn clear(&mut self) {
        self.elements.clear();
        self.packets.clear();
    }

    pub(super) fn record(&mut self, f: &Function, facts: &Facts, i: Value, len: Value) {
        if let Some((index, len)) = slice_element(f, facts, i, len) {
            self.elements.insert((index, len));
        }
        if let Some((key, count)) = slice_packet(f, facts, i, len) {
            self.packets
                .entry(key)
                .and_modify(|old| *old = old.combine(count))
                .or_insert(count);
        }
    }
}

fn slice_element(f: &Function, facts: &Facts, i: Value, len: Value) -> Option<(Index, Value)> {
    let len = facts.root(len);
    f.is_slice_length(len).then(|| (index(f, facts, i), len))
}

fn slice_packet(
    f: &Function,
    facts: &Facts,
    i: Value,
    len: Value,
) -> Option<((Value, Index), PacketCounts)> {
    let count = last_lane(f, facts, i)?;
    let key = remaining(f, facts, len);
    f.is_slice_length(key.0).then_some((key, count.into()))
}

pub(super) enum BoundsPoint<'a> {
    End(Block),
    InBlock(Block, &'a LocalChecks),
}

// The odd position denotes the end of a block, once all its checks succeeded.
// A guard is available on entry to its successor, at the even position.
fn position(dom: &DomTree, block: Block) -> usize {
    2 * dom.scope(block).start
}

impl Bounds {
    pub(super) fn new(
        f: &Function,
        cfg: &ControlFlowGraph,
        dom: &DomTree,
        facts: &Facts,
        loop_bounds: &[Value],
    ) -> Self {
        let mut before = Witnesses::default();
        let mut checked = Witnesses::default();
        let mut minimum_lengths = Witnesses::default();
        let mut packets = Witnesses::default();
        let mut checked_packets = Witnesses::default();
        let declared = f.has_slice_params();
        for &block in cfg.reverse_postorder() {
            if declared {
                for inst in f.instructions(block) {
                    if let Op::BoundsCheck(i, len) = *f.op(inst) {
                        let scope = dom.scope(block);
                        let after = 2 * scope.start + 1..2 * scope.end;
                        if let Some(key) = slice_element(f, facts, i, len) {
                            checked.record(key, after.clone(), ());
                        }
                        if let Some((key, count)) = slice_packet(f, facts, i, len) {
                            checked_packets.record(key, after, count);
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
            if let Binary::Eq | Binary::Ne = kind {
                // Equal operands bound each other from both sides.
                let equal = if kind == Binary::Eq { yes } else { no };
                if cfg.predecessors(equal) == [block] {
                    let scope = dom.scope(equal);
                    let scope = 2 * scope.start..2 * scope.end;
                    for (lower, upper) in [(a, b), (b, a)] {
                        if loop_bounds.binary_search(&lower).is_ok() {
                            before.record((lower, upper), scope.clone(), ());
                        }
                    }
                }
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
                let scope = dom.scope(scope);
                let scope = 2 * scope.start..2 * scope.end;
                if loop_bounds.binary_search(&lower).is_ok() {
                    before.record((lower, upper), scope.clone(), ());
                }
                if declared && let Some(minimum) = facts.constant(f, lower) {
                    let (length, minimum) = if minimum < 0 {
                        let Some(length) = unsigned_slice_length(f, facts, upper) else {
                            continue;
                        };
                        (length, minimum ^ i32::MIN)
                    } else {
                        (upper, minimum)
                    };
                    if f.is_slice_length(length) {
                        minimum_lengths.record(length, scope, minimum);
                    }
                }
            }
            if cfg.predecessors(enough) == [block]
                && let Some(count) = Count::of(f, facts, low)
            {
                // `high >= count` with a positive count holds for fixed and
                // lane counts alike; `Count::at_most` compares them at every
                // supported vector width.
                let scope = dom.scope(enough);
                packets.record(
                    remaining(f, facts, high),
                    2 * scope.start..2 * scope.end,
                    count.into(),
                );
            }
        }
        Self {
            before: before.index(|(), ()| ()),
            checked: checked.index(|(), ()| ()),
            minimum_lengths: minimum_lengths.index(i32::max),
            packets: packets.index(PacketCounts::combine),
            checked_packets: checked_packets.index(PacketCounts::combine),
        }
    }

    pub(super) fn before(&self, dom: &DomTree, lower: Value, upper: Value, block: Block) -> bool {
        self.before
            .at(&(lower, upper), position(dom, block))
            .is_some()
    }

    // Only declared slice lengths are used here. They are nonnegative, so
    // len - i cannot wrap if i is nonnegative. The unsigned last-lane check
    // also needs a separate proof that the starting index fits.
    pub(super) fn covers(
        &self,
        dom: &DomTree,
        i: Index,
        len: Value,
        count: Count,
        at: BoundsPoint<'_>,
    ) -> bool {
        let (at, local) = match at {
            BoundsPoint::End(block) => (position(dom, block) + 1, None),
            BoundsPoint::InBlock(block, local) => (position(dom, block), Some(local)),
        };
        let nonnegative = matches!(i, Index::Constant(n) if n >= 0);
        let mut first = false;
        if let Index::Constant(n) = i
            && n >= 0
            && let Some(min) = self.minimum_lengths.at(&len, at)
        {
            if i64::from(n) + i64::from(count.at_width(MAX_VECTOR_BYTES)) <= i64::from(min) {
                return true;
            }
            first = n < min;
        }
        let first = first
            || local.is_some_and(|local| local.elements.contains(&(i, len)))
            || self.checked.at(&(i, len), at).is_some();
        let applies = |facts: &ScopedFacts<_, _>| {
            facts
                .at(&(len, i), at)
                .is_some_and(|packet: PacketCounts| packet.covers(count))
        };
        (first
            && (count == Count::Fixed(1)
                || local
                    .and_then(|local| local.packets.get(&(len, i)))
                    .is_some_and(|packet| packet.covers(count))
                || applies(&self.checked_packets)))
            || ((first || nonnegative) && applies(&self.packets))
    }
}

pub(super) fn remove_full_packet_checks(
    f: &mut Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    facts: &Facts,
    bounds: &mut Option<Bounds>,
) {
    let mut redundant = Vec::new();
    for &block in cfg.reverse_postorder() {
        for inst in f.instructions(block) {
            if let Op::BoundsCheck(i, n) = *f.op(inst)
                && let Some(count) = last_lane(f, facts, i)
                && bounds
                    .get_or_insert_with(|| Bounds::new(f, cfg, dom, facts, &[]))
                    .packets
                    .at(&remaining(f, facts, n), position(dom, block))
                    .is_some_and(|packet| packet.covers(count))
            {
                // The same i32 expression is positive and at least count
                // on this path, irrespective of how it was computed.
                redundant.push((i, n));
            }
        }
        if !redundant.is_empty() {
            redundant.sort_unstable();
            f.rewrite_instructions(block, |op, _, _| match op {
                Op::BoundsCheck(i, n) if redundant.binary_search(&(i, n)).is_ok() => None,
                _ => Some(op),
            });
            redundant.clear();
        }
    }
}
