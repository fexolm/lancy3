//! Reuse available memory values and remove fully overwritten stores.
use super::{
    bounds::{Bounds, BoundsPoint, Count, Index, LocalChecks},
    facts::Facts,
};
use crate::hir::{
    Block, Function, MAX_VECTOR_BYTES, MIN_VECTOR_BYTES, Op, Scale, Type, Value,
    transforms::{ControlFlowGraph, DomTree},
};
use crate::support::slotmap::DenseMap;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Address {
    base: Value,
    index: Option<(Value, Scale)>,
    displacement: u64,
}

impl Address {
    fn base(base: Value) -> Self {
        Self {
            base,
            index: None,
            displacement: 0,
        }
    }

    fn offset(self, base: Value, index: Value, scale: Scale, constant: Option<i32>) -> Self {
        if let Some(n) = constant {
            return Self {
                displacement: self
                    .displacement
                    .wrapping_add_signed(i64::from(n) << scale as u8),
                ..self
            };
        }
        let start = if self.index.is_none() {
            self
        } else {
            Self::base(base)
        };
        Self {
            index: Some((index, scale)),
            ..start
        }
    }

    fn distance_to(self, other: Self) -> Option<u64> {
        // Distinct bases do not prove disjointness for arbitrary derived pointers.
        (self.base == other.base && self.index == other.index)
            .then(|| other.displacement.wrapping_sub(self.displacement))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Mask {
    indices: Option<Value>,
    value: Value,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Location {
    address: Address,
    ty: Type,
    mask: Option<Mask>,
}

impl Location {
    fn indexed(self) -> bool {
        self.mask.is_some_and(|mask| mask.indices.is_some())
    }

    fn may_overlap(self, other: Self) -> bool {
        if self.indexed() || other.indexed() {
            return true;
        }
        // Use the largest supported vector width so the proof applies to every
        // target, even though the actual vector width is chosen after HIR.
        self.address
            .distance_to(other.address)
            .is_none_or(|offset| {
                offset < self.ty.bytes(MAX_VECTOR_BYTES) as u64
                    || offset.wrapping_neg() < other.ty.bytes(MAX_VECTOR_BYTES) as u64
            })
    }

    fn overwrites(self, previous: Self) -> bool {
        // The same mask and indices write the same bytes in the same lane order.
        if self == previous {
            return true;
        }
        if self.mask.is_some() || previous.indexed() {
            return false;
        }
        self.address
            .distance_to(previous.address)
            .is_some_and(|offset| {
                // Every vector occupies the same number of bytes for a given target.
                if offset == 0
                    && self.ty.bytes(MAX_VECTOR_BYTES) == previous.ty.bytes(MAX_VECTOR_BYTES)
                {
                    return true;
                }
                offset
                    .checked_add(previous.ty.bytes(MAX_VECTOR_BYTES) as u64)
                    .is_some_and(|end| end <= self.ty.bytes(MIN_VECTOR_BYTES) as u64)
            })
    }
}

#[derive(Clone, Copy)]
struct AccessRange {
    location: Location,
    in_slice: bool,
}

impl AccessRange {
    fn may_overlap(self, other: Self, f: &Function) -> bool {
        !(self.in_slice
            && other.in_slice
            && f.disjoint_buffer_params(self.location.address.base, other.location.address.base))
            && self.location.may_overlap(other.location)
    }
}

fn in_slice(
    f: &Function,
    location: Location,
    bounds: &Bounds,
    dom: &DomTree,
    at: BoundsPoint<'_>,
) -> Option<()> {
    if location.indexed() {
        return None;
    }
    let address = location.address;
    let slice = f.slice_param(address.base)?;
    let (len, element) = (slice.len, u64::from(slice.element));
    let byte_width = location.ty.bytes(MAX_VECTOR_BYTES) as u64;
    let (index, count) = if let Some((index, scale)) = address.index {
        let stride = 1u64 << scale as u8;
        if stride > element {
            return None;
        }
        let count = if address.displacement == 0 {
            Count::access(location.ty, stride as usize)
        } else {
            let end = address.displacement.checked_add(byte_width)?;
            Count::Fixed(i32::try_from(end.div_ceil(stride)).ok()?)
        };
        (Index::Ssa(index), count)
    } else {
        let (i, offset) = (
            address.displacement / element,
            address.displacement % element,
        );
        let index = Index::Constant(i32::try_from(i).ok()?);
        let count = if offset == 0 {
            Count::access(location.ty, element as usize)
        } else {
            Count::Fixed((offset + byte_width).div_ceil(element) as i32)
        };
        (index, count)
    };
    bounds.covers(dom, index, len, count, at).then_some(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Read {
    value: Value,
    passthrough: Option<Value>,
}

#[derive(Clone, Copy)]
struct Known {
    range: AccessRange,
    read: Option<Read>,
    removable_store: Option<(Block, usize)>,
}

enum Exit {
    Unseen,
    Traps,
    Known(Vec<Known>),
}

/// Carry a small window of memory facts through the forward CFG. A store is
/// removable only along a straight-line chain, so alternate exits can see it.
fn incoming(cfg: &ControlFlowGraph, exits: &mut DenseMap<Block, Exit>, block: Block) -> Vec<Known> {
    let preds = cfg.predecessors(block);
    if let [pred] = *preds
        && cfg.successors(pred).count() == 1
        && let Exit::Known(known) = &mut exits[pred]
    {
        return std::mem::take(known);
    }
    let mut merged: Option<Vec<Known>> = None;
    for &pred in preds {
        match &exits[pred] {
            // A backedge may change memory on the next iteration.
            Exit::Unseen => return Vec::new(),
            Exit::Traps => {}
            Exit::Known(known) => {
                if let Some(merged) = &mut merged {
                    merged.retain_mut(|entry| {
                        if let Some(other) = known.iter().find(|other| {
                            entry.range.location == other.range.location && entry.read == other.read
                        }) {
                            entry.range.in_slice &= other.range.in_slice;
                            true
                        } else {
                            false
                        }
                    });
                } else {
                    merged = Some(
                        known
                            .iter()
                            .filter(|entry| entry.read.is_some())
                            .map(|entry| Known {
                                removable_store: None,
                                ..*entry
                            })
                            .collect(),
                    );
                }
            }
        }
    }
    merged.unwrap_or_default()
}

#[derive(Clone, Copy)]
enum Access {
    Read(Option<Value>),
    Write(Value),
}

fn access(
    f: &Function,
    op: Op,
    root: impl Fn(Value) -> Value,
    address: impl Fn(Value) -> Address,
) -> Option<(Location, Access)> {
    let access = op.access(f)?;
    let value = access.value.map(&root);
    let kind = if access.write {
        Access::Write(value.expect("a store has a value"))
    } else {
        Access::Read(value)
    };
    let mask = access.mask.map(|mask| Mask {
        indices: access.indices.map(&root),
        value: root(mask),
    });
    Some((
        Location {
            address: address(access.address),
            ty: access.ty,
            mask,
        },
        kind,
    ))
}

pub(super) fn optimize(
    f: &mut Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    facts: &mut Facts,
    bounds: Option<&Bounds>,
) {
    let accesses = f
        .blocks()
        .flat_map(|block| f.instructions(block))
        .filter(|&inst| f.op(inst).access(f).is_some())
        .take(2)
        .count();
    if accesses < 2 {
        return;
    }

    let built_bounds;
    let bounds = if f.has_disjoint_slice_params() {
        built_bounds = bounds
            .is_none()
            .then(|| Bounds::new(f, cfg, dom, facts, &[]));
        bounds.or(built_bounds.as_ref())
    } else {
        None
    };
    let mut checks = LocalChecks::default();
    let mut aliases = DenseMap::from_fn(f.value_count(), |value| value);
    let mut addresses = DenseMap::filled(f.value_count(), None);
    let mut exits = DenseMap::from_fn(f.block_capacity(), |_| Exit::Unseen);
    let mut removed = Vec::new();
    let mut forwarded = false;
    for &block in cfg.reverse_postorder() {
        checks.clear();
        let mut known = incoming(cfg, &mut exits, block);
        let mut traps = false;
        for (position, inst) in f.instructions(block).enumerate() {
            let op = *f.op(inst);
            let root = |value| aliases[facts.root(value)];
            let address = |pointer| {
                let pointer = root(pointer);
                addresses[pointer].unwrap_or_else(|| Address::base(pointer))
            };
            let result = f.result(inst);
            let Some((location, kind)) = access(f, op, root, address) else {
                match op {
                    Op::PtrOffset(base, index, scale) => {
                        let normalized = address(base).offset(
                            root(base),
                            root(index),
                            scale,
                            facts.constant(f, root(index)),
                        );
                        addresses[result.unwrap()] = Some(normalized);
                    }
                    Op::BoundsCheck(i, n) if root(i) == root(n) => {
                        // Splice's short-vector store commits its prefix before
                        // this trap. It contributes no value to a normal join.
                        traps = true;
                        break;
                    }
                    Op::BoundsCheck(i, n) => {
                        if bounds.is_some() {
                            checks.record(f, facts, root(i), root(n));
                        }
                        for entry in &mut known {
                            entry.removable_store = None;
                        }
                    }
                    _ if op.effects(f).writes => known.clear(),
                    _ if !facts.is_total(f, inst) => {
                        // A check or arithmetic trap may expose completed stores.
                        for entry in &mut known {
                            entry.removable_store = None;
                        }
                    }
                    _ => {}
                }
                continue;
            };

            let site = (block, position);
            if let Access::Read(passthrough) = kind
                && let Some(read) = known.iter().find_map(|entry| {
                    entry.read.filter(|read| {
                        entry.range.location == location
                            && read.passthrough == passthrough
                            && dom.dominates(facts.defs[read.value].block, block)
                    })
                })
            {
                debug_assert_eq!(root(read.value), read.value);
                aliases[result.unwrap()] = read.value;
                removed.push(site);
                forwarded = true;
                continue;
            }
            let range = AccessRange {
                location,
                in_slice: bounds.is_some_and(|bounds| {
                    in_slice(
                        f,
                        location,
                        bounds,
                        dom,
                        BoundsPoint::InBlock(block, &checks),
                    )
                    .is_some()
                }),
            };
            let entry = match kind {
                Access::Read(passthrough) => {
                    let value = result.unwrap();
                    for entry in &mut known {
                        if entry.removable_store.is_some() && range.may_overlap(entry.range, f) {
                            entry.removable_store = None;
                        }
                    }
                    Known {
                        range,
                        read: Some(Read { value, passthrough }),
                        removable_store: None,
                    }
                }
                Access::Write(value) => {
                    known.retain(|entry| {
                        if location.overwrites(entry.range.location) {
                            removed.extend(entry.removable_store);
                        }
                        !range.may_overlap(entry.range, f)
                    });
                    // A scatter may overwrite an earlier lane at the same index.
                    // Only full stores supply a ready-to-use value for a load.
                    let read = location.mask.is_none().then_some(Read {
                        value,
                        passthrough: None,
                    });
                    Known {
                        range,
                        read,
                        removable_store: Some(site),
                    }
                }
            };
            if known.len() == super::limits::KNOWN_ACCESSES {
                known.remove(0);
            }
            known.push(entry);
        }
        exits[block] = if traps {
            Exit::Traps
        } else {
            Exit::Known(known)
        };
    }
    if removed.is_empty() {
        return;
    }
    if forwarded {
        facts.forward_loads(f, cfg, dom, &aliases);
    }
    removed.sort_unstable();
    for group in removed.chunk_by(|a, b| a.0 == b.0) {
        let mut positions = group.iter().map(|&(_, position)| position).peekable();
        let mut position = 0;
        f.rewrite_instructions(group[0].0, |op, _, _| {
            let keep = positions.next_if_eq(&position).is_none();
            position += 1;
            keep.then_some(op)
        });
    }
}
