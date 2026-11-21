//! Reuse available memory values and remove fully overwritten stores.
use super::facts::Facts;
use crate::hir::{
    Binary, Block, Function, Op, Scale, Simd, Type, Value,
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
                offset < self.ty.bytes(64) as u64
                    || offset.wrapping_neg() < other.ty.bytes(64) as u64
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
                if offset == 0 && self.ty.bytes(64) == previous.ty.bytes(64) {
                    return true;
                }
                offset
                    .checked_add(previous.ty.bytes(64) as u64)
                    .is_some_and(|end| end <= self.ty.bytes(16) as u64)
            })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Read {
    value: Value,
    passthrough: Option<Value>,
}

#[derive(Clone, Copy)]
struct Known {
    location: Location,
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
                    merged.retain(|entry| {
                        known.iter().any(|other| {
                            entry.location == other.location && entry.read == other.read
                        })
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
    let (pointer, ty, mask, kind) = match op {
        Op::Load(ty, ptr) => (ptr, ty, None, Access::Read(None)),
        Op::Simd(Simd::Load(lane, ptr)) => (ptr, Type::Vector(lane), None, Access::Read(None)),
        Op::Store(ptr, value) | Op::Simd(Simd::Store(ptr, value)) => {
            (ptr, f.value_type(value), None, Access::Write(root(value)))
        }
        Op::Simd(
            Simd::MaskedLoad {
                address,
                mask,
                passthrough: value,
            }
            | Simd::MaskedStore {
                address,
                mask,
                value,
            },
        ) => {
            let mask = Mask {
                indices: address.indices.map(&root),
                value: root(mask),
            };
            let kind = if matches!(op, Op::Simd(Simd::MaskedLoad { .. })) {
                Access::Read(Some(root(value)))
            } else {
                Access::Write(root(value))
            };
            (address.base, f.value_type(value), Some(mask), kind)
        }
        _ => return None,
    };
    Some((
        Location {
            address: address(pointer),
            ty,
            mask,
        },
        kind,
    ))
}

pub(super) fn optimize(f: &mut Function, cfg: &ControlFlowGraph, dom: &DomTree) {
    let accesses = f
        .blocks()
        .flat_map(|block| f.instructions(block))
        .filter(|&inst| {
            matches!(
                f.op(inst),
                Op::Load(..)
                    | Op::Store(..)
                    | Op::Simd(
                        Simd::Load(..)
                            | Simd::Store(..)
                            | Simd::MaskedLoad { .. }
                            | Simd::MaskedStore { .. }
                    )
            )
        })
        .take(2)
        .count();
    if accesses < 2 {
        return;
    }

    let facts = Facts::new(f, cfg, dom);
    let mut aliases = DenseMap::from_fn(f.value_count(), |value| value);
    let mut addresses = DenseMap::filled(f.value_count(), None);
    let mut exits = DenseMap::from_fn(f.block_capacity(), |_| Exit::Unseen);
    let mut removed = Vec::new();
    for &block in cfg.reverse_postorder() {
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
                    Op::Call(_) => known.clear(),
                    Op::Bin(Binary::Div | Binary::Rem, _, divisor)
                    | Op::Simd(Simd::Bin(Binary::Div | Binary::Rem, _, divisor))
                        if facts.nontrapping_divisor(f, root(divisor)) => {}
                    _ if !op.pure(result.map(|value| f.value_type(value))) => {
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
            let entry = match kind {
                Access::Read(passthrough) => {
                    let value = result.unwrap();
                    if let Some(read) = known.iter().find_map(|entry| {
                        entry.read.filter(|read| {
                            entry.location == location
                                && read.passthrough == passthrough
                                && dom.dominates(facts.defs[read.value].block, block)
                        })
                    }) {
                        aliases[value] = read.value;
                        removed.push(site);
                        continue;
                    }
                    for entry in &mut known {
                        if location.may_overlap(entry.location) {
                            entry.removable_store = None;
                        }
                    }
                    Known {
                        location,
                        read: Some(Read { value, passthrough }),
                        removable_store: None,
                    }
                }
                Access::Write(value) => {
                    known.retain(|entry| {
                        if location.overwrites(entry.location) {
                            removed.extend(entry.removable_store);
                        }
                        !location.may_overlap(entry.location)
                    });
                    // A scatter may overwrite an earlier lane at the same index.
                    // Only full stores supply a ready-to-use value for a load.
                    let read = location.mask.is_none().then_some(Read {
                        value,
                        passthrough: None,
                    });
                    Known {
                        location,
                        read,
                        removable_store: Some(site),
                    }
                }
            };
            const MAX_KNOWN: usize = 16;
            if known.len() == MAX_KNOWN {
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
    f.substitute_values(&aliases);
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
