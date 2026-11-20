//! Fuse adjacent unit-stride loops with equal iteration spaces.
use super::{CountedLoop, Facts, NaturalLoops};
use crate::hir::{
    Binary, Block, Function, Op, Scale, Terminator, Type, Value,
    ssa::Edge,
    transforms::{ControlFlowGraph, DomTree},
};
use crate::support::slotmap::DenseMap;

struct Footprint {
    base: Value,
    writes: bool,
}

#[derive(Clone, Copy)]
enum AccessKind {
    Load(Value),
    Store(Value),
}

#[derive(Clone, Copy)]
struct Access {
    block: Block,
    position: usize,
    base: Value,
    kind: AccessKind,
}

/// A single-entry linear chain from the loop's true successor to its latch.
/// Fusion may reorder operations across iterations, so all intervening
/// arithmetic must be total and all memory accesses must be pointwise.
struct Body {
    first: Block,
    latch: Block,
    backedge: Edge,
    memory: Vec<Footprint>,
    accesses: Vec<Access>,
}

impl Body {
    fn describe(
        f: &Function,
        cfg: &ControlFlowGraph,
        dom: &DomTree,
        loops: &NaturalLoops,
        facts: &Facts,
        counted: CountedLoop,
        operand_available: impl Fn(Value) -> bool,
    ) -> Option<Self> {
        let (first, args) = f.edge(counted.body);
        if !args.iter().copied().all(&operand_available) {
            return None;
        }
        let mut previous = counted.header;
        let mut block = first;
        let mut memory: Vec<Footprint> = Vec::new();
        let mut accesses = Vec::new();
        while block != counted.header
            && loops.owner[block] == counted.header
            && cfg.predecessors(block) == [previous]
        {
            for (position, inst) in f.instructions(block).enumerate() {
                let mut available = true;
                f.visit_operands(inst, |value| available &= operand_available(value));
                if !available {
                    return None;
                }
                let op = *f.op(inst);
                let (address, kind) = match op {
                    Op::Load(Type::I32, address) => (address, AccessKind::Load(f.result(inst)?)),
                    Op::Store(address, value) if f.value_type(value) == Type::I32 => {
                        (address, AccessKind::Store(value))
                    }
                    Op::Bin(Binary::Div | Binary::Rem, a, b)
                        if f.value_type(a) == Type::I32
                            && matches!(facts.op(f, b), Some(Op::Const(n)) if n != 0 && n != -1) =>
                    {
                        continue;
                    }
                    _ if op.pure(f.result(inst).map(|value| f.value_type(value))) => continue,
                    _ => return None,
                };
                let Some(Op::PtrOffset(base, index, Scale::Four)) = facts.op(f, address) else {
                    return None;
                };
                if facts.root(index) != facts.root(counted.iv) {
                    return None;
                }
                let base = facts.available(base, counted.preheader, dom)?;
                accesses.push(Access {
                    block,
                    position,
                    base,
                    kind,
                });
                let writes = matches!(kind, AccessKind::Store(_));
                if let Some(entry) = memory.iter_mut().find(|entry| entry.base == base) {
                    entry.writes |= writes;
                } else {
                    // Keep the pairwise check bounded, including for unusual kernels
                    // with many distinct pointer sources.
                    if memory.len() >= 8 {
                        return None;
                    }
                    memory.push(Footprint { base, writes });
                }
            }

            let Terminator::Jump(edge) = *f.terminator(block) else {
                return None;
            };
            let (next, args) = f.edge(edge);
            if !args.iter().copied().all(&operand_available) {
                return None;
            }
            if next == counted.header {
                return Some(Self {
                    first,
                    latch: block,
                    backedge: edge,
                    memory,
                    accesses,
                });
            }
            previous = block;
            block = next;
        }
        None
    }
}

fn same_value(f: &Function, facts: &Facts, a: Value, b: Value) -> bool {
    facts.root(a) == facts.root(b)
        || matches!((facts.op(f, a), facts.op(f, b)), (Some(Op::Const(x)), Some(Op::Const(y))) if x == y)
}

fn guard(f: &Function, counted: CountedLoop) -> Option<Value> {
    let Terminator::Branch { cond, yes, .. } = *f.terminator(counted.header) else {
        return None;
    };
    (yes == counted.body && f.value_type(cond) == Type::I32).then_some(cond)
}

struct Pair {
    left: CountedLoop,
    right: CountedLoop,
    left_body: Body,
    right_body: Body,
    left_initial: Edge,
    aliases: Vec<(Value, Value)>,
    carries: Vec<(Value, Value, Value)>,
}

impl Pair {
    fn find(
        f: &Function,
        cfg: &ControlFlowGraph,
        dom: &DomTree,
        loops: &NaturalLoops,
        facts: &Facts,
        counted: &DenseMap<Block, Option<CountedLoop>>,
        left: CountedLoop,
    ) -> Option<Self> {
        let bridge = f.edge(left.exit).0;
        if cfg.predecessors(bridge) != [left.header] {
            return None;
        }
        let Terminator::Jump(right_initial) = *f.terminator(bridge) else {
            return None;
        };
        let right = counted[f.edge(right_initial).0]?;
        if right.preheader != bridge
            || loops.parent[left.header] != loops.parent[right.header]
            || !same_value(f, facts, left.start, right.start)
            || !same_value(f, facts, left.bound, right.bound)
        {
            return None;
        }
        let left_guard = guard(f, left)?;
        let right_guard = guard(f, right)?;
        // The surviving condition computes the same mask at every iteration,
        // including at the exit. Do not discard any other header computations.
        if facts.defs[right_guard].block != right.header
            || facts.defs[left_guard].block != left.header
            || !f
                .instructions(right.header)
                .all(|inst| f.result(inst) == Some(right_guard))
            || !f
                .instructions(bridge)
                .all(|inst| matches!(f.op(inst), Op::Const(_) | Op::Literal(..)))
            || !f
                .instructions(left.header)
                .all(|inst| f.op(inst).pure(f.result(inst).map(|v| f.value_type(v))))
        {
            return None;
        }
        let Terminator::Jump(left_initial) = *f.terminator(left.preheader) else {
            return None;
        };
        let left_body = Body::describe(f, cfg, dom, loops, facts, left, |_| true)?;
        let right_body = Body::describe(f, cfg, dom, loops, facts, right, |value| {
            let definition = facts.defs[value].block;
            // The second loop must not read the first loop's final live-out
            // as an invariant. The bridge and right header values are either
            // moved or explicitly remapped when constructing the fused loop.
            definition == bridge
                || loops.owner[definition] == right.header
                || dom.dominates(definition, left.preheader)
        })?;

        // Both loops use the same index and four-byte spacing. Equal bases
        // therefore conflict only in the same iteration, which stays ordered.
        // Different bases need a disjointness proof when either side writes.
        if left_body.memory.iter().any(|a| {
            right_body.memory.iter().any(|b| {
                (a.writes || b.writes)
                    && a.base != b.base
                    && !f.disjoint_buffer_params(a.base, b.base)
            })
        }) {
            return None;
        }

        let mut aliases = vec![(right.iv, left.iv), (right_guard, left_guard)];
        for &param in f.params(bridge) {
            aliases.push((param, facts.available(param, left.preheader, dom)?));
        }
        let mut carries = Vec::new();
        let (_, initial) = f.edge(right_initial);
        let (_, updates) = f.edge(right_body.backedge);
        for ((&param, &init), &update) in f.params(right.header).iter().zip(initial).zip(updates) {
            if param == right.iv {
                continue;
            }
            if let Some(value) = facts.available(param, left.preheader, dom) {
                aliases.push((param, value));
            } else {
                let source = facts.root(init);
                let initial = facts.available(init, left.preheader, dom).or_else(|| {
                    (facts.defs[source].block == bridge
                        && matches!(facts.op(f, source), Some(Op::Const(_) | Op::Literal(..))))
                    .then_some(source)
                })?;
                carries.push((param, initial, update));
            }
        }

        Some(Self {
            left,
            right,
            left_body,
            right_body,
            left_initial,
            aliases,
            carries,
        })
    }

    fn apply(self, f: &mut Function) {
        let mut initial = f.edge(self.left_initial).1.to_vec();
        let mut updates = f.edge(self.left_body.backedge).1.to_vec();
        let mut aliases = self.aliases;
        for (param, init, update) in self.carries {
            let replacement = f.append_typed_param(self.left.header, f.value_type(param));
            aliases.push((param, replacement));
            initial.push(init);
            updates.push(update);
        }

        let right_args = f.edge(self.right.body).1.to_vec();
        f.hoist_instructions(self.right.preheader, |op, _, _| {
            Some((self.left.preheader, op))
        });
        f.jump(self.left.preheader, self.left.header, &initial);
        f.jump(self.left_body.latch, self.right_body.first, &right_args);
        f.jump(self.right_body.latch, self.left.header, &updates);
        let Terminator::Branch { cond, yes, .. } = *f.terminator(self.left.header) else {
            unreachable!();
        };
        f.set_terminator(
            self.left.header,
            Terminator::Branch {
                cond,
                yes,
                no: self.right.exit,
            },
        );

        let mut map = DenseMap::from_fn(f.value_count(), |value| value);
        for (old, new) in aliases {
            map[old] = new;
        }
        forward_stores(
            f,
            &mut map,
            self.left_body
                .accesses
                .into_iter()
                .chain(self.right_body.accesses),
        );
        // The right entry/header are now unreachable and are removed by
        // the caller. Their old edge records do not participate in the CFG.
        f.substitute_values(&map);
    }
}

/// Within a fused iteration, the last exact store supplies subsequent loads.
/// A store overwritten before an observing load can be removed because the
/// fusion proof excluded all intervening operations that could trap.
fn forward_stores(
    f: &mut Function,
    aliases: &mut DenseMap<Value, Value>,
    accesses: impl Iterator<Item = Access>,
) {
    struct KnownStore {
        base: Value,
        value: Option<Value>,
        removable: Option<(Block, usize)>,
    }

    let mut stores: Vec<KnownStore> = Vec::new();
    let mut remove = Vec::new();
    for access in accesses {
        let exact = stores.iter().position(|store| store.base == access.base);
        match access.kind {
            AccessKind::Load(result) => {
                if let Some(value) = exact.and_then(|index| stores[index].value) {
                    aliases[result] = value;
                    remove.push((access.block, access.position));
                } else {
                    for store in &mut stores {
                        if !f.disjoint_buffer_params(store.base, access.base) {
                            store.removable = None;
                        }
                    }
                }
            }
            AccessKind::Store(value) => {
                for store in &mut stores {
                    if store.base != access.base
                        && !f.disjoint_buffer_params(store.base, access.base)
                    {
                        store.removable = None;
                        store.value = None;
                    }
                }
                let current = KnownStore {
                    base: access.base,
                    value: Some(value),
                    removable: Some((access.block, access.position)),
                };
                if let Some(index) = exact {
                    remove.extend(stores[index].removable);
                    stores[index] = current;
                } else {
                    stores.push(current);
                }
            }
        }
    }
    if remove.is_empty() {
        return;
    }
    // Stored values can themselves be loads forwarded earlier in the fused
    // body. Resolve the resulting SSA alias chains once.
    for value in aliases.keys() {
        let mut root = value;
        while aliases[root] != root {
            let parent = aliases[root];
            aliases[root] = aliases[parent];
            root = aliases[root];
        }
        aliases[value] = root;
    }
    remove.sort_unstable();
    for block_removals in remove.chunk_by(|a, b| a.0 == b.0) {
        let mut positions = block_removals
            .iter()
            .map(|&(_, position)| position)
            .peekable();
        let mut position = 0;
        f.rewrite_instructions(block_removals[0].0, |op, _, _| {
            let keep = positions.next_if_eq(&position).is_none();
            position += 1;
            keep.then_some(op)
        });
    }
}

pub(super) fn fuse(
    f: &mut Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    loops: &NaturalLoops,
    facts: &Facts,
    counted: &DenseMap<Block, Option<CountedLoop>>,
) -> bool {
    for &header in cfg.reverse_postorder() {
        if let Some(left) = counted[header]
            && let Some(pair) = Pair::find(f, cfg, dom, loops, facts, counted, left)
        {
            pair.apply(f);
            return true;
        }
    }
    false
}
