//! Fuse adjacent counted loops with equal iteration spaces.
use super::super::facts::flatten;
use super::{Bounds, Count, CountedLoop, Facts, NaturalLoops};
use crate::hir::{
    Block, Function, Op, Scale, Simd, Terminator, Type, Unary, Value, VectorAddress,
    ssa::Edge,
    transforms::{ControlFlowGraph, DomTree},
};
use crate::support::slotmap::DenseMap;

struct Footprint {
    base: Value,
    scale: Scale,
    writes: bool,
    in_slice: bool,
}

struct Analysis<'a> {
    f: &'a Function,
    cfg: &'a ControlFlowGraph,
    dom: &'a DomTree,
    loops: &'a NaturalLoops,
    facts: &'a Facts,
    bounds: &'a Bounds,
}

/// A single-entry linear chain from the loop's true successor to its latch.
/// Fusion may reorder operations across iterations, so all intervening
/// arithmetic must be total and all memory accesses must be pointwise.
struct Body {
    first: Block,
    latch: Block,
    backedge: Edge,
    memory: Vec<Footprint>,
}

impl Body {
    fn describe(
        analysis: &Analysis<'_>,
        counted: CountedLoop,
        operand_available: impl Fn(Value) -> bool,
    ) -> Option<Self> {
        let &Analysis {
            f,
            cfg,
            dom,
            loops,
            facts,
            bounds,
        } = analysis;
        let (first, args) = f.edge(counted.body);
        if !args.iter().copied().all(&operand_available) {
            return None;
        }
        let mut previous = counted.header;
        let mut block = first;
        let mut memory: Vec<Footprint> = Vec::new();
        while block != counted.header
            && loops.owner[block] == counted.header
            && cfg.predecessors(block) == [previous]
        {
            for inst in f.instructions(block) {
                let mut available = true;
                f.visit_operands(inst, |value| available &= operand_available(value));
                if !available {
                    return None;
                }
                let op = *f.op(inst);
                let (address, ty, writes) = match op {
                    Op::Load(ty, address) => (address, ty, false),
                    Op::Simd(Simd::Load(lane, address)) => (address, Type::Vector(lane), false),
                    Op::Store(address, value) | Op::Simd(Simd::Store(address, value)) => {
                        (address, f.value_type(value), true)
                    }
                    Op::Simd(Simd::MaskedLoad {
                        address:
                            VectorAddress {
                                base,
                                indices: None,
                            },
                        passthrough,
                        ..
                    }) => (base, f.value_type(passthrough), false),
                    Op::Simd(Simd::MaskedStore {
                        address:
                            VectorAddress {
                                base,
                                indices: None,
                            },
                        value,
                        ..
                    }) => (base, f.value_type(value), true),
                    _ if facts.is_total(f, inst) => continue,
                    _ => return None,
                };
                let Some(Op::PtrOffset(base, index, scale)) = facts.op(f, address) else {
                    return None;
                };
                if facts.root(index) != facts.root(counted.iv) {
                    return None;
                }
                let base = facts.available(base, counted.preheader, dom)?;
                let stride = 1 << scale as u8;
                let width = Count::access(ty, stride);
                // Accesses must fit in one iteration's byte range at every
                // supported SIMD width. Later iterations must never overwrite it.
                if !width.at_most(counted.step) {
                    return None;
                }
                let in_slice = f.slice_param(base).is_some_and(|(len, element, _)| {
                    stride <= usize::from(element)
                        && bounds.loop_covers(f, facts, dom, counted, len, width)
                });
                if let Some(entry) = memory.iter_mut().find(|entry| entry.base == base) {
                    if entry.scale != scale {
                        return None;
                    }
                    entry.writes |= writes;
                    entry.in_slice &= in_slice;
                } else {
                    // Keep the pairwise check bounded, including for unusual kernels
                    // with many distinct pointer sources.
                    if memory.len() >= 8 {
                        return None;
                    }
                    memory.push(Footprint {
                        base,
                        scale,
                        writes,
                        in_slice,
                    });
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

fn body_on_true(f: &Function, counted: CountedLoop) -> bool {
    matches!(f.terminator(counted.header), Terminator::Branch { yes, .. } if *yes == counted.body)
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
        analysis: &Analysis<'_>,
        counted: &DenseMap<Block, Option<CountedLoop>>,
        left: CountedLoop,
    ) -> Option<Self> {
        let &Analysis {
            f,
            cfg,
            dom,
            loops,
            facts,
            ..
        } = analysis;
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
            || left.step != right.step
            || left.packet_size() != right.packet_size()
        {
            return None;
        }
        // The surviving condition computes the same mask at every iteration,
        // including at the exit. A packet guard also computes the remaining count.
        if facts.defs[right.guard].block != right.header
            || facts.defs[left.guard].block != left.header
            || !f.instructions(right.header).all(|inst| {
                f.result(inst).is_some_and(|value| {
                    value == right.guard
                        || right
                            .packet
                            .is_some_and(|(_, remaining)| value == remaining)
                })
            })
            || !f.instructions(bridge).all(|inst| {
                matches!(
                    f.op(inst),
                    Op::Const(_) | Op::Literal(..) | Op::Simd(Simd::LaneCount(_))
                )
            })
            || !f
                .instructions(left.header)
                .all(|inst| facts.is_total(f, inst))
        {
            return None;
        }
        let Terminator::Jump(left_initial) = *f.terminator(left.preheader) else {
            return None;
        };
        let left_body = Body::describe(analysis, left, |_| true)?;
        let right_body = Body::describe(analysis, right, |value| {
            let definition = facts.defs[value].block;
            // The second loop must not read the first loop's final live-out
            // as an invariant. The bridge and right header values are either
            // moved or explicitly remapped when constructing the fused loop.
            definition == bridge
                || loops.owner[definition] == right.header
                || dom.dominates(definition, left.preheader)
        })?;

        // Accesses at the same base and scale can only intersect within an
        // iteration, whose order is preserved. The disjoint slice contract
        // covers distinct bases only within their declared ranges.
        if left_body.memory.iter().any(|a| {
            right_body.memory.iter().any(|b| {
                (a.writes || b.writes)
                    && if a.base == b.base {
                        a.scale != b.scale
                    } else {
                        !(a.in_slice && b.in_slice && f.disjoint_buffer_params(a.base, b.base))
                    }
            })
        }) {
            return None;
        }

        let mut aliases = vec![(right.iv, left.iv)];
        if let (Some((_, left_remaining)), Some((_, right_remaining))) = (left.packet, right.packet)
        {
            aliases.push((right_remaining, left_remaining));
        }
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
                        && matches!(
                            facts.op(f, source),
                            Some(Op::Const(_) | Op::Literal(..) | Op::Simd(Simd::LaneCount(_)))
                        ))
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

    fn apply(self, f: &mut Function, replacements: &mut Vec<(Value, Value)>) {
        let mut initial = f.edge(self.left_initial).1.to_vec();
        let mut updates = f.edge(self.left_body.backedge).1.to_vec();
        let mut aliases = self.aliases;
        let guard = if body_on_true(f, self.left) == body_on_true(f, self.right) {
            self.left.guard
        } else {
            // Both comparisons produce canonical i32 masks. The second
            // loop's body and live-outs still see its original polarity.
            f.append(self.left.header, Op::Unary(Unary::Not, self.left.guard))
                .unwrap()
        };
        aliases.push((self.right.guard, guard));
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
        let (yes, no) = if yes == self.left.body {
            (self.left.body, self.right.exit)
        } else {
            (self.right.exit, self.left.body)
        };
        f.set_terminator(self.left.header, Terminator::Branch { cond, yes, no });

        replacements.extend(aliases);
    }
}

pub(super) fn fuse(
    f: &mut Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    loops: &NaturalLoops,
    facts: &Facts,
    counted: &DenseMap<Block, Option<CountedLoop>>,
    bounds: &Bounds,
) -> bool {
    let analysis = Analysis {
        f,
        cfg,
        dom,
        loops,
        facts,
        bounds,
    };
    let mut pairs = Vec::new();
    let mut used = DenseMap::filled(f.block_capacity(), false);
    for &header in cfg.reverse_postorder() {
        if !used[header]
            && let Some(left) = counted[header]
            && let Some(pair) = Pair::find(&analysis, counted, left)
            && !used[pair.right.header]
        {
            used[header] = true;
            used[pair.right.header] = true;
            pairs.push(pair);
        }
    }
    if pairs.is_empty() {
        return false;
    }
    let mut replacements = Vec::new();
    for pair in pairs {
        pair.apply(f, &mut replacements);
    }
    let mut aliases = DenseMap::from_fn(f.value_count(), |value| value);
    for (old, new) in replacements {
        aliases[old] = new;
    }
    // A later pair can use the final value carried out of an earlier pair.
    flatten(&mut aliases);
    f.substitute_values(&aliases);
    true
}
