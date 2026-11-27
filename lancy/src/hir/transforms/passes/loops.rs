use super::bounds::{Bounds, Count, at_least};
use super::facts::Facts;
use super::pipeline::{Analyses, Preserved};
use crate::hir::transforms::{ControlFlowGraph, DomTree, NaturalLoops};
use crate::hir::{
    Binary, Block, Function, IrError, Lane, MAX_VECTOR_BYTES, Op, Simd, Terminator, Type, Unary,
    Value, ssa::Edge,
};
use crate::support::slotmap::DenseMap;
use splice_target::Arch;
use std::collections::{HashMap, HashSet};
mod fusion;
mod memory;
mod recurrence;
mod unswitch;

enum Rewrite {
    None,
    Reduced,
    Unswitched,
}

/// Loop transformations to a fixed point. Each round that changes a loop
/// refreshes the graph; the facts and bounds of the last round are kept for
/// the passes after this one.
pub(super) fn run(f: &mut Function, analyses: &mut Analyses<'_>) -> Result<Preserved, IrError> {
    let mut allow_unswitch = true;
    let arch = analyses.arch;
    loop {
        let (cfg, dom) = analyses.graph(f)?;
        if !cfg.has_cycle() {
            return Ok(Preserved::All);
        }
        let mut analysis = None;
        match simplify_loops(f, cfg, dom, allow_unswitch, arch, &mut analysis) {
            Rewrite::None => {
                if let Some((facts, bounds)) = analysis {
                    analyses.keep(facts, bounds);
                }
                return Ok(Preserved::All);
            }
            Rewrite::Reduced => {}
            // One round of specialization avoids cloning the same loop for
            // every combination of invariant predicates.
            Rewrite::Unswitched => allow_unswitch = false,
        }
        f.prune_unreachable()?;
        analyses.invalidate(Preserved::Nothing);
    }
}

fn simplify_loops(
    f: &mut Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    allow_unswitch: bool,
    arch: Arch,
    analysis: &mut Option<(Facts, Bounds)>,
) -> Rewrite {
    let loops = NaturalLoops::discover(cfg, dom);
    if loops
        .headers
        .iter()
        .all(|&header| loops.preheader[header] == Block::INVALID)
    {
        return Rewrite::None;
    }
    let mut facts = Facts::new(f, cfg, dom);
    hoist(f, dom, &loops, &mut facts, arch);
    let mut counted = DenseMap::filled(f.block_capacity(), None);
    for &header in &loops.headers {
        counted[header] = counted_loop(f, cfg, dom, &loops, &facts, header);
    }
    let mut loop_bounds: Vec<_> = counted
        .values()
        .filter_map(|lp| lp.map(|lp| lp.bound))
        .collect();
    loop_bounds.sort_unstable();
    loop_bounds.dedup();
    let bounds = Bounds::new(f, cfg, dom, &facts, &loop_bounds);
    // Each reduction removes a branch or an entire loop. Traversal and
    // dominance are refreshed before another pass uses the changed graph.
    if bounds.optimize(f, cfg, dom, &loops, &mut facts, &counted)
        || delete_finite_dead_loops(f, cfg, dom, &loops, &facts, &counted)
        || recurrence::collapse(f, cfg, dom, &loops, &facts, &counted)
        || fusion::fuse(f, cfg, dom, &loops, &facts, &counted, &bounds)
    {
        Rewrite::Reduced
    } else if allow_unswitch && unswitch::run(f, cfg, dom, &loops, &facts) {
        Rewrite::Unswitched
    } else {
        *analysis = Some((facts, bounds));
        Rewrite::None
    }
}

fn hoist(f: &mut Function, dom: &DomTree, loops: &NaturalLoops, facts: &mut Facts, arch: Arch) {
    // Keep the outermost preheader as a fast path for fully invariant values.
    let mut outer = DenseMap::filled(f.block_capacity(), Block::INVALID);
    for &header in loops.headers.iter().rev() {
        let parent = loops.parent[header];
        outer[header] = outer
            .get(parent)
            .copied()
            .filter(|&b| b != Block::INVALID)
            .unwrap_or(loops.preheader[header]);
    }

    // Hoisting many vector-valued expressions across a backedge can trade
    // cheap recomputation for spills (see `hoist_vector_values`). Count
    // definitions as an upper bound on loop-local vector register demand.
    let budget = super::limits::hoist_vector_values(arch);
    let mut vector_values = DenseMap::filled(f.block_capacity(), 0);
    for &block in dom.preorder() {
        let Some(pressure) = vector_values.get_mut(loops.owner[block]) else {
            continue;
        };
        if *pressure > budget {
            continue;
        }
        if block != loops.owner[block] && matches!(f.terminator(block), Terminator::Branch { .. }) {
            *pressure = budget + 1;
            continue;
        }
        *pressure += f
            .params(block)
            .iter()
            .copied()
            .chain(
                f.instructions(block)
                    .filter(|&inst| !is_literal(f, facts, *f.op(inst)))
                    .filter_map(|inst| f.result(inst)),
            )
            .filter(|&v| match f.value_type(v) {
                Type::Vector(_) => true,
                ty => ty
                    .lane()
                    .is_some_and(|lane| lane.float() || lane.bytes() == 16),
            })
            .count();
    }

    // A literal or splat that was already moved to the same preheader is
    // reused, so each constant occupies one register across the loop.
    let mut constants = HashMap::new();
    // An `i32` constant moves only to be splatted before the loop: other
    // uses take it as an immediate or keep a general register short-lived.
    let mut splatted = HashSet::new();
    for &block in dom.preorder() {
        if loops.owner[block] != Block::INVALID {
            for inst in f.instructions(block) {
                if let Op::Simd(Simd::Splat(scalar)) = *f.op(inst)
                    && let Some(Op::Const(_)) = facts.op(f, scalar)
                {
                    splatted.insert(facts.root(scalar));
                }
            }
        }
    }
    let mut aliases = DenseMap::from_fn(f.value_count(), |v| v);
    let mut merged = false;

    // Results that can be evaluated early without observing a trap.
    let mut total = Vec::new();
    for &block in dom.preorder() {
        let header = loops.owner[block];
        if header == Block::INVALID {
            continue;
        }
        let nearest = loops.preheader[header];
        let outermost = outer[header];
        if nearest == Block::INVALID && outermost == Block::INVALID {
            continue;
        }
        let extended = nearest != Block::INVALID && vector_values[header] <= budget;
        // Without `extended`, only `i32` arithmetic other than division moves,
        // which has no effects.
        total.clear();
        if extended {
            total.extend(
                f.instructions(block)
                    .filter(|&inst| facts.is_total(f, inst))
                    .filter_map(|inst| f.result(inst)),
            );
            total.sort_unstable();
        }
        f.hoist_instructions(block, |op, result, types| {
            let scalar_i32 = matches!(op, Op::Bin(kind, a, _) if types[a] == Type::I32
                && !matches!(kind, Binary::Div | Binary::Rem));
            if !scalar_i32 && !extended {
                return None;
            }
            let cheap = match op {
                Op::Literal(..) => true,
                Op::Const(_) => splatted.contains(&result),
                Op::Bin(kind, ..) => !matches!(kind, Binary::Pow | Binary::Atan2 | Binary::Hypot),
                Op::Unary(kind, ..) => matches!(
                    kind,
                    Unary::Neg
                        | Unary::Abs
                        | Unary::Not
                        | Unary::Popcnt
                        | Unary::LeadingZeros
                        | Unary::TrailingZeros
                ),
                Op::Cast(..)
                | Op::Bitcast(..)
                | Op::Ternary(..)
                | Op::Simd(Simd::LaneCount(_) | Simd::Iota(_) | Simd::Splat(_)) => true,
                _ => false,
            };
            if !cheap || (extended && total.binary_search(&result).is_err()) {
                return None;
            }
            let available = |target| {
                let mut available = true;
                let mut operands = op;
                operands.map_operands(|value| {
                    available &= facts.available(value, target, dom).is_some();
                    value
                });
                available
            };
            let target = if scalar_i32 && outermost != Block::INVALID && available(outermost) {
                outermost
            } else {
                let mut target = Block::INVALID;
                let mut loop_header = header;
                while loop_header != Block::INVALID {
                    let preheader = loops.preheader[loop_header];
                    if preheader != Block::INVALID {
                        // An enclosing preheader dominates this one. A
                        // missing operand cannot become available further out.
                        if !available(preheader) {
                            break;
                        }
                        target = preheader;
                    }
                    if !scalar_i32 {
                        break;
                    }
                    loop_header = loops.parent[loop_header];
                }
                (target != Block::INVALID).then_some(target)?
            };
            let mut op = op;
            op.map_operands(|v| facts.root(v));
            facts.defs[result].block = target;
            let constant = match op {
                Op::Literal(lane, bits) => Some(Constant::Literal(lane, bits)),
                Op::Const(n) => Some(Constant::Literal(Lane::I32, u128::from(n as u32))),
                Op::Simd(Simd::Splat(scalar)) => Some(Constant::Splat(scalar)),
                _ => None,
            };
            if let Some(constant) = constant {
                let earlier = *constants.entry((target, constant)).or_insert(result);
                if earlier != result {
                    aliases[result] = earlier;
                    facts.merge_value(result, earlier);
                    merged = true;
                }
            }
            Some((target, op))
        });
    }
    if merged {
        facts.merge_values(f, &aliases);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Constant {
    Literal(Lane, u128),
    Splat(Value),
}

/// A literal, or the splat of one or of an `i32` constant: hoisting it costs no more registers than
/// reloading it costs loads (see `hoist_vector_values`).
fn is_literal(f: &Function, facts: &Facts, op: Op) -> bool {
    match op {
        Op::Literal(..) => true,
        Op::Simd(Simd::Splat(scalar)) => {
            matches!(facts.op(f, scalar), Some(Op::Literal(..) | Op::Const(_)))
        }
        _ => false,
    }
}

#[derive(Clone, Copy)]
struct CountedLoop {
    header: Block,
    preheader: Block,
    body: Edge,
    exit: Edge,
    iv: Value,
    start: Value,
    bound: Value,
    step: Count,
    guard: Value,
    packet: Option<(Count, Value)>,
}

impl CountedLoop {
    fn packet_size(self) -> Option<Count> {
        self.packet.map(|(size, _)| size)
    }

    fn strict_unit_step(self) -> bool {
        self.packet.is_none() && self.step == Count::Fixed(1)
    }
}

/// Analyze SSA forwarding without extending live ranges in the actual IR.
fn counted_loop(
    f: &Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    loops: &NaturalLoops,
    facts: &Facts,
    header: Block,
) -> Option<CountedLoop> {
    let preheader = loops.preheader[header];
    if preheader == Block::INVALID {
        return None;
    }
    let Terminator::Branch {
        cond,
        yes: body,
        no: exit,
    } = *f.terminator(header)
    else {
        return None;
    };
    let Op::Bin(kind, a, b) = facts.op(f, cond)? else {
        return None;
    };
    if f.value_type(a) != Type::I32 {
        return None;
    }
    let strict = match kind {
        Binary::Lt => Some((a, b, true)),
        Binary::Gt => Some((b, a, true)),
        Binary::Ge => Some((a, b, false)),
        Binary::Le => Some((b, a, false)),
        _ => None,
    }
    .filter(|&(i, _, _)| f.param_index(header, facts.root(i)).is_some());
    let (iv, bound, body_on_true, packet) = if let Some((iv, bound, yes)) = strict {
        (iv, bound, yes, None)
    } else {
        let (remaining, amount, yes) = at_least(kind, a, b)?;
        let Op::Bin(Binary::Sub, bound, iv) = facts.op(f, remaining)? else {
            return None;
        };
        (
            iv,
            bound,
            yes,
            Some((Count::of(f, facts, amount)?, facts.root(remaining))),
        )
    };
    let (body, exit) = if body_on_true {
        (body, exit)
    } else {
        (exit, body)
    };
    let iv = facts.root(iv);
    let bound = facts.available(bound, preheader, dom)?;
    let index = f.param_index(header, iv)?;
    if f.value_type(iv) != Type::I32
        || !loops.contains(header, f.edge(body).0)
        || loops.contains(header, f.edge(exit).0)
    {
        return None;
    }
    let Terminator::Jump(enter) = *f.terminator(preheader) else {
        return None;
    };
    let start = facts.root(f.edge(enter).1[index]);
    let mut step = None;
    for (latch, edge) in cfg.incoming_edges(header) {
        if latch == preheader {
            continue;
        }
        let next = f.edge(edge).1[index];
        let Op::Bin(Binary::Add, a, b) = facts.op(f, next)? else {
            return None;
        };
        let (a, b) = (facts.root(a), facts.root(b));
        let amount = if a == iv {
            b
        } else if b == iv {
            a
        } else {
            return None;
        };
        let count = Count::of(f, facts, amount)?;
        if *step.get_or_insert(count) != count {
            return None;
        }
    }
    let step = step?;
    if let Some((packet, _)) = packet {
        // Starting at zero, a negative bound fails the first packet test.
        // Otherwise both operands of the subtraction must be nonnegative.
        let initial = facts.constant(f, start)?;
        let nonnegative_bound =
            f.is_slice_length(bound) || facts.constant(f, bound).is_some_and(|n| n >= 0);
        if initial < 0 || (initial != 0 && !nonnegative_bound) || !step.at_most(packet) {
            return None;
        }
    } else if step != Count::Fixed(1)
        && facts
            .constant(f, bound)
            .is_none_or(|n| n > i32::MAX - (step.at_width(MAX_VECTOR_BYTES) - 1))
    {
        return None;
    }
    Some(CountedLoop {
        header,
        preheader,
        body,
        exit,
        iv,
        start,
        bound,
        step,
        guard: facts.root(cond),
        packet,
    })
}

fn delete_finite_dead_loops(
    f: &mut Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    loops: &NaturalLoops,
    facts: &Facts,
    counted: &DenseMap<Block, Option<CountedLoop>>,
) -> bool {
    let mut exits = DenseMap::filled(f.block_capacity(), None);
    for &header in &loops.headers {
        if let Some(lp) = counted[header]
            && f.edge(lp.exit)
                .1
                .iter()
                .all(|&v| facts.available(v, lp.preheader, dom).is_some())
        {
            exits[header] = Some(lp.exit);
        }
    }
    for &parent in loops.parent.values() {
        if parent != Block::INVALID {
            exits[parent] = None;
        }
    }
    if exits.values().all(Option::is_none) {
        return false;
    }
    let order = cfg.reverse_postorder();
    let mut rank = DenseMap::filled(f.block_capacity(), 0);
    for (i, &block) in order.iter().enumerate() {
        rank[block] = i;
    }
    for &block in order {
        let owner = loops.owner[block];
        let Some(exit) = exits.get(owner).copied().flatten() else {
            continue;
        };
        let term = *f.terminator(block);
        // A second exit could skip iterations and another internal
        // cycle could prevent reaching the bounded induction test.
        let other_path = matches!(term, Terminator::Return(_))
            || term.edges().any(|e| {
                let target = f.edge(e).0;
                e != exit
                    && (loops.owner[target] != owner
                        || (target != owner && rank[target] <= rank[block]))
            });
        if other_path || f.instructions(block).any(|inst| !facts.is_total(f, inst)) {
            exits[owner] = None;
        }
    }

    let mut remaining = exits.values().filter(|exit| exit.is_some()).count();
    let mut forwarded = Vec::new();
    for &block in order {
        if remaining == 0 {
            return false;
        }
        let owner = loops.owner[block];
        if remaining == 1 && exits.get(owner).is_some_and(Option::is_some) {
            continue;
        }
        let term = *f.terminator(block);
        let mut escapes = |value: Value| {
            let origin = loops.owner[facts.defs[value].block];
            let root = facts.root(value);
            let root_origin = loops.owner[facts.defs[root].block];
            // Keep a different loop whose definition a bypass would need.
            if root != value
                && root_origin != Block::INVALID
                && root_origin != owner
                && exits[root_origin].take().is_some()
            {
                remaining -= 1;
            }
            if origin != Block::INVALID && origin != owner && exits[origin].is_some() {
                if let Some(root) = facts.available(value, loops.preheader[origin], dom) {
                    forwarded.push((origin, value, root));
                } else {
                    exits[origin] = None;
                    remaining -= 1;
                }
            }
        };
        for inst in f.instructions(block) {
            f.visit_operands(inst, &mut escapes);
        }
        if let Terminator::Branch { cond: v, .. } | Terminator::Return(v) = term {
            escapes(v);
        }
        for edge in term.edges() {
            for &v in f.edge(edge).1 {
                escapes(v);
            }
        }
    }

    let mut changed = false;
    for &header in &loops.headers {
        if let Some(exit) = exits[header] {
            let (target, args) = f.edge(exit);
            let args: Vec<_> = args.iter().map(|&v| facts.root(v)).collect();
            f.jump(loops.preheader[header], target, &args);
            changed = true;
        }
    }
    if changed && !forwarded.is_empty() {
        let mut aliases = DenseMap::from_fn(f.value_count(), |v| v);
        for (header, value, root) in forwarded {
            if exits[header].is_some() {
                aliases[value] = root;
            }
        }
        f.substitute_values(&aliases);
    }
    changed
}
