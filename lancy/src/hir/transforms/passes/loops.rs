use super::facts::Facts;
use crate::hir::transforms::{ControlFlowGraph, DomTree};
use crate::hir::{Binary, Block, Function, Op, Simd, Terminator, Type, Unary, Value, ssa::Edge};
use crate::support::slotmap::DenseMap;
mod fusion;
mod memory;
mod recurrence;
mod unswitch;

struct NaturalLoops {
    headers: Vec<Block>,
    owner: DenseMap<Block, Block>,
    parent: DenseMap<Block, Block>,
    preheader: DenseMap<Block, Block>,
}

impl NaturalLoops {
    fn discover(cfg: &ControlFlowGraph, dom: &DomTree) -> Self {
        let capacity = cfg.block_capacity();
        let mut loops = Self {
            headers: Vec::new(),
            owner: DenseMap::filled(capacity, Block::INVALID),
            parent: DenseMap::filled(capacity, Block::INVALID),
            preheader: DenseMap::filled(capacity, Block::INVALID),
        };
        let mut representative = DenseMap::from_fn(capacity, |block| block);
        let mut pending = Vec::new();
        // Find natural loops from backedges. Walk innermost headers first and
        // collapse each discovered region for the enclosing reverse walk.
        for &header in cfg.reverse_postorder().iter().rev() {
            let mut entry = None;
            let mut multiple_entries = false;
            for &pred in cfg.predecessors(header) {
                if dom.dominates(header, pred) {
                    pending.push(pred);
                } else {
                    multiple_entries |= entry.replace(pred).is_some();
                }
            }
            if pending.is_empty() {
                continue;
            }
            loops.headers.push(header);
            loops.owner[header] = header;
            if let Some(entry) = entry
                && !multiple_entries
                && cfg.successors(entry).count() == 1
                && dom.dominates(entry, header)
            {
                loops.preheader[header] = entry;
            }
            while let Some(mut root) = pending.pop() {
                while representative[root] != root {
                    let parent = representative[root];
                    representative[root] = representative[parent];
                    root = representative[root];
                }
                if root == header || !dom.dominates(header, root) {
                    continue;
                }
                representative[root] = header;
                if loops.owner[root] == root {
                    loops.parent[root] = header;
                } else {
                    loops.owner[root] = header;
                }
                pending.extend_from_slice(cfg.predecessors(root));
            }
        }
        loops
    }
}

pub(super) fn simplify_loops(f: &mut Function, cfg: &ControlFlowGraph, dom: &DomTree) -> bool {
    let loops = NaturalLoops::discover(cfg, dom);
    if loops
        .headers
        .iter()
        .all(|&header| loops.preheader[header] == Block::INVALID)
    {
        return false;
    }
    let mut facts = Facts::new(f, cfg, dom);
    hoist(f, dom, &loops, &mut facts);
    let mut counted = DenseMap::filled(f.block_capacity(), None);
    for &header in &loops.headers {
        counted[header] = counted_loop(f, cfg, dom, &loops, &facts, header);
    }
    let bounds = memory::Bounds::new(f, cfg, &facts);
    bounds.optimize(f, cfg, dom, &loops, &mut facts, &counted)
        || delete_finite_dead_loops(f, cfg, dom, &loops, &facts, &counted)
        || recurrence::collapse(f, cfg, dom, &loops, &facts, &counted)
        || fusion::fuse(f, cfg, dom, &loops, &facts, &counted, &bounds)
        || unswitch::run(f, cfg, dom, &loops, &facts.defs)
}

fn hoist(f: &mut Function, dom: &DomTree, loops: &NaturalLoops, facts: &mut Facts) {
    // A definition available in the outermost preheader can be placed there
    // directly. Otherwise try the nearest preheader. Work per instruction
    // stays bounded even in deep nests.
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
    // cheap recomputation for spills. Restrict new kinds of hoists to linear,
    // low-pressure bodies. Count definitions as an upper bound on loop-local
    // vector register demand and leave room for incoming live values.
    const VECTOR_BUDGET: usize = 12;
    let mut vector_values = DenseMap::filled(f.block_capacity(), 0);
    for &block in dom.preorder() {
        let Some(pressure) = vector_values.get_mut(loops.owner[block]) else {
            continue;
        };
        if *pressure > VECTOR_BUDGET {
            continue;
        }
        if block != loops.owner[block] && matches!(f.terminator(block), Terminator::Branch { .. }) {
            *pressure = VECTOR_BUDGET + 1;
            continue;
        }
        *pressure += f
            .params(block)
            .iter()
            .copied()
            .chain(f.instructions(block).filter_map(|inst| f.result(inst)))
            .filter(|&v| match f.value_type(v) {
                Type::Vector(_) => true,
                ty => ty
                    .lane()
                    .is_some_and(|lane| lane.float() || lane.bytes() == 16),
            })
            .count();
    }

    let mut total_divisions = Vec::new();
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
        let extended = nearest != Block::INVALID && vector_values[header] <= VECTOR_BUDGET;
        total_divisions.clear();
        if extended {
            for inst in f.instructions(block) {
                if let Op::Bin(Binary::Div | Binary::Rem, _, divisor)
                | Op::Simd(Simd::Bin(Binary::Div | Binary::Rem, _, divisor)) = *f.op(inst)
                    && facts.nontrapping_divisor(f, divisor)
                {
                    total_divisions.push(f.result(inst).unwrap());
                }
            }
            total_divisions.sort_unstable();
        }
        f.hoist_instructions(block, |op, result, types| {
            let scalar_i32 = matches!(op, Op::Bin(kind, a, _) if types[a] == Type::I32
                && !matches!(kind, Binary::Div | Binary::Rem));
            if !scalar_i32 && !extended {
                return None;
            }
            let cheap = match op {
                Op::Bin(kind, ..) | Op::Simd(Simd::Bin(kind, ..)) => {
                    !matches!(kind, Binary::Pow | Binary::Atan2 | Binary::Hypot)
                }
                Op::Unary(kind, ..) | Op::Simd(Simd::Unary(kind, ..)) => matches!(
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
                | Op::Simd(
                    Simd::Cast(..)
                    | Simd::Bitcast(..)
                    | Simd::Ternary(..)
                    | Simd::LaneCount(_)
                    | Simd::Iota(_)
                    | Simd::Splat(_),
                ) => true,
                _ => false,
            };
            if !cheap
                || (!op.pure(Some(types[result]))
                    && total_divisions.binary_search(&result).is_err())
            {
                return None;
            }
            let target = [if scalar_i32 { outermost } else { nearest }, nearest]
                .into_iter()
                .find(|&target| {
                    if target == Block::INVALID {
                        return false;
                    }
                    let mut available = true;
                    let mut operands = op;
                    operands.map_operands(|value| {
                        available &= facts.available(value, target, dom).is_some();
                        value
                    });
                    available
                })?;
            let mut op = op;
            op.map_operands(|v| facts.root(v));
            facts.defs[result].block = target;
            Some((target, op))
        });
    }
}

/// Positive counts that can be compared without fixing the backend's SIMD width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Count {
    Fixed(i32),
    Lanes(usize),
}

impl Count {
    fn of(f: &Function, facts: &Facts, value: Value) -> Option<Self> {
        match facts.op(f, value)? {
            Op::Const(n) if n > 0 => Some(Self::Fixed(n)),
            Op::Simd(Simd::LaneCount(lane)) => Some(Self::Lanes(lane.bytes())),
            _ => None,
        }
    }

    fn access(ty: Type, scale: usize) -> Self {
        match ty {
            Type::Vector(_) => Self::Lanes(scale),
            _ => Self::Fixed(ty.bytes(0).div_ceil(scale) as i32),
        }
    }

    fn at_width(self, bytes: usize) -> i32 {
        match self {
            Self::Fixed(n) => n,
            Self::Lanes(element) => bytes.div_ceil(element) as i32,
        }
    }

    fn at_most(self, other: Self) -> bool {
        // Counts are constant or linear in the supported vector widths.
        [16, 64]
            .into_iter()
            .all(|bytes| self.at_width(bytes) <= other.at_width(bytes))
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

/// Normalize an edge that guarantees a signed lower bound.
fn at_least(kind: Binary, a: Value, b: Value) -> Option<(Value, Value, bool)> {
    match kind {
        Binary::Ge => Some((a, b, true)),
        Binary::Le => Some((b, a, true)),
        Binary::Lt => Some((a, b, false)),
        Binary::Gt => Some((b, a, false)),
        _ => None,
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
    let (iv, bound, body, exit, packet) =
        if kind == Binary::Lt && f.params(header).binary_search(&facts.root(a)).is_ok() {
            (a, b, body, exit, None)
        } else {
            let (remaining, amount, yes) = at_least(kind, a, b)?;
            let Op::Bin(Binary::Sub, bound, iv) = facts.op(f, remaining)? else {
                return None;
            };
            let (body, exit) = if yes { (body, exit) } else { (exit, body) };
            (
                iv,
                bound,
                body,
                exit,
                Some((Count::of(f, facts, amount)?, facts.root(remaining))),
            )
        };
    let iv = facts.root(iv);
    let bound = facts.available(bound, preheader, dom)?;
    let index = f.params(header).binary_search(&iv).ok()?;
    if f.value_type(iv) != Type::I32
        || loops.owner[f.edge(body).0] != header
        || loops.owner[f.edge(exit).0] == header
    {
        return None;
    }
    let Terminator::Jump(enter) = *f.terminator(preheader) else {
        return None;
    };
    let start = facts.root(f.edge(enter).1[index]);
    let mut step = None;
    for &latch in cfg.predecessors(header) {
        if latch == preheader {
            continue;
        }
        for edge in f
            .terminator(latch)
            .edges()
            .filter(|&e| f.edge(e).0 == header)
        {
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
            .is_none_or(|n| n > i32::MAX - (step.at_width(64) - 1))
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
        if other_path
            || f.instructions(block)
                .any(|inst| !f.op(inst).pure(f.result(inst).map(|v| f.value_type(v))))
        {
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
