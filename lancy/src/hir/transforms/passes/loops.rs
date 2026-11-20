use crate::hir::transforms::{ControlFlowGraph, DomTree};
use crate::hir::{Binary, Block, Function, Inst, Op, Terminator, Type, Value, ssa::Edge};
use crate::support::slotmap::DenseMap;
mod facts;
mod fusion;
mod memory;
mod recurrence;
mod unswitch;
use facts::Facts;

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

#[derive(Clone, Copy)]
struct Definition {
    block: Block,
    inst: Inst,
}

fn definitions(f: &Function) -> DenseMap<Value, Definition> {
    let mut defs = DenseMap::filled(
        f.value_count(),
        Definition {
            block: Block::INVALID,
            inst: Inst::INVALID,
        },
    );
    for block in f.blocks() {
        for &v in f.params(block) {
            defs[v].block = block;
        }
        for inst in f.instructions(block) {
            if let Some(v) = f.result(inst) {
                defs[v] = Definition { block, inst };
            }
        }
    }
    defs
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
    memory::optimize(f, cfg, dom, &loops, &mut facts, &counted);
    delete_finite_dead_loops(f, cfg, dom, &loops, &facts, &counted)
        || recurrence::collapse(f, cfg, dom, &loops, &facts, &counted)
        || fusion::fuse(f, cfg, dom, &loops, &facts, &counted)
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
    for &block in dom.preorder() {
        let header = loops.owner[block];
        if header == Block::INVALID {
            continue;
        }
        let nearest = loops.preheader[header];
        let outermost = outer[header];
        f.hoist_instructions(block, |op, result, types| {
            let available = |target, v| facts.available(v, target, dom).is_some();
            let target = [outermost, nearest].into_iter().find(|&target| match op {
                Op::Bin(kind, a, b)
                    if types[a] == Type::I32 && !matches!(kind, Binary::Div | Binary::Rem) =>
                {
                    available(target, a) && available(target, b)
                }
                #[cfg(feature = "experimental-simd")]
                _ => super::experimental::invariant_simd(op, |v| available(target, v)),
                #[cfg(not(feature = "experimental-simd"))]
                _ => false,
            })?;
            let mut op = op;
            op.map_operands(|v| facts.root(v));
            facts.defs[result].block = target;
            Some((target, op))
        });
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
}

/// A strict signed bound and unit increment terminate without wrapping.
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
    let Op::Bin(Binary::Lt, iv, bound) = facts.op(f, cond)? else {
        return None;
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
            if !(a == iv && facts.constant(f, b) == Some(1)
                || b == iv && facts.constant(f, a) == Some(1))
            {
                return None;
            }
        }
    }
    Some(CountedLoop {
        header,
        preheader,
        body,
        exit,
        iv,
        start,
        bound,
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
