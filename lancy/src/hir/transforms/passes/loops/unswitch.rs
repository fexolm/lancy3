use super::{Facts, NaturalLoops};
use crate::hir::transforms::{ControlFlowGraph, DomTree};
use crate::hir::{Block, Edge, Function, Op, Terminator, Value};
use std::collections::{BTreeSet, HashMap};

use super::super::limits::{UNSWITCH_BLOCKS, UNSWITCH_COST};

#[derive(Default)]
struct Region {
    blocks: Vec<Block>,
    branch: Option<(Block, Value)>,
    exit: Option<(Block, Block)>,
    cost: usize,
    unsupported: bool,
    live_out: BTreeSet<Value>,
    users: BTreeSet<Block>,
}

pub(super) fn run(
    f: &mut Function,
    cfg: &ControlFlowGraph,
    dom: &DomTree,
    loops: &NaturalLoops,
    facts: &Facts,
) -> bool {
    let mut regions = HashMap::new();
    for &header in &loops.headers {
        let preheader = loops.preheader[header];
        if preheader != Block::INVALID && matches!(f.terminator(preheader), Terminator::Jump(_)) {
            regions.insert(header, Region::default());
        }
    }
    for &parent in loops.parent.values() {
        regions.remove(&parent);
    }
    let order = cfg.reverse_postorder();
    for &block in order {
        let header = loops.owner[block];
        let Some(region) = regions.get_mut(&header) else {
            continue;
        };
        let preheader = loops.preheader[header];
        if region.unsupported {
            continue;
        }
        if block != header
            && cfg
                .predecessors(block)
                .iter()
                .any(|&pred| loops.owner[pred] != header)
        {
            region.unsupported = true;
            continue;
        }
        region.blocks.push(block);
        region.cost += f.params(block).len();
        for inst in f.instructions(block) {
            region.cost += 1;
            if let Op::Call(site) = *f.op(inst) {
                region.cost += f.call(site).1.len();
            }
        }
        let term = *f.terminator(block);
        for edge in term.edges() {
            let (target, args) = f.edge(edge);
            region.cost += args.len();
            if loops.owner[target] != header
                && (region.exit.replace((block, target)).is_some()
                    || cfg.predecessors(target) != [block])
            {
                region.unsupported = true;
            }
        }
        region.unsupported |= region.blocks.len() > UNSWITCH_BLOCKS
            || region.cost > UNSWITCH_COST
            || matches!(term, Terminator::Return(_));
        // The predicate is already computed on entry, including for zero
        // iterations. Both branch arms must lead back into the loop region.
        if region.branch.is_none()
            && let Terminator::Branch { cond, yes, no } = term
            && let Some(invariant) = facts.available(cond, preheader, dom)
            && loops.owner[f.edge(yes).0] == header
            && loops.owner[f.edge(no).0] == header
        {
            region.branch = Some((block, invariant));
        }
    }

    let eligible =
        |region: &Region| !region.unsupported && region.branch.is_some() && region.exit.is_some();
    regions.retain(|_, region| eligible(region));
    if regions.is_empty() {
        return false;
    }
    for region in regions.values() {
        let (branch, cond) = region.branch.unwrap();
        let Terminator::Branch { yes, no, .. } = *f.terminator(branch) else {
            unreachable!();
        };
        // Record direct uses of these roots before cloning any loop. An
        // earlier loop may need an exit parameter for a later predicate.
        f.set_terminator(branch, Terminator::Branch { cond, yes, no });
    }
    // Find direct SSA uses outside each candidate. Arguments on the original
    // exit edge are already in the right form and will be remapped by cloning.
    for &block in order {
        let mut record_use = |v: Value| {
            let owner = loops.owner[facts.defs[v].block];
            if owner != loops.owner[block]
                && let Some(region) = regions.get_mut(&owner)
            {
                region.live_out.insert(v);
                region.users.insert(block);
            }
        };
        for inst in f.instructions(block) {
            f.visit_operands(inst, &mut record_use);
        }
        let term = *f.terminator(block);
        if let Terminator::Return(v) | Terminator::Branch { cond: v, .. } = term {
            record_use(v);
        }
        for edge in term.edges() {
            for &v in f.edge(edge).1 {
                record_use(v);
            }
        }
    }

    // Transform in dominance order so downstream clones see the shared exit
    // parameters of any earlier loop they use.
    for &header in order {
        if let Some(region) = regions.get(&header) {
            split(f, loops.preheader[header], region);
        }
    }
    true
}

fn split(f: &mut Function, preheader: Block, region: &Region) {
    let (exiting, exit) = region.exit.unwrap();
    let (branch, _) = region.branch.unwrap();

    // The only incoming edge to the exit is from the loop, so the exit
    // dominates all direct external uses of loop-defined values.
    if !region.live_out.is_empty() {
        let replacements: HashMap<_, _> = region
            .live_out
            .iter()
            .map(|&v| (v, f.append_typed_param(exit, f.value_type(v))))
            .collect();
        for &block in &region.users {
            let insts: Vec<_> = f.instructions(block).collect();
            for inst in insts {
                let op = map_op(f, *f.op(inst), &replacements);
                f.replace_op_preserving_type(inst, op);
            }
            let term = map_term(f, *f.terminator(block), &HashMap::new(), &replacements);
            f.set_terminator(block, term);
        }
        let term = map_terminator(
            *f.terminator(exiting),
            |v| v,
            |edge| {
                let (target, args) = f.edge(edge);
                if target != exit {
                    return edge;
                }
                let mut args = args.to_vec();
                args.extend(region.live_out.iter().copied());
                f.add_edge(target, &args)
            },
        );
        f.set_terminator(exiting, term);
    }

    let mut blocks = HashMap::new();
    let mut values = HashMap::new();
    for &block in &region.blocks {
        let copy = f.create_block();
        blocks.insert(block, copy);
        for param in f.params(block).to_vec() {
            values.insert(param, f.append_typed_param(copy, f.value_type(param)));
        }
    }
    for &block in &region.blocks {
        let insts: Vec<_> = f.instructions(block).collect();
        for inst in insts {
            let op = map_op(f, *f.op(inst), &values);
            let copy = f
                .append_inst(blocks[&block], op)
                .expect("a copy of a verified operation");
            if let Some(copy) = f.result(copy) {
                values.insert(f.result(inst).unwrap(), copy);
            }
        }
    }

    let Terminator::Branch { cond, yes, no } = *f.terminator(branch) else {
        unreachable!()
    };
    for &block in &region.blocks {
        let term = if block == branch {
            Terminator::Jump(no)
        } else {
            *f.terminator(block)
        };
        let term = map_term(f, term, &blocks, &values);
        f.set_terminator(blocks[&block], term);
    }
    f.set_terminator(branch, Terminator::Jump(yes));
    let Terminator::Jump(yes) = *f.terminator(preheader) else {
        unreachable!()
    };
    let no = map_edge(f, yes, &blocks, &values);
    f.set_terminator(preheader, Terminator::Branch { cond, yes, no });
}

fn map_op(f: &mut Function, mut op: Op, values: &HashMap<Value, Value>) -> Op {
    let map = |v| values.get(&v).copied().unwrap_or(v);
    if let Op::Call(site) = op {
        let (callee, args) = f.call(site);
        if args.iter().any(|v| values.contains_key(v)) {
            let args: Vec<_> = args.iter().copied().map(map).collect();
            op = Op::Call(f.add_call(callee.clone(), &args));
        }
    } else {
        op.map_operands(map);
    }
    op
}

fn map_edge(
    f: &mut Function,
    edge: Edge,
    blocks: &HashMap<Block, Block>,
    values: &HashMap<Value, Value>,
) -> Edge {
    let (target, args) = f.edge(edge);
    if !blocks.contains_key(&target) && args.iter().all(|v| !values.contains_key(v)) {
        return edge;
    }
    let args: Vec<_> = args
        .iter()
        .map(|v| values.get(v).copied().unwrap_or(*v))
        .collect();
    f.add_edge(blocks.get(&target).copied().unwrap_or(target), &args)
}

fn map_term(
    f: &mut Function,
    term: Terminator,
    blocks: &HashMap<Block, Block>,
    values: &HashMap<Value, Value>,
) -> Terminator {
    map_terminator(
        term,
        |v| values.get(&v).copied().unwrap_or(v),
        |edge| map_edge(f, edge, blocks, values),
    )
}

fn map_terminator(
    term: Terminator,
    map: impl Fn(Value) -> Value,
    mut edge: impl FnMut(Edge) -> Edge,
) -> Terminator {
    match term {
        Terminator::Return(v) => Terminator::Return(map(v)),
        Terminator::Jump(e) => Terminator::Jump(edge(e)),
        Terminator::Branch { cond, yes, no } => Terminator::Branch {
            cond: map(cond),
            yes: edge(yes),
            no: edge(no),
        },
    }
}
