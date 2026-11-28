use crate::hir::transforms::ControlFlowGraph;
use crate::hir::{Binary, Block, Edge, Function, Lane, Op, Simd, Terminator, Ternary, Type, Value};
use crate::support::slotmap::DenseMap;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy)]
enum Path {
    New,
    Visiting,
    Cycle,
    End { block: Block, distance: usize },
}

struct ForwardingPaths {
    next: DenseMap<Block, Option<Edge>>,
    paths: DenseMap<Block, Path>,
}

impl ForwardingPaths {
    fn new(f: &Function, cfg: &ControlFlowGraph) -> Self {
        let mut next = DenseMap::filled(f.block_capacity(), None);
        let mut paths = DenseMap::filled(f.block_capacity(), Path::New);
        for &block in cfg.reverse_postorder() {
            if f.instructions(block).next().is_none()
                && let Terminator::Jump(edge) = *f.terminator(block)
            {
                next[block] = Some(edge);
            }
        }
        let mut stack = Vec::new();
        for &start in cfg.reverse_postorder() {
            let mut block = start;
            let mut path = loop {
                match paths[block] {
                    Path::New => {
                        let Some(edge) = next[block] else {
                            paths[block] = Path::End { block, distance: 0 };
                            break paths[block];
                        };
                        paths[block] = Path::Visiting;
                        stack.push(block);
                        block = f.edge(edge).0;
                    }
                    Path::Visiting => break Path::Cycle,
                    known => break known,
                }
            };
            while let Some(block) = stack.pop() {
                if let Path::End { distance, .. } = &mut path {
                    *distance += 1;
                }
                paths[block] = path;
            }
        }
        Self { next, paths }
    }

    fn merge(&self, f: &Function, mut a: Block, mut b: Block) -> Option<Block> {
        if a == b {
            return Some(a);
        }
        let (
            Path::End {
                block: end_a,
                distance: mut da,
            },
            Path::End {
                block: end_b,
                distance: mut db,
            },
        ) = (self.paths[a], self.paths[b])
        else {
            return None;
        };
        if end_a != end_b {
            return None;
        }
        while a != b {
            if da >= db {
                a = f.edge(self.next[a].unwrap()).0;
                da -= 1;
            } else {
                b = f.edge(self.next[b].unwrap()).0;
                db -= 1;
            }
        }
        Some(a)
    }

    fn arguments(&self, f: &Function, mut edge: Edge, merge: Block) -> Vec<Value> {
        let mut bindings = HashMap::new();
        let mut args = Vec::new();
        loop {
            let (block, values) = f.edge(edge);
            args.clear();
            args.extend(
                values
                    .iter()
                    .map(|v| bindings.get(v).copied().unwrap_or(*v)),
            );
            if block == merge {
                return args;
            }
            // Downstream jumps can use any parameter defined along the path.
            bindings.extend(f.params(block).iter().copied().zip(args.iter().copied()));
            edge = self.next[block].unwrap();
        }
    }
}

/// Replace a branch joining through forwarding blocks with a numeric select.
/// Other predecessors of a forwarding block can keep using that block.
pub(super) fn convert_branches(f: &mut Function, cfg: &ControlFlowGraph) -> bool {
    if !cfg
        .reverse_postorder()
        .iter()
        .any(|&b| matches!(f.terminator(b), Terminator::Branch { .. }))
    {
        return false;
    }
    let paths = ForwardingPaths::new(f, cfg);
    let mut changed = false;
    for &block in cfg.reverse_postorder() {
        let Terminator::Branch { cond, yes, no } = *f.terminator(block) else {
            continue;
        };
        let yes_target = f.edge(yes).0;
        let Some(target) = paths.merge(f, yes_target, f.edge(no).0) else {
            if !cfg.has_cycle() {
                changed |= convert_computed_branch(f, cfg, block, cond, yes, no);
            }
            continue;
        };
        if target == block {
            continue;
        }
        let mut args = paths.arguments(f, yes, target);
        let right_args = paths.arguments(f, no, target);
        let (first, second) = {
            let mut differing = args
                .iter()
                .copied()
                .zip(right_args)
                .enumerate()
                .filter(|(_, (x, y))| x != y);
            (differing.next(), differing.next())
        };
        let Some((index, (x, y))) = first else {
            if yes_target == target {
                f.set_terminator(block, Terminator::Jump(yes));
            } else {
                f.jump(block, target, &args);
            }
            changed = true;
            continue;
        };
        let ty = f.value_type(x);
        if ty == Type::Ptr || second.is_some() {
            continue;
        }
        let cond = if matches!(ty, Type::Vector(_)) {
            // Branches test nonzero; SIMD selects test the sign bit per lane.
            let zero = f.append(block, Op::Const(0)).unwrap();
            let truth = f.append(block, Op::Bin(Binary::Ne, cond, zero)).unwrap();
            let mask_type = ty.mask_type().lane().unwrap().scalar();
            let mask = if mask_type == Type::I32 {
                truth
            } else {
                f.append(block, Op::Cast(mask_type, truth)).unwrap()
            };
            f.append(block, Op::Simd(Simd::Splat(mask))).unwrap()
        } else {
            cond
        };
        args[index] = f
            .append(block, Op::Ternary(Ternary::Select, cond, x, y))
            .unwrap();
        f.jump(block, target, &args);
        changed = true;
    }
    changed
}

/// A small straight-line diamond can feed a select even when its arms
/// compute the incoming value. Only total, wrapping i32 operations may
/// execute on the other path. Limit the extra work on a predictable branch.
fn convert_computed_branch(
    f: &mut Function,
    cfg: &ControlFlowGraph,
    block: Block,
    cond: Value,
    yes: Edge,
    no: Edge,
) -> bool {
    let arm = |edge| {
        let b = f.edge(edge).0;
        if b != block
            && cfg.predecessors(b) == [block]
            && let Terminator::Jump(exit) = *f.terminator(b)
        {
            let (join, _) = f.edge(exit);
            if join != b && join != block {
                return (join, Some(exit));
            }
        }
        (b, None)
    };
    let (mut join, mut yes_exit) = arm(yes);
    let (mut other, mut no_exit) = arm(no);
    if join != other {
        // One incoming edge may go directly to the merge, which itself
        // can end in a jump.
        if join == f.edge(no).0 {
            other = join;
            no_exit = None;
        } else if other == f.edge(yes).0 {
            join = other;
            yes_exit = None;
        }
    }
    if join != other || join == block {
        return false;
    }

    let inputs = |edge, exit| {
        let (arm, args) = f.edge(edge);
        let Some(exit) = exit else {
            return args.to_vec();
        };
        f.edge(exit)
            .1
            .iter()
            .map(|&v| f.param_index(arm, v).map_or(v, |i| args[i]))
            .collect()
    };
    let mut left = inputs(yes, yes_exit);
    let right = inputs(no, no_exit);
    let mut differing = left
        .iter()
        .zip(&right)
        .enumerate()
        .filter(|(_, (x, y))| x != y);
    let (Some((index, _)), None) = (differing.next(), differing.next()) else {
        return false;
    };
    if f.value_type(f.params(join)[index]) != Type::I32 {
        return false;
    }

    let mut budget = 4;
    let mut cheap = 8;
    for edge in [yes, no]
        .into_iter()
        .zip([yes_exit, no_exit])
        .filter_map(|(edge, exit)| exit.map(|_| edge))
    {
        for inst in f.instructions(f.edge(edge).0) {
            match *f.op(inst) {
                Op::Const(_) if cheap > 0 => cheap -= 1,
                Op::Cast(ty, v)
                    if ty == Lane::Usize.scalar() && f.value_type(v) == Type::I32 && cheap > 0 =>
                {
                    cheap -= 1;
                }
                Op::Bin(
                    Binary::Add
                    | Binary::Sub
                    | Binary::Mul
                    | Binary::And
                    | Binary::Or
                    | Binary::Xor,
                    a,
                    _,
                ) if f.value_type(a) == Type::I32 && budget > 0 => budget -= 1,
                _ => return false,
            }
        }
    }
    if budget == 4 && cheap == 8 {
        return false;
    }

    let mut speculate = |edge: Edge, exit: Option<Edge>, result: Value| {
        let (arm, args) = f.edge(edge);
        if exit.is_none() {
            return result;
        }
        let mut values: Vec<_> = f
            .params(arm)
            .iter()
            .copied()
            .zip(args.iter().copied())
            .collect();
        let insts: Vec<_> = f.instructions(arm).collect();
        let remap = |values: &[(Value, Value)], v| {
            values
                .iter()
                .find(|&&(from, _)| from == v)
                .map_or(v, |&(_, to)| to)
        };
        for inst in insts {
            let mut op = *f.op(inst);
            op.map_operands(|v| remap(&values, v));
            let original = f.result(inst).unwrap();
            let value = f.append(block, op).unwrap();
            values.push((original, value));
        }
        remap(&values, result)
    };
    let a = speculate(yes, yes_exit, left[index]);
    let b = speculate(no, no_exit, right[index]);
    left[index] = f
        .append(block, Op::Ternary(Ternary::Select, cond, a, b))
        .unwrap();
    f.jump(block, join, &left);
    true
}

pub(super) fn convert_computed_branches(
    f: &mut Function,
    cfg: &ControlFlowGraph,
    facts: &super::facts::Facts,
) -> bool {
    let mut changed = false;
    for &block in cfg.reverse_postorder() {
        if let Terminator::Branch { cond, yes, no } = *f.terminator(block)
            // A threshold can change frequently. Keep equality tests and
            // other predicates on their original control-flow paths.
            && matches!(
                facts.op(f, cond),
                Some(Op::Bin(Binary::Lt | Binary::Le | Binary::Gt | Binary::Ge, ..))
            )
        {
            changed |= convert_computed_branch(f, cfg, block, cond, yes, no);
        }
    }
    changed
}

/// Send each edge that passes a constant condition into an empty block,
/// which only branches on that parameter, straight to the chosen successor.
/// A short-circuit condition such as `a || b` joins its operands through
/// such a block. Threading the constant operand exposes the comparison of
/// the other one as the branch that guards the code, for the passes that
/// use dominating comparisons (bounds checks). Loop headers keep their
/// incoming edges, so loops stay reducible.
pub(super) fn thread_constant_branches(f: &mut Function, cfg: &ControlFlowGraph) -> bool {
    let rpo = cfg.reverse_postorder();
    let mut order = DenseMap::filled(f.block_capacity(), usize::MAX);
    for (i, &b) in rpo.iter().enumerate() {
        order[b] = i;
    }
    let mut candidates = Vec::new();
    for &b in rpo {
        if f.instructions(b).next().is_none()
            && let Terminator::Branch { cond, yes, no } = *f.terminator(b)
            && let Some(index) = f.param_index(b, cond)
            && cfg.predecessors(b).iter().all(|&p| order[p] < order[b])
        {
            candidates.push((b, index, yes, no));
        }
    }
    if candidates.is_empty() {
        return false;
    }
    // `b` stops dominating the chosen successor, so a parameter used past
    // its own branch must equal one value on every incoming edge, which
    // then replaces it.
    let mut owner = HashMap::new();
    for &(b, ..) in &candidates {
        owner.extend(f.params(b).iter().map(|&v| (v, b)));
    }
    let mut escaping = HashSet::new();
    let mut constants = HashMap::new();
    for &b in rpo {
        let mut note = |v: Value| {
            if owner.get(&v).is_some_and(|&o| o != b) {
                escaping.insert(v);
            }
        };
        for inst in f.instructions(b) {
            f.visit_operands(inst, &mut note);
            if let Op::Const(n) = *f.op(inst) {
                constants.insert(f.result(inst).unwrap(), n);
            }
        }
        let term = *f.terminator(b);
        term.operand().into_iter().for_each(&mut note);
        for edge in term.edges() {
            f.edge(edge).1.iter().copied().for_each(&mut note);
        }
    }
    let mut params = HashMap::new();
    if !escaping.is_empty() {
        for &b in rpo {
            params.extend(f.params(b).iter().enumerate().map(|(i, &v)| (v, (b, i))));
        }
    }
    let mut aliases = Vec::new();
    candidates.retain(|&(b, index, ..)| {
        let mut replaced = Vec::new();
        for (i, &param) in f.params(b).iter().enumerate() {
            if !escaping.contains(&param) {
                continue;
            }
            let Some(value) = (i != index)
                .then(|| uniform_argument(f, cfg, &params, b, i))
                .flatten()
            else {
                return false;
            };
            replaced.push((param, value));
        }
        aliases.extend(replaced);
        true
    });
    if !aliases.is_empty() {
        let mut map = DenseMap::from_fn(f.value_count(), |v| v);
        for (param, value) in aliases {
            map[param] = value;
        }
        f.substitute_values(&map);
    }
    let mut changed = false;
    for (b, index, yes, no) in candidates {
        for (pred, edge) in cfg.incoming_edges(b) {
            let (_, args) = f.edge(edge);
            let Some(&n) = constants.get(&args[index]) else {
                continue;
            };
            let incoming = args.to_vec();
            let (target, chosen) = f.edge(if n != 0 { yes } else { no });
            // Values other than parameters of `b` are defined in a strict
            // dominator of `b`, which also dominates its predecessor.
            let forwarded: Vec<_> = chosen
                .iter()
                .map(|&v| f.param_index(b, v).map_or(v, |i| incoming[i]))
                .collect();
            let threaded = f.add_edge(target, &forwarded);
            let retarget = |e: Edge| if e == edge { threaded } else { e };
            let term = match *f.terminator(pred) {
                Terminator::Jump(e) => Terminator::Jump(retarget(e)),
                Terminator::Branch { cond, yes, no } => Terminator::Branch {
                    cond,
                    yes: retarget(yes),
                    no: retarget(no),
                },
                Terminator::Return(_) => unreachable!("a predecessor has an edge"),
            };
            f.set_terminator(pred, term);
            changed = true;
        }
    }
    changed
}

/// The value that every edge into `block` passes for parameter `index`,
/// looking through parameters of blocks with a single incoming edge. It is
/// available at each predecessor, so it dominates `block`.
fn uniform_argument(
    f: &Function,
    cfg: &ControlFlowGraph,
    params: &HashMap<Value, (Block, usize)>,
    block: Block,
    index: usize,
) -> Option<Value> {
    let resolve = |mut v: Value| {
        // A chain of single-edge blocks is acyclic in reachable code; the
        // limit only bounds the walk.
        for _ in 0..8 {
            let Some(&(b, i)) = params.get(&v) else {
                break;
            };
            let mut incoming = cfg.incoming_edges(b);
            let (Some((_, edge)), None) = (incoming.next(), incoming.next()) else {
                break;
            };
            v = f.edge(edge).1[i];
        }
        v
    };
    let mut values = cfg
        .incoming_edges(block)
        .map(|(_, edge)| resolve(f.edge(edge).1[index]));
    let first = values.next()?;
    values.all(|v| v == first).then_some(first)
}
