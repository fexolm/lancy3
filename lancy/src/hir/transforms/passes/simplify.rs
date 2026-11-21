#[cfg(feature = "experimental-simd")]
use super::experimental;
use crate::hir::{
    Binary, Block, Function, IrError, Lane, Op, Simd, Terminator, Ternary, Type, Value,
    ssa::VerifiedBody,
    transforms::{ControlFlowGraph, DomTree},
};
use crate::module::{FunctionIds, Module};
use crate::support::slotmap::DenseMap;
use std::{collections::HashMap, sync::Arc};

use super::{local_cfg, loops, memory, sccp};

fn entry_literal(
    f: &mut Function,
    aliases: &mut DenseMap<Value, Value>,
    literals: &mut HashMap<(Lane, u128), Value>,
    lane: Lane,
    bits: u128,
) -> Value {
    *literals.entry((lane, bits)).or_insert_with(|| {
        let value = f.prepend(f.entry(), sccp::literal(lane, bits)).unwrap();
        assert_eq!(aliases.insert(value), value);
        value
    })
}

fn constant_op(ty: Type, bits: u128, literals: &HashMap<(Lane, u128), Value>) -> Option<Op> {
    match ty {
        Type::Vector(lane) => Some(Op::Simd(Simd::Splat(*literals.get(&(lane, bits))?))),
        _ => Some(sccp::literal(ty.lane()?, bits)),
    }
}

/// Canonicalize dataflow before instruction selection. Division,
/// memory operations and checks remain ordered on every executable path.
pub(crate) fn simplify(f: &mut Function) -> Result<(), IrError> {
    let cfg = ControlFlowGraph::compute(f)?;
    simplify_with_cfg(f, &cfg)
}

fn simplify_with_cfg(f: &mut Function, cfg: &ControlFlowGraph) -> Result<(), IrError> {
    let facts = sccp::analyze(f, cfg);
    let known = |value| facts.get(value).and_then(|fact| fact.bits());
    let mut replacements = Vec::new();
    let mut entry_constants = HashMap::new();
    let mut literals = HashMap::new();
    let mut uniform = Vec::new();
    let entry = f.entry();
    for &b in cfg.reverse_postorder() {
        if b != entry {
            for &param in f.params(b) {
                if let Some(bits) = known(param) {
                    replacements.push((param, bits));
                }
            }
        }
        for i in f.instructions(b) {
            if let Some(dst) = f.result(i)
                && let Some(bits) = known(dst)
                && let Type::Vector(lane) = f.value_type(dst)
                && !matches!(f.op(i), Op::Simd(Simd::Splat(_)))
            {
                uniform.push((lane, bits));
            }
        }
    }
    let mut aliases = DenseMap::from_fn(f.value_count(), |v| v);
    for &(param, bits) in &replacements {
        let ty = f.value_type(param);
        let value = match ty {
            Type::Vector(lane) => *entry_constants.entry((lane, bits)).or_insert_with(|| {
                let scalar = entry_literal(f, &mut aliases, &mut literals, lane, bits);
                let value = f.append(entry, Op::Simd(Simd::Splat(scalar))).unwrap();
                assert_eq!(aliases.insert(value), value);
                value
            }),
            _ => entry_literal(f, &mut aliases, &mut literals, ty.lane().unwrap(), bits),
        };
        aliases[param] = value;
    }
    for (lane, bits) in uniform {
        entry_literal(f, &mut aliases, &mut literals, lane, bits);
    }
    let mut vector_orderings = HashMap::new();
    #[cfg(feature = "experimental-simd")]
    let mut definitions = HashMap::new();
    let mut changed_cfg = false;
    for &b in cfg.reverse_postorder() {
        #[cfg(feature = "experimental-simd")]
        let mut vector_constants = HashMap::new();
        f.rewrite_instructions(b, |op, result, types| {
            #[cfg(feature = "experimental-simd")]
            let op = experimental::simplify_simd(
                op,
                result,
                types,
                &facts,
                &aliases,
                &mut definitions,
                &mut vector_constants,
            )?;
            match (op, result) {
                (
                    Op::Simd(Simd::Bin(
                        kind @ (Binary::Lt | Binary::Le | Binary::Gt | Binary::Ge),
                        x,
                        y,
                    )),
                    Some(dst),
                ) if matches!(types[x], Type::Vector(lane) if !lane.float()) => {
                    vector_orderings.insert(dst, (kind, aliases[x], aliases[y]));
                }
                (Op::Simd(Simd::Ternary(Ternary::Select, mask, yes, no)), _) => {
                    if let Some(&(kind, x, y)) = vector_orderings.get(&aliases[mask]) {
                        let left_is_min = matches!(kind, Binary::Lt | Binary::Le);
                        let yes = aliases[yes];
                        let no = aliases[no];
                        let select_min = if (yes, no) == (x, y) {
                            left_is_min
                        } else if (yes, no) == (y, x) {
                            !left_is_min
                        } else {
                            return Some(op);
                        };
                        let kind = if select_min { Binary::Min } else { Binary::Max };
                        return Some(Op::Simd(Simd::Bin(kind, yes, no)));
                    }
                }
                _ => {}
            }
            if let Some(dst) = result {
                if let Op::Copy(v) = op {
                    aliases[dst] = aliases[v];
                    return None;
                }
                if let Some(bits) = known(dst)
                    && !matches!(op, Op::Simd(Simd::Splat(_)))
                    && let Some(literal) = constant_op(types[dst], bits, &literals)
                {
                    return Some(literal);
                }
                let copy = match op {
                    Op::Bin(kind, x, y) | Op::Simd(Simd::Bin(kind, x, y))
                        if types[x].lane().is_some_and(|lane| !lane.float()) =>
                    {
                        let a = known(x);
                        let b = known(y);
                        match sccp::reduce(kind, types[x], aliases[x], aliases[y], a, b) {
                            Some(sccp::Reduced::Copy(v)) => Some(v),
                            Some(sccp::Reduced::Constant(bits)) => {
                                return constant_op(types[dst], bits, &literals).or(Some(op));
                            }
                            None if matches!(op, Op::Bin(..)) && a.is_some() && b.is_none() => {
                                let flipped = match kind {
                                    Binary::Lt => Binary::Gt,
                                    Binary::Le => Binary::Ge,
                                    Binary::Gt => Binary::Lt,
                                    Binary::Ge => Binary::Le,
                                    Binary::Add
                                    | Binary::Mul
                                    | Binary::And
                                    | Binary::Or
                                    | Binary::Xor
                                    | Binary::Min
                                    | Binary::Max
                                    | Binary::Eq
                                    | Binary::Ne
                                    | Binary::SaturatingAdd => kind,
                                    _ => return Some(op),
                                };
                                return Some(Op::Bin(flipped, y, x));
                            }
                            _ => None,
                        }
                    }
                    Op::Ternary(Ternary::Select, c, yes, no)
                    | Op::Simd(Simd::Ternary(Ternary::Select, c, yes, no)) => {
                        if aliases[yes] == aliases[no] {
                            Some(yes)
                        } else {
                            known(c).map(|bits| {
                                let cond = if matches!(op, Op::Simd(_)) {
                                    bits >> (types[c].lane().unwrap().bytes() * 8 - 1) != 0
                                } else {
                                    bits != 0
                                };
                                if cond { yes } else { no }
                            })
                        }
                    }
                    _ => None,
                };
                if let Some(v) = copy {
                    aliases[dst] = aliases[v];
                    return None;
                }
            }
            Some(op)
        });
        if let Terminator::Branch { cond, yes, no } = *f.terminator(b)
            && let Some(n) = known(cond)
        {
            f.set_terminator(b, Terminator::Jump(if n != 0 { yes } else { no }));
            changed_cfg = true;
        }
    }
    if !replacements.is_empty() {
        f.remove_aliased_params(&aliases);
    }
    f.substitute_values(&aliases);
    changed_cfg |= local_cfg::convert_branches(f, cfg);
    let rebuilt;
    let cfg = if changed_cfg {
        f.prune_unreachable()?;
        rebuilt = ControlFlowGraph::compute(f)?;
        &rebuilt
    } else {
        cfg
    };
    let dom = DomTree::compute(cfg);
    let (updated_cfg, updated_dom);
    let (cfg, dom) = if cfg.has_cycle() {
        // Mechanical forwarding across loop edges can lengthen hot live ranges
        // and prevent the allocator from splitting them at block boundaries.
        if loops::simplify_loops(f, cfg, &dom) {
            f.prune_unreachable()?;
            let mut current_cfg = ControlFlowGraph::compute(f)?;
            let mut current_dom = DomTree::compute(&current_cfg);
            // Refresh loop proofs only after productive CFG edits. Two
            // more rounds cover short pipelines without unbounded work.
            for _ in 0..2 {
                if !current_cfg.has_cycle() || !loops::simplify_loops(f, &current_cfg, &current_dom)
                {
                    break;
                }
                f.prune_unreachable()?;
                current_cfg = ControlFlowGraph::compute(f)?;
                current_dom = DomTree::compute(&current_cfg);
            }
            updated_cfg = current_cfg;
            updated_dom = current_dom;
            (&updated_cfg, &updated_dom)
        } else {
            (cfg, &dom)
        }
    } else {
        local_cfg::forward_params(f, cfg);
        (cfg, &dom)
    };
    #[cfg(feature = "experimental-simd")]
    experimental::remove_full_packet_checks(f, cfg, dom);
    remove_redundant_checks(f, dom);
    memory::optimize(f, cfg, dom);
    remove_dead_computations(f, cfg);
    Ok(())
}

fn remove_redundant_checks(f: &mut Function, dom: &DomTree) {
    let mut checks = HashMap::new();
    for &b in dom.preorder() {
        f.rewrite_instructions(b, |op, _, _| {
            if let Op::BoundsCheck(i, n) = op {
                let previous = checks.entry((i, n)).or_insert(Block::INVALID);
                if dom.dominates(*previous, b) {
                    return None;
                }
                *previous = b;
            }
            Some(op)
        });
    }
}

fn remove_dead_computations(f: &mut Function, cfg: &ControlFlowGraph) {
    let mut live = DenseMap::filled(f.value_count(), false);
    #[cfg(feature = "experimental-simd")]
    let zero = experimental::zero_constants(f, cfg);
    for &b in cfg.reverse_postorder() {
        let term = *f.terminator(b);
        if let Terminator::Branch { cond: v, .. } | Terminator::Return(v) = term {
            live[v] = true;
        }
        for e in term.edges() {
            for &v in f.edge(e).1 {
                live[v] = true;
            }
        }
    }
    let mut instructions = Vec::new();
    for &b in cfg.reverse_postorder().iter().rev() {
        instructions.extend(f.instructions(b));
        for i in instructions.drain(..).rev() {
            let op = *f.op(i);
            if let Some(result) = f.result(i) {
                #[cfg(feature = "experimental-simd")]
                let first_lane = experimental::accesses_first_lane(op, &zero);
                #[cfg(not(feature = "experimental-simd"))]
                let first_lane = false;
                if !live[result] && (first_lane || op.pure(Some(f.value_type(result)))) {
                    continue;
                }
                live[result] = true;
            }
            f.visit_operands(i, |v| live[v] = true);
        }
        f.rewrite_instructions(b, |op, result, _| {
            result.is_none_or(|v| live[v]).then_some(op)
        });
    }
}

/// Rebind calls to optimized callees when the caller explicitly keeps them out of line.
pub(crate) fn simplify_bodies(
    module: &Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<Function, IrError> {
    let mut bodies = DenseMap::with_capacity(module.functions.len());
    for (id, source) in module.functions.iter() {
        let mut body = source.function.clone();
        simplify_with_cfg(&mut body, &source.cfg)?;
        for target in body.callees_mut() {
            if let Some(&child) = ids.get(&Arc::as_ptr(target))
                && let Some(callee) = bodies.get(child)
            {
                *target = Arc::clone(callee);
            }
        }
        if id == module.call_graph.root() {
            return Ok(body);
        }
        bodies.insert(Arc::new(body));
    }
    unreachable!("module has a root")
}
