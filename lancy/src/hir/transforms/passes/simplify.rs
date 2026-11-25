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

use super::{checkpoint, dead, facts::Facts, local_cfg, loops, memory, sccp};

fn entry_literal(
    f: &mut Function,
    aliases: &mut DenseMap<Value, Value>,
    literals: &mut HashMap<(Lane, u128), Value>,
    lane: Lane,
    bits: u128,
) -> Value {
    *literals.entry((lane, bits)).or_insert_with(|| {
        let value = f.prepend(f.entry(), Op::literal(lane, bits)).unwrap();
        assert_eq!(aliases.insert(value), value);
        value
    })
}

/// The operation for a constant SCCP fact. A vector fact is uniform, so it
/// is a `Splat` of the scalar literal (see `sccp::Fact::Constant`).
fn constant_op(ty: Type, bits: u128, literals: &HashMap<(Lane, u128), Value>) -> Option<Op> {
    match ty {
        Type::Vector(lane) => Some(Op::Simd(Simd::Splat(*literals.get(&(lane, bits))?))),
        _ => Some(Op::literal(ty.lane()?, bits)),
    }
}

/// Canonicalize dataflow before instruction selection. Division,
/// memory operations and checks remain ordered on every executable path.
pub(crate) fn simplify(f: &mut Function) -> Result<(), IrError> {
    let cfg = ControlFlowGraph::compute(f)?;
    simplify_with_cfg(f, &cfg, None)
}

fn simplify_with_cfg(
    f: &mut Function,
    cfg: &ControlFlowGraph,
    verified_dom: Option<&DomTree>,
) -> Result<(), IrError> {
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
    let mut changed_aliases = !replacements.is_empty();
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
                            Some(left_is_min)
                        } else if (yes, no) == (y, x) {
                            Some(!left_is_min)
                        } else {
                            None
                        };
                        if let Some(select_min) = select_min {
                            let kind = if select_min { Binary::Min } else { Binary::Max };
                            return Some(Op::Simd(Simd::Bin(kind, yes, no)));
                        }
                    }
                }
                _ => {}
            }
            if let Some(dst) = result {
                if let Op::Copy(v) = op {
                    aliases[dst] = aliases[v];
                    changed_aliases = true;
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
                                if sccp::select_truth(types[c], bits) {
                                    yes
                                } else {
                                    no
                                }
                            })
                        }
                    }
                    _ => None,
                };
                if let Some(v) = copy {
                    aliases[dst] = aliases[v];
                    changed_aliases = true;
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
        f.retain_block_params(|p| aliases[p] == p);
    }
    if changed_aliases {
        f.substitute_values(&aliases);
    }
    changed_cfg |= local_cfg::convert_branches(f, cfg);
    let rebuilt = if changed_cfg {
        f.prune_unreachable()?;
        Some(ControlFlowGraph::compute(f)?)
    } else {
        None
    };
    checkpoint(f, "sccp and local_cfg");
    let cfg = rebuilt.as_ref().unwrap_or(cfg);
    let verified_dom = verified_dom.filter(|_| rebuilt.is_none());
    let computed_dom = verified_dom.is_none().then(|| DomTree::compute(cfg));
    let dom = verified_dom.or(computed_dom.as_ref()).unwrap();
    let updated = loops::run(f, cfg, dom)?;
    checkpoint(f, "loops");
    let (cfg, dom) = updated
        .rebuilt
        .as_ref()
        .map_or((cfg, dom), |(cfg, dom)| (cfg, dom));
    let (mut facts, mut bounds) = updated.analysis.map_or_else(
        || (Facts::new(f, cfg, dom), None),
        |(facts, bounds)| (facts, Some(bounds)),
    );
    // Physical forwarding across loop edges can lengthen hot live ranges.
    if !cfg.has_cycle() {
        facts.forward_acyclic_params(f, cfg);
    }
    super::bounds::remove_full_packet_checks(f, cfg, dom, &facts, &mut bounds);
    checkpoint(f, "bounds");
    remove_redundant_checks(f, dom);
    checkpoint(f, "redundant checks");
    memory::optimize(f, cfg, dom, &mut facts, bounds.as_ref());
    checkpoint(f, "memory");
    dead::eliminate(f, cfg, &facts);
    checkpoint(f, "dead");
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

/// Rebind calls to optimized callees when the caller explicitly keeps them out of line.
pub(crate) fn simplify_bodies(
    module: &Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<Function, IrError> {
    let mut bodies = DenseMap::with_capacity(module.functions.len());
    for (id, source) in module.functions.iter() {
        let mut body = source.function.clone();
        simplify_with_cfg(&mut body, &source.cfg, Some(&source.dom))?;
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
