use crate::hir::{
    Binary, Block, Function, IrError, Op, Simd, Terminator, Ternary, Type,
    ssa::VerifiedBody,
    transforms::{ControlFlowGraph, DomTree},
};
use crate::module::{FunctionIds, Module};
use crate::support::slotmap::DenseMap;
use std::{collections::HashMap, sync::Arc};

use super::{local_cfg, sccp};

/// Canonicalize dataflow before instruction selection. Division,
/// memory operations and checks remain ordered on every executable path.
pub(crate) fn simplify(f: &mut Function) -> Result<(), IrError> {
    let cfg = ControlFlowGraph::compute(f)?;
    simplify_with_cfg(f, &cfg)
}

fn simplify_with_cfg(f: &mut Function, cfg: &ControlFlowGraph) -> Result<(), IrError> {
    let facts = sccp::analyze(f, cfg);
    let mut replacements = Vec::new();
    let mut entry_constants = HashMap::new();
    let entry = f.entry();
    for &b in &cfg.reverse_postorder()[1..] {
        for &param in f.params(b) {
            if let Some(n) = facts[param].constant() {
                replacements.push((param, n));
            }
        }
    }
    let mut aliases = DenseMap::from_fn(f.value_count(), |v| v);
    for &(param, n) in &replacements {
        let value = *entry_constants.entry(n).or_insert_with(|| {
            let value = f.append(entry, Op::Const(n)).unwrap();
            assert_eq!(aliases.insert(value), value);
            value
        });
        aliases[param] = value;
    }
    let mut vector_orderings = HashMap::new();
    let mut changed_cfg = false;
    for &b in cfg.reverse_postorder() {
        f.rewrite_instructions(b, |op, result, types| {
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
                if let Some(n) = facts.get(dst).and_then(|fact| fact.constant()) {
                    return Some(Op::Const(n));
                }
                let copy = match op {
                    Op::Bin(kind, x, y) if types[x] == Type::I32 => {
                        let a = facts[x].constant();
                        let b = facts[y].constant();
                        match sccp::reduce(kind, aliases[x], aliases[y], a, b) {
                            Some(Op::Copy(v)) => Some(v),
                            Some(op) => return Some(op),
                            None if a.is_some() && b.is_none() => {
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
                    Op::Ternary(Ternary::Select, c, yes, no) => match facts[c].constant() {
                        Some(0) => Some(no),
                        Some(_) => Some(yes),
                        None if aliases[yes] == aliases[no] => Some(yes),
                        None => None,
                    },
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
            && let Some(n) = facts[cond].constant()
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
    if cfg.has_cycle() {
        // Mechanical forwarding across loop edges can lengthen hot live ranges
        // and prevent the allocator from splitting them at block boundaries.
        local_cfg::hoist_loop_invariants(f, cfg, &dom);
    } else {
        local_cfg::forward_params(f, cfg);
    }
    remove_redundant_checks(f, &dom);
    remove_dead_computations(f, cfg);
    Ok(())
}

pub(super) fn fold_i32(op: Binary, a: i32, b: i32) -> Option<i32> {
    Some(match op {
        Binary::Add => a.wrapping_add(b),
        Binary::Sub => a.wrapping_sub(b),
        Binary::Mul => a.wrapping_mul(b),
        Binary::Div => a.checked_div(b)?,
        Binary::Rem => a.checked_rem(b)?,
        Binary::And => a & b,
        Binary::Or => a | b,
        Binary::Xor => a ^ b,
        Binary::Eq => -i32::from(a == b),
        Binary::Ne => -i32::from(a != b),
        Binary::Lt => -i32::from(a < b),
        Binary::Le => -i32::from(a <= b),
        Binary::Gt => -i32::from(a > b),
        Binary::Ge => -i32::from(a >= b),
        Binary::Shl => a.wrapping_shl(b as u32),
        Binary::Shr => a.wrapping_shr(b as u32),
        Binary::Min => a.min(b),
        Binary::Max => a.max(b),
        Binary::RotateLeft => a.rotate_left(b as u32),
        Binary::RotateRight => a.rotate_right(b as u32),
        Binary::SaturatingAdd => a.saturating_add(b),
        Binary::SaturatingSub => a.saturating_sub(b),
        _ => return None,
    })
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
                if !live[result] && op.pure(Some(f.value_type(result))) {
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
