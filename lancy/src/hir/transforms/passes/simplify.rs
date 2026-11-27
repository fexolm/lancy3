#[cfg(feature = "experimental-simd")]
use super::experimental;
use crate::hir::{
    Binary, Function, IrError, Lane, Op, Simd, Terminator, Ternary, Type, Value, ssa::VerifiedBody,
    transforms::ControlFlowGraph,
};
use crate::module::{FunctionId, Module};
use crate::support::slotmap::DenseMap;
use splice_target::Arch;
use std::{collections::HashMap, sync::Arc};

use super::pipeline::{self, Analyses, Held, Preserved};
use super::{local_cfg, sccp};

fn entry_literal(
    f: &mut Function,
    aliases: &mut DenseMap<Value, Value>,
    literals: &mut HashMap<(Lane, u128), Value>,
    lane: Lane,
    bits: u128,
) -> Value {
    *literals.entry((lane, bits)).or_insert_with(|| {
        let entry = f.entry();
        let inst = f
            .insert_after(entry, None, Op::literal(lane, bits))
            .unwrap();
        let value = f.result(inst).unwrap();
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

/// Canonicalize dataflow before instruction selection for `arch`.
pub(crate) fn simplify(f: &mut Function, arch: Arch) -> Result<(), IrError> {
    let cfg = ControlFlowGraph::compute(f)?;
    pipeline::optimize(f, Analyses::new(Held::Owned(cfg), None, arch))
}

/// What a rule does with an instruction.
enum Rewrite {
    /// Keep the instruction with this operation.
    Op(Op),
    /// Remove the instruction; its result is this value.
    Alias(Value),
    /// Remove an instruction without a result.
    Remove,
}

/// The local rewrite rules of `propagate`, applied in order to each
/// instruction. `aliases` holds the values already replaced in the block
/// order of the walk.
struct Rules<'a> {
    known: &'a dyn Fn(Value) -> Option<u128>,
    literals: &'a HashMap<(Lane, u128), Value>,
    /// Integer vector orderings `(kind, x, y)` by their mask.
    orderings: HashMap<Value, (Binary, Value, Value)>,
}

impl Rules<'_> {
    fn rewrite(
        &mut self,
        op: Op,
        result: Option<Value>,
        types: &DenseMap<Value, Type>,
        aliases: &DenseMap<Value, Value>,
    ) -> Rewrite {
        let op = self.min_max(op, result, types, aliases);
        let Some(dst) = result else {
            return match op {
                Op::TrapIf(cond, trap) => match (self.known)(cond) {
                    Some(0) => Rewrite::Remove,
                    Some(_) => Rewrite::Op(Op::Trap(trap)),
                    None => Rewrite::Op(op),
                },
                _ => Rewrite::Op(op),
            };
        };
        if let Op::Copy(v) = op {
            return Rewrite::Alias(v);
        }
        if let Some(bits) = (self.known)(dst)
            && !matches!(op, Op::Simd(Simd::Splat(_)))
            && let Some(literal) = constant_op(types[dst], bits, self.literals)
        {
            return Rewrite::Op(literal);
        }
        self.binary(op, dst, types, aliases)
            .or_else(|| self.select(op, types, aliases))
            .unwrap_or(Rewrite::Op(op))
    }

    /// `select(x < y, x, y)` on integer vectors is `min(x, y)`, and the
    /// other orderings and arm orders likewise.
    fn min_max(
        &mut self,
        op: Op,
        result: Option<Value>,
        types: &DenseMap<Value, Type>,
        aliases: &DenseMap<Value, Value>,
    ) -> Op {
        match (op, result) {
            (
                Op::Bin(kind @ (Binary::Lt | Binary::Le | Binary::Gt | Binary::Ge), x, y),
                Some(dst),
            ) if matches!(types[x], Type::Vector(lane) if !lane.float()) => {
                self.orderings.insert(dst, (kind, aliases[x], aliases[y]));
                op
            }
            (Op::Ternary(Ternary::Select, mask, yes, no), _) => {
                let Some(&(kind, x, y)) = self.orderings.get(&aliases[mask]) else {
                    return op;
                };
                let left_is_min = matches!(kind, Binary::Lt | Binary::Le);
                let (yes, no) = (aliases[yes], aliases[no]);
                let select_min = if (yes, no) == (x, y) {
                    left_is_min
                } else if (yes, no) == (y, x) {
                    !left_is_min
                } else {
                    return op;
                };
                let kind = if select_min { Binary::Min } else { Binary::Max };
                Op::Bin(kind, yes, no)
            }
            _ => op,
        }
    }

    /// Integer identities and constants (`sccp::reduce`), and a scalar
    /// constant moved to the right operand.
    fn binary(
        &self,
        op: Op,
        dst: Value,
        types: &DenseMap<Value, Type>,
        aliases: &DenseMap<Value, Value>,
    ) -> Option<Rewrite> {
        let Op::Bin(kind, x, y) = op else {
            return None;
        };
        if types[x].lane().is_none_or(Lane::float) {
            return None;
        }
        let (a, b) = ((self.known)(x), (self.known)(y));
        match sccp::reduce(kind, types[x], aliases[x], aliases[y], a, b) {
            Some(sccp::Reduced::Copy(v)) => Some(Rewrite::Alias(v)),
            Some(sccp::Reduced::Constant(bits)) => Some(Rewrite::Op(
                constant_op(types[dst], bits, self.literals).unwrap_or(op),
            )),
            None if !matches!(types[x], Type::Vector(_)) && a.is_some() && b.is_none() => {
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
                    _ => return Some(Rewrite::Op(op)),
                };
                Some(Rewrite::Op(Op::Bin(flipped, y, x)))
            }
            None => None,
        }
    }

    /// A select with equal arms or a known condition is one of its arms.
    fn select(
        &self,
        op: Op,
        types: &DenseMap<Value, Type>,
        aliases: &DenseMap<Value, Value>,
    ) -> Option<Rewrite> {
        let Op::Ternary(Ternary::Select, c, yes, no) = op else {
            return None;
        };
        if aliases[yes] == aliases[no] {
            return Some(Rewrite::Alias(yes));
        }
        let bits = (self.known)(c)?;
        Some(Rewrite::Alias(if sccp::select_truth(types[c], bits) {
            yes
        } else {
            no
        }))
    }
}

/// Sparse conditional constant propagation, the local rewrite rules,
/// constant branches and local if-conversion.
pub(super) fn propagate(
    f: &mut Function,
    analyses: &mut Analyses<'_>,
) -> Result<Preserved, IrError> {
    let cfg = analyses.cfg(f)?;
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
    let mut rules = Rules {
        known: &known,
        literals: &literals,
        orderings: HashMap::new(),
    };
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
            match rules.rewrite(op, result, types, &aliases) {
                Rewrite::Op(op) => Some(op),
                Rewrite::Remove => None,
                Rewrite::Alias(v) => {
                    aliases[result.unwrap()] = aliases[v];
                    changed_aliases = true;
                    None
                }
            }
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
    changed_cfg |= local_cfg::thread_constant_branches(f, cfg);
    changed_cfg |= local_cfg::convert_branches(f, cfg);
    Ok(if changed_cfg {
        f.prune_unreachable()?;
        Preserved::Nothing
    } else {
        Preserved::Cfg
    })
}

/// Optimize every body of the module, callees first, and rebind the calls
/// of each caller to its optimized callees.
pub(crate) fn simplify_bodies(
    module: &Module<VerifiedBody<'_>>,
    arch: Arch,
) -> Result<Function, IrError> {
    let mut bodies: DenseMap<FunctionId, Arc<Function>> =
        DenseMap::with_capacity(module.functions.len());
    for (id, source) in module.functions.iter() {
        let mut body = source.function.clone();
        for (site, target) in body.calls_mut() {
            let callee = source.callee(site);
            if callee != FunctionId::INVALID {
                *target = Arc::clone(bodies.get(callee).expect("callees are numbered first"));
            }
        }
        pipeline::optimize(
            &mut body,
            Analyses::new(Held::Borrowed(&source.cfg), Some(&source.dom), arch),
        )?;
        if id == module.call_graph.root() {
            return Ok(body);
        }
        debug_assert_eq!(bodies.len(), id.index());
        bodies.insert(Arc::new(body));
    }
    unreachable!("module has a root")
}
