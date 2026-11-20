use crate::hir::transforms::{ControlFlowGraph, DomTree};
use crate::hir::{Binary, Block, Function, Op, Terminator, Ternary, Type, Value, ssa::Edge};
use crate::support::slotmap::DenseMap;

/// Turn a single i32 merge into a select when neither arm computes anything.
/// Bypassing a forwarding arm is legal even if other branches still use it.
pub(super) fn convert_branches(f: &mut Function, cfg: &ControlFlowGraph) -> bool {
    fn endpoint(f: &Function, edge: Edge, bypass: bool) -> (Edge, &[Value], &[Value]) {
        let (block, args) = f.edge(edge);
        if bypass
            && f.instructions(block).next().is_none()
            && let Terminator::Jump(next) = *f.terminator(block)
        {
            return (next, f.params(block), args);
        }
        (edge, &[], &[])
    }

    fn arguments<'a>(
        args: &'a [Value],
        params: &'a [Value],
        incoming: &'a [Value],
    ) -> impl Iterator<Item = Value> + 'a {
        // Parameters are appended in value-ID order.
        args.iter().map(|&arg| match params.binary_search(&arg) {
            Ok(i) => incoming[i],
            Err(_) => arg,
        })
    }

    let mut changed = false;
    for &block in cfg.reverse_postorder() {
        let Terminator::Branch { cond, yes, no } = *f.terminator(block) else {
            continue;
        };
        let bypass = f.edge(yes).0 != f.edge(no).0;
        let (left, lp, la) = endpoint(f, yes, bypass);
        let (right, rp, ra) = endpoint(f, no, bypass);
        let (target, a) = f.edge(left);
        let (other, b) = f.edge(right);
        if target != other || target == block {
            continue;
        }
        let left_args = || arguments(a, lp, la);
        let right_args = arguments(b, rp, ra);
        let (first, second) = {
            let mut differing = left_args()
                .zip(right_args)
                .enumerate()
                .filter(|(_, (x, y))| x != y);
            (differing.next(), differing.next())
        };
        let Some((index, (x, y))) = first else {
            if lp.is_empty() {
                f.set_terminator(block, Terminator::Jump(left));
            } else {
                let args: Vec<_> = left_args().collect();
                f.jump(block, target, &args);
            }
            changed = true;
            continue;
        };
        if f.value_type(x) != Type::I32 || second.is_some() {
            continue;
        }
        let mut args: Vec<_> = left_args().collect();
        args[index] = f
            .append(block, Op::Ternary(Ternary::Select, cond, x, y))
            .unwrap();
        f.jump(block, target, &args);
        changed = true;
    }
    changed
}

pub(super) fn forward_params(f: &mut Function, cfg: &ControlFlowGraph) {
    let mut inputs: Option<DenseMap<Value, Option<Value>>> = None;
    for &block in cfg.reverse_postorder() {
        for edge in f.terminator(block).edges() {
            let (target, args) = f.edge(edge);
            if args.is_empty() {
                continue;
            }
            let inputs = inputs.get_or_insert_with(|| DenseMap::filled(f.value_count(), None));
            for (&param, &arg) in f.params(target).iter().zip(args) {
                if arg != param {
                    let prior = inputs[param].get_or_insert(arg);
                    if *prior != arg {
                        *prior = Value::INVALID;
                    }
                }
            }
        }
    }
    let Some(inputs) = inputs else {
        return;
    };
    let mut aliases = DenseMap::from_fn(f.value_count(), |v| v);
    let mut changed = false;
    for &block in &cfg.reverse_postorder()[1..] {
        for &p in f.params(block) {
            if let Some(v) = inputs[p].filter(|&v| v != Value::INVALID) {
                aliases[p] = v;
                changed = true;
            }
        }
    }
    if changed {
        // An incoming value common to all non-self edges is available on
        // entry to the block. Compress chains using path halving.
        for v in aliases.keys() {
            let mut root = v;
            while aliases[root] != root {
                let parent = aliases[root];
                let grandparent = aliases[parent];
                aliases[root] = grandparent;
                root = grandparent;
            }
            aliases[v] = root;
        }
        f.remove_aliased_params(&aliases);
        f.substitute_values(&aliases);
    }
}

/// Hoist total i32 arithmetic in natural loops with an existing preheader
/// and a straight-line body. Reject internal branches and side entries
/// rather than building a loop forest or cloning CFG.
pub(super) fn hoist_loop_invariants(f: &mut Function, cfg: &ControlFlowGraph, dom: &DomTree) {
    let mut defs = None;
    let mut body = Vec::new();
    for &header in cfg.reverse_postorder() {
        let &[a, b] = cfg.predecessors(header) else {
            continue;
        };
        let (preheader, latch) = if dom.dominates(header, b) {
            (a, b)
        } else {
            (b, a)
        };
        if dom.dominates(header, preheader)
            || !dom.dominates(header, latch)
            || !matches!(*f.terminator(preheader), Terminator::Jump(_))
            || (latch != header && !matches!(*f.terminator(latch), Terminator::Jump(_)))
        {
            continue;
        }
        body.clear();
        let mut cursor = latch;
        while cursor != header {
            let &[pred] = cfg.predecessors(cursor) else {
                break;
            };
            if pred != header && !matches!(*f.terminator(pred), Terminator::Jump(_)) {
                break;
            }
            body.push(cursor);
            cursor = pred;
        }
        if cursor != header {
            continue;
        }
        body.push(header);
        let defs = defs.get_or_insert_with(|| {
            let mut defs = DenseMap::filled(f.value_count(), Block::INVALID);
            for block in f.blocks() {
                for &v in f.params(block) {
                    defs[v] = block;
                }
                for i in f.instructions(block) {
                    if let Some(v) = f.result(i) {
                        defs[v] = block;
                    }
                }
            }
            defs
        });
        for &block in body.iter().rev() {
            f.hoist_instructions(block, preheader, |op, result, types| {
                let Op::Bin(kind, a, b) = op else {
                    return false;
                };
                if types[a] == Type::I32
                    && !matches!(kind, Binary::Div | Binary::Rem)
                    && dom.dominates(defs[a], preheader)
                    && dom.dominates(defs[b], preheader)
                {
                    defs[result] = preheader;
                    return true;
                }
                false
            });
        }
    }
}
