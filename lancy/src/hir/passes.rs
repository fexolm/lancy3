//! Shared transformations for every function, including ordinary loops over slices.
use super::{Function, Op};
use std::{collections::HashMap, sync::Arc};

/// Optimize a call DAG once per distinct body, retaining shared callees.
/// Invalid or unfinished public graphs are rejected before inspecting operands.
pub fn optimize(root: &Function) -> Result<Function, String> {
    let mut done = HashMap::<*const Function, Arc<Function>>::new();
    let mut work = vec![(root, false)];
    while let Some((input, ready)) = work.pop() {
        let key = std::ptr::from_ref(input);
        if done.contains_key(&key) {
            continue;
        }
        if !ready {
            input.verify()?;
            work.push((input, true));
            for block in input.blocks() {
                for inst in input.instructions(block) {
                    if let Op::Call(id) = *input.op(inst) {
                        work.push((input.call(id).0.as_ref(), false));
                    }
                }
            }
            continue;
        }
        let mut f = input.clone();
        for block in input.blocks() {
            for inst in input.instructions(block) {
                if let Op::Call(id) = *input.op(inst) {
                    let callee = &done[&Arc::as_ptr(input.call(id).0)];
                    f.set_callee(id, Arc::clone(callee));
                }
            }
        }
        f.merge_verified_linear_blocks();
        fold_constants(&mut f);
        if std::ptr::eq(input, root) {
            return Ok(f);
        }
        done.insert(key, Arc::new(f));
    }
    unreachable!("call traversal always includes the root")
}

fn fold_constants(f: &mut Function) {
    let mut known = vec![None; f.value_count()];
    let mut replacements = Vec::new();
    for block in f.blocks() {
        for inst in f.instructions(block) {
            let literal = match *f.op(inst) {
                Op::Const(n) => Some(n),
                Op::Copy(source) => known[source.index()],
                Op::Bin(op, a, b) => known[a.index()]
                    .zip(known[b.index()])
                    .and_then(|(a, b)| op.checked_eval(a, b)),
                _ => None,
            };
            if let (Some(n), Some(result)) = (literal, f.result(inst)) {
                known[result.index()] = Some(n);
                if !matches!(f.op(inst), Op::Const(_)) {
                    replacements.push((inst, Op::Const(n)));
                }
            }
        }
    }
    for (inst, op) in replacements {
        f.set_op_unchecked(inst, op);
    }
}
