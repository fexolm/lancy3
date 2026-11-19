use super::{Builder, Destination, KernelBuilder, binary};
use lancy::hir::{self, Op};
use splice::{Function, hir::Instr};
use std::{collections::HashMap, sync::Arc};

/// Lower a verified portable function and its callees using the same SSA
/// construction as direct Splice staging. Preserve shared callee identities.
pub fn lower(root: &Function) -> Result<hir::Function, String> {
    splice_target::simd()?;
    let mut callees = HashMap::new();
    let mut pending = vec![(root, false)];
    while let Some((f, visited)) = pending.pop() {
        let key = std::ptr::from_ref(f);
        if callees.contains_key(&key) {
            continue;
        }
        if !visited {
            pending.push((f, true));
            for inst in f.instructions() {
                if let Instr::Call { function, .. } = inst {
                    pending.push((function, false));
                }
            }
            continue;
        }
        let graph = lower_body(f, &callees)?;
        if std::ptr::eq(f, root) {
            return Ok(graph);
        }
        callees.insert(key, Arc::new(graph));
    }
    unreachable!("the root is always scheduled")
}

fn lower_body(
    function: &Function,
    callees: &HashMap<*const Function, Arc<hir::Function>>,
) -> Result<hir::Function, String> {
    let mut b = Builder::new(function.arity(), function.buffer_kinds())?;
    b.types = (0..function.register_count())
        .map(|i| function.reg_type(splice::hir::Reg(i as u32)))
        .collect();
    b.labels
        .resize_with(function.label_count(), Destination::default);
    for inst in function.instructions() {
        match inst {
            Instr::Const(dst, n) => b.define(*dst, Op::Const(*n)),
            Instr::Simd { dst, inst, args } => {
                let v = b.emit_simd(*inst, args)?;
                b.bind_simd(*dst, v);
            }
            Instr::Bin(dst, op, lhs, rhs) => {
                let op = Op::Bin(binary(*op), b.read(*lhs), b.read(*rhs));
                b.define(*dst, op);
            }
            Instr::BufferLen { dst, buffer } => {
                let (_, len) = b.slice(*buffer, false);
                b.define(*dst, Op::Copy(len));
            }
            Instr::Load { dst, buffer, index } => {
                let index = b.read(*index);
                let addr = b.address(*buffer, index, false);
                b.define(*dst, Op::Load(hir::Type::I32, addr));
            }
            Instr::Call {
                dst,
                function,
                args,
                buffers,
            } => {
                splice::hir::check_forwarding(b.kinds, function.buffer_kinds(), buffers)?;
                let op = b.call_op(callees[&Arc::as_ptr(function)].clone(), args, buffers);
                b.define(*dst, op);
            }
            Instr::Copy(dst, src) => b.assign(*dst, *src),
            Instr::Store {
                buffer,
                index,
                value,
            } => b.store(*buffer, *index, *value),
            Instr::Label(label) => b.mark(*label),
            Instr::Jump(label) => b.jump(*label),
            Instr::Branch { cond, yes, no } => b.branch(*cond, *yes, *no),
            Instr::Return(value) => b.ret(*value),
        }
    }
    b.seal()
}
