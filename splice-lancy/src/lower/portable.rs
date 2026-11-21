use super::{Builder, Destination, KernelBuilder, binary};
use lancy::hir::{self, Op};
use splice::{Function, hir::Instr};
use std::{collections::HashMap, sync::Arc};

/// Lower a verified portable function and its callees using the same SSA
/// construction as direct Splice staging. Preserve shared callee identities.
/// Raw invocation of the result must uphold Splice's slice contract:
/// immutable buffers may overlap and writable buffers have exclusive access.
pub fn lower(root: &Function) -> Result<hir::Function, String> {
    lower_mode(root, None)
}

pub(crate) type DebugSources<'a> = Vec<(&'a Function, Option<&'a splice::debug::Point>)>;

pub(crate) fn lower_debug<'a>(
    root: &'a Function,
    sources: &mut DebugSources<'a>,
) -> Result<hir::Function, String> {
    lower_mode(root, Some(sources))
}

fn lower_mode<'a>(
    root: &'a Function,
    mut sources: Option<&mut DebugSources<'a>>,
) -> Result<hir::Function, String> {
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
        let graph = lower_body(f, &callees, sources.as_deref_mut())?;
        if std::ptr::eq(f, root) {
            return Ok(graph);
        }
        callees.insert(key, Arc::new(graph));
    }
    unreachable!("the root is always scheduled")
}

fn lower_body<'a>(
    function: &'a Function,
    callees: &HashMap<*const Function, Arc<hir::Function>>,
    mut sources: Option<&mut DebugSources<'a>>,
) -> Result<hir::Function, String> {
    let mut b = Builder::new(function.arity(), function.buffer_kinds())?;
    b.types = (0..function.register_count())
        .map(|i| function.reg_type(splice::hir::Reg(i as u32)))
        .collect();
    b.labels
        .resize_with(function.label_count(), Destination::default);
    if let Some(sources) = &mut sources {
        let id = u32::try_from(sources.len()).map_err(|_| "too many debug points")?;
        b.graph.append(b.graph.entry(), Op::DebugPoint(id));
        sources.push((function, None));
    }
    for inst in function.instructions() {
        match inst {
            Instr::Debug(point) => {
                let (Some(block), Some(sources)) = (b.current, &mut sources) else {
                    continue;
                };
                for (var, ty, offset) in point.slots(function) {
                    let offset = u32::try_from(offset).map_err(|_| "debug frame too large")?;
                    match var.value {
                        splice::debug::Value::Register(reg, _) => {
                            let mut value = b.read(reg);
                            if matches!(
                                ty,
                                splice::debug::Type::Scalar(splice::debug::Scalar::Index)
                            ) {
                                value = b
                                    .graph
                                    .append(block, Op::Cast(hir::Type::I64, value))
                                    .unwrap();
                            }
                            b.graph.append(block, Op::DebugStore(offset, value));
                        }
                        splice::debug::Value::Slice(buffer) => {
                            let (data, len) = b.slice(buffer, false);
                            let length_offset =
                                offset.checked_add(8).ok_or("debug frame too large")?;
                            b.graph.append(block, Op::DebugStore(offset, data));
                            let len = b
                                .graph
                                .append(block, Op::Cast(hir::Type::I64, len))
                                .unwrap();
                            b.graph.append(block, Op::DebugStore(length_offset, len));
                        }
                        splice::debug::Value::Constant(_) => {
                            unreachable!("only runtime values have slots")
                        }
                    }
                }
                let id = u32::try_from(sources.len()).map_err(|_| "too many debug points")?;
                b.graph.append(block, Op::DebugPoint(id));
                sources.push((function, Some(point)));
            }
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
                b.define_load(*dst, *buffer, index);
            }
            Instr::Call {
                dst,
                function,
                args,
                buffers,
            } => {
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
