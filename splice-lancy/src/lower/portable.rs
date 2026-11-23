use super::{Builder, Destination, KernelBuilder};
use lancy::hir::{self, Op};
use splice_core::{Function, hir::Instr};
use std::{collections::HashMap, sync::Arc};

/// Lower a verified portable function and its callees using the same SSA
/// construction as direct Splice staging. Preserve shared callee identities.
/// Raw invocation of the result must uphold Splice's slice contract:
/// immutable buffers may overlap and writable buffers have exclusive access.
pub fn lower(root: &Function, target: &splice_core::Target) -> splice_core::Result<hir::Function> {
    lower_mode(root, target.vector_bytes(), None)
}

pub(crate) type DebugSources<'a> = Vec<(&'a Function, Option<&'a splice_core::debug::Point>)>;

pub(crate) fn lower_debug<'a>(
    root: &'a Function,
    target: &splice_core::Target,
    sources: &mut DebugSources<'a>,
) -> splice_core::Result<hir::Function> {
    lower_mode(root, target.vector_bytes(), Some(sources))
}

fn lower_mode<'a>(
    root: &'a Function,
    vector_bytes: usize,
    mut sources: Option<&mut DebugSources<'a>>,
) -> splice_core::Result<hir::Function> {
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
        let graph = lower_body(f, &callees, vector_bytes, sources.as_deref_mut())?;
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
    vector_bytes: usize,
    mut sources: Option<&mut DebugSources<'a>>,
) -> splice_core::Result<hir::Function> {
    let mut b = Builder::new(function.arity(), function.buffer_kinds(), vector_bytes)?;
    b.types = (0..function.register_count())
        .map(|i| function.reg_type(splice_core::hir::Reg(i as u32)))
        .collect();
    b.labels
        .resize_with(function.label_count(), Destination::default);
    if let Some(sources) = &mut sources {
        let id = u32::try_from(sources.len())
            .map_err(|_| splice_core::BuildError::unsupported("too many debug points"))?;
        b.graph.append(b.graph.entry(), Op::DebugPoint(id));
        sources.push((function, None));
    }
    for inst in function.instructions() {
        match inst {
            Instr::Debug(point) => {
                let (Some(block), Some(sources)) = (b.current, &mut sources) else {
                    continue;
                };
                for (var, ty, offset) in point.slots(function, vector_bytes) {
                    let offset = u32::try_from(offset).map_err(|_| {
                        splice_core::BuildError::unsupported("debug frame too large")
                    })?;
                    match var.value {
                        splice_core::debug::Value::Register(reg, _) => {
                            let mut value = b.observe(reg);
                            match ty {
                                splice_core::debug::Type::Scalar(
                                    splice_core::debug::Scalar::Index,
                                ) => {
                                    value = b
                                        .graph
                                        .append(block, Op::Cast(hir::Type::I64, value))
                                        .unwrap();
                                }
                                // A DWARF bool is one byte holding 0 or 1.
                                splice_core::debug::Type::Scalar(
                                    splice_core::debug::Scalar::Bool,
                                ) => {
                                    let one = b.graph.append(block, Op::Const(1)).unwrap();
                                    value = b
                                        .graph
                                        .append(block, Op::Bin(hir::Binary::And, value, one))
                                        .unwrap();
                                }
                                _ => {}
                            }
                            b.graph.append(block, Op::DebugStore(offset, value));
                        }
                        splice_core::debug::Value::Slice(buffer) => {
                            let (data, len) = b.slice(buffer, false);
                            let length_offset = offset.checked_add(8).ok_or_else(|| {
                                splice_core::BuildError::unsupported("debug frame too large")
                            })?;
                            b.graph.append(block, Op::DebugStore(offset, data));
                            let len = b
                                .graph
                                .append(block, Op::Cast(hir::Type::I64, len))
                                .unwrap();
                            b.graph.append(block, Op::DebugStore(length_offset, len));
                        }
                        splice_core::debug::Value::Constant(_) => {
                            unreachable!("only runtime values have slots")
                        }
                    }
                }
                let id = u32::try_from(sources.len())
                    .map_err(|_| splice_core::BuildError::unsupported("too many debug points"))?;
                b.graph.append(block, Op::DebugPoint(id));
                sources.push((function, Some(point)));
            }
            Instr::Const(dst, n) => b.define(*dst, Op::Const(*n)),
            Instr::Simd { dst, inst, args } => {
                let v = b.emit_simd(*inst, args)?;
                b.bind_simd(*dst, v);
            }
            Instr::Bin(dst, op, lhs, rhs) => {
                b.arithmetic(*dst, *op, *lhs, *rhs);
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
            Instr::RustCall { dst, import, args } => b.define_rust_call(*dst, *import, args)?,
            Instr::RustPacketCall { dst, import, args } => {
                b.define_packet_call(*dst, import, args)?
            }
            Instr::RustContextCall {
                dst,
                target,
                args,
                slices,
            } => {
                b.define_context_call(*dst, *target, args, slices)?;
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
            Instr::TrapBounds => b.trap_bounds()?,
        }
    }
    b.seal()
}
