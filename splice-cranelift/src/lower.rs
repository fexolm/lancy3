use crate::{BOUNDS, DIV_ZERO, Descriptor, INTEGER_OVERFLOW, OVERFLOW};
use cranelift_codegen::ir::{
    DebugTag, FuncRef, InstBuilder, MemFlags, Opcode, StackSlot, StackSlotData, StackSlotKind,
    TrapCode, Value,
    condcodes::IntCC,
    types::{I8, I32, I64},
};
use cranelift_frontend::{FunctionBuilder, Variable};
use splice::hir::{Binary, ContextSlice, Instr, Reg, RustContextTarget};
use splice::{ContextArguments, Function};
use std::collections::HashMap;
use std::mem::{offset_of, size_of};
use std::sync::Arc;

mod simd;

pub(crate) use simd::keep_compared_loads_in_registers;

fn packet_stack(
    builder: &mut FunctionBuilder,
    scratch: &mut Option<StackSlot>,
    bytes: u32,
) -> StackSlot {
    let slot = *scratch.get_or_insert_with(|| {
        builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, bytes, 3))
    });
    let slot_data = &mut builder.func.sized_stack_slots[slot];
    slot_data.size = slot_data.size.max(bytes);
    slot
}

fn packet_words(
    builder: &mut FunctionBuilder,
    scratch: StackSlot,
    args: impl Iterator<Item = Variable>,
) {
    for (index, arg) in args.enumerate() {
        let word = builder.use_var(arg);
        builder.ins().stack_store(word, scratch, index as i32 * 4);
    }
}

fn import_result(builder: &mut FunctionBuilder, packed: Value) -> Value {
    let failed = builder
        .ins()
        .icmp_imm(IntCC::UnsignedGreaterThan, packed, i64::from(u32::MAX));
    let panic = builder.ins().iconst(I64, crate::RUST_PANIC);
    return_if(builder, failed, panic);
    builder.ins().ireduce(I32, packed)
}

fn parameters(b: &mut FunctionBuilder, f: &Function, is_entry: bool) -> Vec<Value> {
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);
    if !is_entry {
        return b.block_params(entry).to_vec();
    }
    let scalars = b.block_params(entry)[0];
    let descriptors = b.block_params(entry)[1];
    let mut params = Vec::new();
    // Executable::call checks both array lengths and constructs repr(C) descriptors.
    let flags = MemFlags::trusted();
    for i in 0..f.arity() {
        let addr = b.ins().iadd_imm(scalars, (i * size_of::<i32>()) as i64);
        params.push(b.ins().load(I32, flags, addr, 0));
    }
    for i in 0..f.buffer_kinds().len() {
        let addr = b
            .ins()
            .iadd_imm(descriptors, (i * size_of::<Descriptor>()) as i64);
        params.push(
            b.ins()
                .load(I64, flags, addr, offset_of!(Descriptor, data) as i32),
        );
        params.push(
            b.ins()
                .load(I32, flags, addr, offset_of!(Descriptor, len) as i32),
        );
    }
    params
}

pub(super) fn emit(
    mut b: FunctionBuilder,
    f: &Function,
    is_entry: bool,
    imports: &HashMap<*const Function, FuncRef>,
    debug_bytes: Option<u32>,
) {
    let params = parameters(&mut b, f, is_entry);
    let (scalars, pairs) = params.split_at(f.arity());
    let buffers = pairs.as_chunks::<2>().0;
    let vars = simd::Registers::new(&mut b, f);
    let var = |r: Reg| vars.get(r)[0];
    for (index, &value) in scalars.iter().enumerate() {
        b.def_var(var(Reg(index as u32)), value);
    }
    let labels: Vec<_> = (0..f.label_count()).map(|_| b.create_block()).collect();
    let snapshots = debug_bytes.map(|bytes| {
        b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, bytes, 3))
    });
    let mut rust_scratch = None;
    let mut open = true;
    for (index, op) in f.instructions().iter().enumerate() {
        match *op {
            Instr::Label(label) => {
                let block = labels[label.0 as usize];
                if open {
                    b.ins().jump(block, &[]);
                }
                b.switch_to_block(block);
                open = true;
            }
            _ if !open => continue,
            Instr::Debug(ref point) => {
                let Some(slot) = snapshots else {
                    continue;
                };
                use splice::debug::Value as DebugValue;
                let mut slots = point.slots(f).peekable();
                if slots.peek().is_none() {
                    // Sequence points themselves occupy no bytes in Cranelift.
                    // Give consecutive empty statements distinct addresses.
                    let zero = b.ins().iconst(I8, 0);
                    b.ins().stack_store(zero, slot, 0);
                }
                for (var, ty, offset) in slots {
                    let offset = offset as i32;
                    match var.value {
                        DebugValue::Register(reg, _) => {
                            for (part, &variable) in vars.get(reg).iter().enumerate() {
                                let mut value = b.use_var(variable);
                                if matches!(
                                    ty,
                                    splice::debug::Type::Scalar(splice::debug::Scalar::Index)
                                ) {
                                    value = b.ins().sextend(I64, value);
                                }
                                b.ins()
                                    .stack_store(value, slot, offset + (part * 16) as i32);
                            }
                        }
                        DebugValue::Slice(buffer) => {
                            let [data, len] = buffers[buffer.index()];
                            b.ins().stack_store(data, slot, offset);
                            let len = b.ins().uextend(I64, len);
                            b.ins().stack_store(len, slot, offset + 8);
                        }
                        DebugValue::Constant(_) => unreachable!("only runtime values have slots"),
                    }
                }
                let at = b.ins().sequence_point();
                b.func.debug_tags.set(
                    at,
                    [DebugTag::User(index as u32), DebugTag::StackSlot(slot)],
                );
            }
            Instr::Const(dst, n) => {
                let value = b.ins().iconst(I32, i64::from(n));
                b.def_var(var(dst), value);
            }
            Instr::Copy(dst, src) => {
                for (&d, &s) in vars.get(dst).iter().zip(vars.get(src)) {
                    let value = b.use_var(s);
                    b.def_var(d, value);
                }
            }
            Instr::Simd {
                dst,
                inst,
                ref args,
            } => {
                simd::emit(&mut b, inst, &vars, dst, args, buffers);
            }
            Instr::Bin(dst, op, left, right) => {
                let lhs = b.use_var(var(left));
                let rhs = b.use_var(var(right));
                let value = binary(&mut b, op, lhs, rhs);
                b.def_var(var(dst), value);
            }
            Instr::BufferLen { dst, buffer } => b.def_var(var(dst), buffers[buffer.index()][1]),
            Instr::Load { dst, buffer, index } => {
                let index = b.use_var(var(index));
                let addr = element(&mut b, buffers[buffer.index()], index);
                let value = b.ins().load(I32, MemFlags::new(), addr, 0);
                b.def_var(var(dst), value);
            }
            Instr::Store {
                buffer,
                index,
                value,
            } => {
                let index = b.use_var(var(index));
                let value = b.use_var(var(value));
                let addr = element(&mut b, buffers[buffer.index()], index);
                b.ins().store(MemFlags::new(), value, addr, 0);
            }
            Instr::RustCall {
                dst,
                import,
                ref args,
            } => {
                use cranelift_codegen::ir::{AbiParam, Signature};
                // KernelModule uses the target ISA's system calling convention.
                // The signature here matches ScalarEntry on Windows and Unix.
                let mut signature = Signature::new(b.func.signature.call_conv);
                signature.params.extend([AbiParam::new(I32); 2]);
                signature.returns.push(AbiParam::new(I64));
                let sig = b.import_signature(signature);
                let zero = b.ins().iconst(I32, 0);
                let mut words = [zero; 2];
                for (out, &arg) in words.iter_mut().zip(args) {
                    *out = b.use_var(var(arg));
                }
                let address = b
                    .ins()
                    .iconst(I64, (import.symbol.resolve)() as usize as i64);
                let call = b.ins().call_indirect(sig, address, &words);
                let packed = b.inst_results(call)[0];
                let value = import_result(&mut b, packed);
                b.def_var(var(dst), value);
            }
            Instr::RustPacketCall {
                dst,
                import,
                ref args,
            } => {
                use cranelift_codegen::ir::{AbiParam, Signature};
                let mut signature = Signature::new(b.func.signature.call_conv);
                signature.params.extend([
                    AbiParam::new(I64),
                    AbiParam::new(I64),
                    AbiParam::new(I32),
                ]);
                signature.returns.push(AbiParam::new(I64));
                let sig = b.import_signature(signature);
                let scratch = packet_stack(
                    &mut b,
                    &mut rust_scratch,
                    import.packet_bytes().expect("verified Rust packet"),
                );
                packet_words(&mut b, scratch, args.iter().copied().map(var));
                let symbol = b.ins().iconst(I64, std::ptr::from_ref(import) as i64);
                let words = b.ins().stack_addr(I64, scratch, 0);
                let count = b.ins().iconst(I32, i64::from(import.arity));
                let address = b.ins().iconst(
                    I64,
                    splice_target::abi::invoke_packet_import as *const () as usize as i64,
                );
                let call = b.ins().call_indirect(sig, address, &[symbol, words, count]);
                let packed = b.inst_results(call)[0];
                let value = import_result(&mut b, packed);
                b.def_var(var(dst), value);
            }
            Instr::RustContextCall {
                dst,
                target,
                ref args,
                ref slices,
            } => {
                use cranelift_codegen::ir::{AbiParam, Signature};
                let (data, operation) = match target {
                    RustContextTarget::Borrowed { context, operation } => {
                        (Some(buffers[context.index()][0]), operation)
                    }
                    RustContextTarget::Static(_) => (None, 0),
                };
                let ptr = I64;
                let mut signature = Signature::new(b.func.signature.call_conv);
                signature
                    .params
                    .extend([AbiParam::new(ptr), AbiParam::new(I32)]);
                let zero = b.ins().iconst(I32, 0);
                let operation = b.ins().iconst(I32, i64::from(operation));
                let (entry, words) = if let Some(data) = data
                    && args.len() <= 2
                    && slices.is_empty()
                {
                    let mut words = [data, operation, zero, zero];
                    for (out, &arg) in words[2..].iter_mut().zip(args) {
                        *out = b.use_var(var(arg));
                    }
                    signature.params.extend([AbiParam::new(I32); 2]);
                    (splice::invoke_runtime_context as *const (), words)
                } else {
                    signature
                        .params
                        .extend([AbiParam::new(ptr), AbiParam::new(I32)]);
                    let layout =
                        ContextArguments::packet_layout(args.len() as u32, slices.len() as u32)
                            .expect("validated packet size");
                    let scratch = packet_stack(&mut b, &mut rust_scratch, layout.bytes.max(8));
                    packet_words(&mut b, scratch, args.iter().copied().map(var));
                    for (i, &ContextSlice { buffer, writable }) in slices.iter().enumerate() {
                        let offset =
                            (layout.slices_at + i as u32 * ContextArguments::SLICE_BYTES) as i32;
                        let [data, len] = buffers[buffer.index()];
                        let element = f.buffer_kinds()[buffer.index()].element();
                        let tag = b.ins().iconst(
                            I32,
                            i64::from(ContextArguments::view_tag(element, writable)),
                        );
                        b.ins().stack_store(data, scratch, offset);
                        b.ins().stack_store(
                            len,
                            scratch,
                            offset + ContextArguments::LENGTH_OFFSET as i32,
                        );
                        b.ins().stack_store(
                            tag,
                            scratch,
                            offset + ContextArguments::ELEMENT_OFFSET as i32,
                        );
                    }
                    let packet = b.ins().stack_addr(ptr, scratch, 0);
                    let entry = match target {
                        RustContextTarget::Borrowed { .. } => {
                            splice::invoke_packet_runtime_context as *const ()
                        }
                        RustContextTarget::Static(symbol) => symbol.entry() as *const (),
                    };
                    (entry, [data.unwrap_or(packet), operation, packet, zero])
                };
                signature.returns.push(AbiParam::new(I64));
                let sig = b.import_signature(signature);
                let address = b.ins().iconst(ptr, entry as usize as i64);
                let call = b.ins().call_indirect(sig, address, &words);
                let packed = b.inst_results(call)[0];
                let value = import_result(&mut b, packed);
                b.def_var(var(dst), value);
            }
            Instr::Call {
                dst,
                ref function,
                ref args,
                buffers: ref forwarded,
            } => {
                let mut actuals: Vec<_> = args.iter().map(|&r| b.use_var(var(r))).collect();
                for slice in forwarded {
                    actuals.extend(buffers[slice.index()]);
                }
                let call = b.ins().call(imports[&Arc::as_ptr(function)], &actuals);
                let result = b.inst_results(call)[0];
                if function.needs_status() {
                    let failed = b.ins().icmp_imm(IntCC::SignedLessThan, result, 0);
                    return_if(&mut b, failed, result);
                }
                let value = b.ins().ireduce(I32, result);
                b.def_var(var(dst), value);
            }
            Instr::Jump(label) => {
                b.ins().jump(labels[label.0 as usize], &[]);
                open = false;
            }
            Instr::Branch { cond, yes, no } => {
                let condition = b.use_var(var(cond));
                let condition = branch_condition(&b, condition);
                b.ins().brif(
                    condition,
                    labels[yes.0 as usize],
                    &[],
                    labels[no.0 as usize],
                    &[],
                );
                open = false;
            }
            Instr::Return(r) => {
                let value = b.use_var(var(r));
                let result = b.ins().uextend(I64, value);
                b.ins().return_(&[result]);
                open = false;
            }
            Instr::TrapBounds => {
                let fault = b.ins().iconst(I64, BOUNDS);
                b.ins().return_(&[fault]);
                open = false;
            }
        }
    }
    if open {
        // The Splice verifier only permits fallthrough at an unreachable tail.
        b.ins().trap(TrapCode::unwrap_user(1));
    }
    b.seal_all_blocks();
    b.finalize();
}

fn branch_condition(b: &FunctionBuilder, mut value: Value) -> Value {
    // brif only tests for zero. These operations preserve that test while
    // forming a Splice mask. Other users still see the materialized mask.
    let dfg = &b.func.dfg;
    while let Some(inst) = dfg.value_def(value).inst() {
        match dfg.insts[inst].opcode() {
            Opcode::Ineg | Opcode::Uextend | Opcode::Bmask => value = dfg.inst_args(inst)[0],
            _ => break,
        }
    }
    value
}

fn return_if(b: &mut FunctionBuilder, condition: Value, result: Value) {
    let exit = b.create_block();
    let next = b.create_block();
    b.set_cold_block(exit);
    b.ins().brif(condition, exit, &[], next, &[]);
    b.seal_block(exit);
    b.seal_block(next);
    b.switch_to_block(exit);
    b.ins().return_(&[result]);
    b.switch_to_block(next);
}

fn element(b: &mut FunctionBuilder, [data, len]: [Value; 2], index: Value) -> Value {
    // Slice lengths fit i32, so unsigned comparison also rejects negative indices.
    let invalid = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, index, len);
    let bounds = b.ins().iconst(I64, BOUNDS);
    return_if(b, invalid, bounds);
    let index = b.ins().uextend(I64, index);
    let bytes = b.ins().imul_imm(index, size_of::<i32>() as i64);
    b.ins().iadd(data, bytes)
}

fn binary(b: &mut FunctionBuilder, op: Binary, lhs: Value, rhs: Value) -> Value {
    use Binary::*;
    let cc = match op {
        Add => return b.ins().iadd(lhs, rhs),
        Sub => return b.ins().isub(lhs, rhs),
        Mul => return b.ins().imul(lhs, rhs),
        CheckedAdd | CheckedSub | CheckedMul => {
            let (result, overflow) = match op {
                CheckedAdd => b.ins().sadd_overflow(lhs, rhs),
                CheckedSub => b.ins().ssub_overflow(lhs, rhs),
                CheckedMul => b.ins().smul_overflow(lhs, rhs),
                _ => unreachable!(),
            };
            let fault = b.ins().iconst(I64, INTEGER_OVERFLOW);
            return_if(b, overflow, fault);
            return result;
        }
        And => return b.ins().band(lhs, rhs),
        Or => return b.ins().bor(lhs, rhs),
        Xor => return b.ins().bxor(lhs, rhs),
        Shl => return b.ins().ishl(lhs, rhs),
        Shr => return b.ins().sshr(lhs, rhs),
        CheckedShl | CheckedShr => {
            let invalid = b.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, rhs, 32);
            let fault = b.ins().iconst(I64, INTEGER_OVERFLOW);
            return_if(b, invalid, fault);
            return if op == CheckedShl {
                b.ins().ishl(lhs, rhs)
            } else {
                b.ins().sshr(lhs, rhs)
            };
        }
        Min => return b.ins().smin(lhs, rhs),
        Max => return b.ins().smax(lhs, rhs),
        Div | Rem => {
            let zero = b.ins().icmp_imm(IntCC::Equal, rhs, 0);
            let fault = b.ins().iconst(I64, DIV_ZERO);
            return_if(b, zero, fault);
            let min = b.ins().icmp_imm(IntCC::Equal, lhs, i64::from(i32::MIN));
            let minus_one = b.ins().icmp_imm(IntCC::Equal, rhs, -1);
            let overflow = b.ins().band(min, minus_one);
            let fault = b.ins().iconst(I64, OVERFLOW);
            return_if(b, overflow, fault);
            return if op == Div {
                b.ins().sdiv(lhs, rhs)
            } else {
                b.ins().srem(lhs, rhs)
            };
        }
        Eq => IntCC::Equal,
        Ne => IntCC::NotEqual,
        Lt => IntCC::SignedLessThan,
        Le => IntCC::SignedLessThanOrEqual,
        Gt => IntCC::SignedGreaterThan,
        Ge => IntCC::SignedGreaterThanOrEqual,
    };
    // Cranelift icmp produces 0/1. Splice exposes an all-bits-set true mask.
    let flag = b.ins().icmp(cc, lhs, rhs);
    let bit = b.ins().uextend(I32, flag);
    b.ins().ineg(bit)
}
