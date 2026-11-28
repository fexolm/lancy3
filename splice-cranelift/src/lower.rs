use cranelift_codegen::ir::{
    DebugTag, FuncRef, InstBuilder, MemFlags, StackSlotData, StackSlotKind, TrapCode, Type, Value,
    condcodes::{FloatCC, IntCC},
    immediates::{Ieee32, Ieee64},
    types::{F32, F64, I8, I16, I32, I64},
};
use cranelift_frontend::FunctionBuilder;
use splice_core::Function;
use splice_core::Trap;
use splice_core::call::SliceDescriptor;
use splice_core::hir::{
    Binary, Constant, Instr, PacketStore, PacketValue, Reg, ScalarType, ValueType,
};
use std::collections::HashMap;
use std::mem::{offset_of, size_of};
use std::sync::Arc;

mod simd;

pub(crate) use simd::{Machine, keep_compared_loads_in_registers};

/// The CLIF type of a scalar register. A `Bool` is an `I8` holding 0 or 1;
/// a `Char` is its code point.
pub(crate) fn scalar_type(ty: ScalarType) -> Type {
    match ty {
        ScalarType::Bool | ScalarType::I8 | ScalarType::U8 => I8,
        ScalarType::I16 | ScalarType::U16 => I16,
        ScalarType::Char | ScalarType::I32 | ScalarType::U32 => I32,
        ScalarType::I64 | ScalarType::U64 | ScalarType::Index => I64,
        ScalarType::F32 => F32,
        ScalarType::F64 => F64,
    }
}

/// A function returns its status. A result of at most 32 bits is the low
/// half of a success status; a wider result goes to a slot whose address
/// the caller passes before the scalars.
pub(crate) fn returns_in_memory(ty: ScalarType) -> bool {
    ty.bits() > 32
}

fn scalar(f: &Function, reg: Reg) -> ScalarType {
    match f.reg_type(reg) {
        ValueType::Scalar(ty) => ty,
        ty => unreachable!("verified scalar register, got {ty:?}"),
    }
}

/// A scalar as the 8-byte word of a Rust call operand: its bits
/// zero-extended.
fn word(builder: &mut FunctionBuilder, value: Value, ty: ScalarType) -> Value {
    let bits = match ty {
        ScalarType::F32 => builder.ins().bitcast(I32, MemFlags::new(), value),
        ScalarType::F64 => return builder.ins().bitcast(I64, MemFlags::new(), value),
        _ if ty.bits() == 64 => return value,
        _ => value,
    };
    builder.ins().uextend(I64, bits)
}

/// Return a failed status from this function unchanged.
fn forward_status(builder: &mut FunctionBuilder, status: Value) {
    let failed = builder
        .ins()
        .icmp_imm(IntCC::UnsignedGreaterThan, status, i64::from(u32::MAX));
    return_if(builder, failed, status);
}

/// The result in the low half of a success status.
fn from_status(b: &mut FunctionBuilder, ty: ScalarType, status: Value) -> Value {
    if ty == ScalarType::F32 {
        let word = b.ins().ireduce(I32, status);
        b.ins().bitcast(F32, MemFlags::new(), word)
    } else {
        b.ins().ireduce(scalar_type(ty), status)
    }
}

/// The success status carrying a result of at most 32 bits.
fn to_status(b: &mut FunctionBuilder, ty: ScalarType, value: Value) -> Value {
    let word = if ty == ScalarType::F32 {
        b.ins().bitcast(I32, MemFlags::new(), value)
    } else {
        value
    };
    b.ins().uextend(I64, word)
}

pub(crate) fn trap(builder: &mut FunctionBuilder, trap: Trap) -> Value {
    builder.ins().iconst(I64, trap.status())
}

/// The scalar and slice parameters, and the result slot of the entry or of
/// a helper that returns its result in memory.
fn parameters(
    b: &mut FunctionBuilder,
    f: &Function,
    is_entry: bool,
) -> (Vec<Value>, Option<Value>) {
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);
    if !is_entry {
        let params = b.block_params(entry);
        return if returns_in_memory(f.result()) {
            (params[1..].to_vec(), Some(params[0]))
        } else {
            (params.to_vec(), None)
        };
    }
    let [scalars, descriptors, result] = b.block_params(entry).try_into().unwrap();
    let mut params = Vec::new();
    // `Executable::call` checks both array lengths and every word, and
    // constructs the descriptors. A word holds its value in the low
    // bytes (the hosts are little-endian), so it loads at the value's type.
    let flags = MemFlags::trusted();
    for (i, &ty) in f.params().iter().enumerate() {
        let addr = b.ins().iadd_imm(scalars, (i * size_of::<u64>()) as i64);
        params.push(b.ins().load(scalar_type(ty), flags, addr, 0));
    }
    for i in 0..f.buffer_kinds().len() {
        let addr = b
            .ins()
            .iadd_imm(descriptors, (i * size_of::<SliceDescriptor>()) as i64);
        params.push(
            b.ins()
                .load(I64, flags, addr, offset_of!(SliceDescriptor, data) as i32),
        );
        params.push(
            b.ins()
                .load(I64, flags, addr, offset_of!(SliceDescriptor, len) as i32),
        );
    }
    // The caller zeroes the result word, so storing the value at its type
    // leaves the word's bits above it zero.
    (params, Some(result))
}

pub(super) fn emit(
    mut b: FunctionBuilder,
    f: &Function,
    is_entry: bool,
    imports: &HashMap<*const Function, FuncRef>,
    m: simd::Machine,
    debug_bytes: Option<u32>,
) {
    let (params, result_slot) = parameters(&mut b, f, is_entry);
    let vars = simd::Registers::new(&mut b, f, m);
    let var = |r: Reg| vars.get(r)[0];
    // A buffer parameter is the slice of its two registers: the data
    // pointer and the `Index` length.
    for (index, &value) in params.iter().enumerate() {
        b.def_var(var(Reg(index as u32)), value);
    }
    let labels: Vec<_> = (0..f.label_count()).map(|_| b.create_block()).collect();
    let snapshots = debug_bytes.map(|bytes| {
        b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, bytes, 3))
    });
    // One slot holds the packet of every Rust call: each call is
    // synchronous, so the next one can reuse it.
    let packet_bytes = f
        .instructions()
        .iter()
        .filter_map(|op| match op {
            Instr::RustCall {
                target,
                args,
                slices,
                ..
            } => Some(
                target
                    .plan(&|r| Some(f.reg_type(r)), args.len(), slices)
                    .bytes,
            ),
            _ => None,
        })
        .max();
    let packet = packet_bytes.map(|bytes| {
        b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, bytes, 3))
    });
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
                use splice_core::debug::Store as DebugStore;
                let mut slots = point.function_slots(f, m.vector_bytes).peekable();
                if slots.peek().is_none() {
                    // Sequence points themselves occupy no bytes in Cranelift.
                    // Give consecutive empty statements distinct addresses.
                    let zero = b.ins().iconst(I8, 0);
                    b.ins().stack_store(zero, slot, 0);
                }
                for point_slot in slots {
                    for (offset, store) in point_slot.stores(|r| f.reg_type(r)) {
                        let offset = offset as i32;
                        match store {
                            // A register already has its memory layout: a
                            // bool is one byte holding 0 or 1, and a mask is
                            // a vector of all-ones or zero index lanes.
                            DebugStore::Scalar(reg, _)
                            | DebugStore::Vector(reg)
                            | DebugStore::Mask(reg, _)
                            | DebugStore::Data(reg)
                            | DebugStore::Length(reg) => {
                                for (part, &variable) in vars.get(reg).iter().enumerate() {
                                    let value = b.use_var(variable);
                                    b.ins()
                                        .stack_store(value, slot, offset + (part * 16) as i32);
                                }
                            }
                        }
                    }
                }
                let at = b.ins().sequence_point();
                b.func.debug_tags.set(
                    at,
                    [DebugTag::User(index as u32), DebugTag::StackSlot(slot)],
                );
            }
            Instr::Const(dst, c) => {
                let value = constant(&mut b, c);
                b.def_var(var(dst), value);
            }
            Instr::Cast { dst, src } => {
                let value = b.use_var(var(src));
                let value = convert(&mut b, value, scalar(f, src), scalar(f, dst));
                b.def_var(var(dst), value);
            }
            Instr::Copy(dst, src) => {
                for (&d, &s) in vars.get(dst).iter().zip(vars.get(src)) {
                    let value = b.use_var(s);
                    b.def_var(d, value);
                }
            }
            Instr::CallIntrinsic {
                dst,
                intrinsic: splice_core::hir::Intrinsic::Simd(inst),
                ref args,
            } => {
                simd::emit(&mut b, m, inst, &vars, dst, args);
            }
            Instr::Bin(dst, op, left, right) => {
                let lhs = b.use_var(var(left));
                let rhs = b.use_var(var(right));
                let value = binary(&mut b, op, scalar(f, left), lhs, rhs);
                b.def_var(var(dst), value);
            }
            Instr::SliceRange {
                dst,
                slice,
                start,
                end,
            } => {
                // `&slice[s..e]` for `e = min(end, len)`, `s = min(start, e)`.
                let ValueType::Ptr(kind) = f.reg_type(slice.ptr) else {
                    unreachable!("verified slice pointer");
                };
                let index = ScalarType::Index;
                let len = b.use_var(var(slice.len));
                let end = if end == slice.len {
                    len
                } else {
                    let end = b.use_var(var(end));
                    binary(&mut b, Binary::Min, index, end, len)
                };
                let start = b.use_var(var(start));
                let start = binary(&mut b, Binary::Min, index, start, end);
                let base = b.use_var(var(slice.ptr));
                let bytes = b.ins().imul_imm(start, kind.element().bytes() as i64);
                let ptr = b.ins().iadd(base, bytes);
                b.def_var(var(dst.ptr), ptr);
                let length = binary(&mut b, Binary::Sub, index, end, start);
                b.def_var(var(dst.len), length);
            }
            Instr::Load { dst, slice, index } => {
                let index = b.use_var(var(index));
                let ty = scalar_type(scalar(f, dst));
                let pair = [b.use_var(var(slice.ptr)), b.use_var(var(slice.len))];
                let addr = element(&mut b, pair, index, ty);
                let value = b.ins().load(ty, MemFlags::new(), addr, 0);
                b.def_var(var(dst), value);
            }
            Instr::Store {
                slice,
                index,
                value,
            } => {
                let index = b.use_var(var(index));
                let ty = scalar_type(scalar(f, value));
                let value = b.use_var(var(value));
                let pair = [b.use_var(var(slice.ptr)), b.use_var(var(slice.len))];
                let addr = element(&mut b, pair, index, ty);
                b.ins().store(MemFlags::new(), value, addr, 0);
            }
            Instr::RustCall {
                dst,
                target,
                ref args,
                ref slices,
                result,
            } => {
                use cranelift_codegen::ir::{AbiParam, Signature};
                let plan = target.plan(&|r| Some(f.reg_type(r)), args.len(), slices);
                let packet = packet.expect("a Rust call has a packet slot");
                for PacketStore { at, value } in plan.stores {
                    let value = match value {
                        PacketValue::Word(n) => {
                            let value = b.use_var(var(args[n]));
                            word(&mut b, value, scalar(f, args[n]))
                        }
                        PacketValue::Data(ptr) => b.use_var(var(ptr)),
                        PacketValue::Length(len) => b.use_var(var(len)),
                        PacketValue::Tag(tag) => b.ins().iconst(I32, i64::from(tag)),
                        PacketValue::Symbol(address) => b.ins().iconst(I64, address as i64),
                    };
                    b.ins().stack_store(value, packet, at as i32);
                }
                let address = b.ins().stack_addr(I64, packet, 0);
                let receiver = plan
                    .context
                    .map_or(address, |context| b.use_var(var(context.ptr)));
                let operation = b.ins().iconst(I32, i64::from(plan.operation));
                let zero = b.ins().iconst(I32, 0);
                let mut signature = Signature::new(b.func.signature.call_conv);
                signature
                    .params
                    .extend([I64, I32, I64, I32].map(AbiParam::new));
                signature.returns.push(AbiParam::new(I64));
                let sig = b.import_signature(signature);
                let entry = b.ins().iconst(I64, plan.entry.entry as usize as i64);
                let call = b
                    .ins()
                    .call_indirect(sig, entry, &[receiver, operation, address, zero]);
                let packed = b.inst_results(call)[0];
                forward_status(&mut b, packed);
                let value = b.ins().stack_load(scalar_type(result), packet, 0);
                b.def_var(var(dst), value);
            }
            Instr::CallIntrinsic {
                dst,
                intrinsic: splice_core::hir::Intrinsic::Math { op, .. },
                ref args,
            } => {
                let dst = dst.expect("verified math result");
                let args: Vec<_> = args.iter().map(|&r| b.use_var(var(r))).collect();
                let value = crate::intrinsic::emit(&mut b, op, &args);
                b.def_var(var(dst), value);
            }
            Instr::Call {
                dst,
                ref function,
                ref args,
                buffers: ref forwarded,
            } => {
                let result = function.result();
                let slot = returns_in_memory(result).then(|| {
                    b.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        result.bytes() as u32,
                        3,
                    ))
                });
                let mut actuals: Vec<_> = slot
                    .map(|slot| b.ins().stack_addr(I64, slot, 0))
                    .into_iter()
                    .collect();
                actuals.extend(args.iter().map(|&r| b.use_var(var(r))));
                for slice in forwarded {
                    let len = b.use_var(var(slice.len));
                    actuals.extend([b.use_var(var(slice.ptr)), len]);
                }
                let call = b.ins().call(imports[&Arc::as_ptr(function)], &actuals);
                let status = b.inst_results(call)[0];
                if function.needs_status() {
                    forward_status(&mut b, status);
                }
                let value = match slot {
                    Some(slot) => b.ins().stack_load(scalar_type(result), slot, 0),
                    None => from_status(&mut b, result, status),
                };
                b.def_var(var(dst), value);
            }
            Instr::Jump(label) => {
                b.ins().jump(labels[label.0 as usize], &[]);
                open = false;
            }
            Instr::Branch { cond, yes, no } => {
                let condition = b.use_var(var(cond));
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
                let status = match result_slot {
                    Some(slot) => {
                        b.ins().store(MemFlags::trusted(), value, slot, 0);
                        b.ins().iconst(I64, 0)
                    }
                    None => to_status(&mut b, f.result(), value),
                };
                b.ins().return_(&[status]);
                open = false;
            }
            Instr::TrapBounds => {
                let fault = trap(&mut b, Trap::Bounds);
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

fn element(b: &mut FunctionBuilder, [data, len]: [Value; 2], index: Value, ty: Type) -> Value {
    let invalid = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, index, len);
    let bounds = trap(b, Trap::Bounds);
    return_if(b, invalid, bounds);
    let bytes = b.ins().imul_imm(index, i64::from(ty.bytes()));
    b.ins().iadd(data, bytes)
}

fn constant(b: &mut FunctionBuilder, c: Constant) -> Value {
    match c.ty() {
        ScalarType::F32 => b.ins().f32const(Ieee32::with_bits(c.bits() as u32)),
        ScalarType::F64 => b.ins().f64const(Ieee64::with_bits(c.bits())),
        ty => b.ins().iconst(scalar_type(ty), c.bits() as i64),
    }
}

/// Rust's `value as to`.
fn convert(b: &mut FunctionBuilder, value: Value, from: ScalarType, to: ScalarType) -> Value {
    let target = scalar_type(to);
    match (from.is_float(), to.is_float()) {
        _ if from == to => value,
        (false, false) => {
            let width = scalar_type(from).bits();
            if width > target.bits() {
                b.ins().ireduce(target, value)
            } else if width == target.bits() {
                value
            } else if from.is_signed() {
                b.ins().sextend(target, value)
            } else {
                b.ins().uextend(target, value)
            }
        }
        // Cranelift converts 32- and 64-bit integers: a narrower one is
        // exactly its `i32`. A `bool` is an `I8`.
        (false, true) if scalar_type(from).bits() < 32 => {
            let wide = ScalarType::integer(32, from.is_signed());
            let value = convert(b, value, from, wide);
            convert(b, value, wide, to)
        }
        (false, true) if from.is_signed() => b.ins().fcvt_from_sint(target, value),
        (false, true) => b.ins().fcvt_from_uint(target, value),
        (true, true) if to == ScalarType::F64 => b.ins().fpromote(target, value),
        (true, true) => b.ins().fdemote(target, value),
        // Saturating, with NaN as zero; to a narrow integer through an
        // `i32`, clamped to its range.
        (true, false) if to.bits() < 32 => {
            let wide = b.ins().fcvt_to_sint_sat(I32, value);
            let (min, max) = if to.is_signed() {
                (-(1i64 << (to.bits() - 1)), (1i64 << (to.bits() - 1)) - 1)
            } else {
                (0, (1i64 << to.bits()) - 1)
            };
            let min = b.ins().iconst(I32, min);
            let max = b.ins().iconst(I32, max);
            let clamped = b.ins().smax(wide, min);
            let clamped = b.ins().smin(clamped, max);
            b.ins().ireduce(target, clamped)
        }
        (true, false) if to.is_signed() => b.ins().fcvt_to_sint_sat(target, value),
        (true, false) => b.ins().fcvt_to_uint_sat(target, value),
    }
}

fn binary(b: &mut FunctionBuilder, op: Binary, ty: ScalarType, lhs: Value, rhs: Value) -> Value {
    use Binary::*;
    if ty.is_float() {
        return float_binary(b, op, lhs, rhs);
    }
    let signed = ty.is_signed();
    let bits = i64::from(ty.bits());
    let cc = match op {
        Add => return b.ins().iadd(lhs, rhs),
        Sub => return b.ins().isub(lhs, rhs),
        Mul => return b.ins().imul(lhs, rhs),
        CheckedAdd | CheckedSub | CheckedMul => {
            let (result, overflow) = match (op, signed) {
                (CheckedAdd, true) => b.ins().sadd_overflow(lhs, rhs),
                (CheckedAdd, false) => b.ins().uadd_overflow(lhs, rhs),
                (CheckedSub, true) => b.ins().ssub_overflow(lhs, rhs),
                (CheckedSub, false) => b.ins().usub_overflow(lhs, rhs),
                (_, true) => b.ins().smul_overflow(lhs, rhs),
                (_, false) => b.ins().umul_overflow(lhs, rhs),
            };
            let fault = trap(b, Trap::Overflow);
            return_if(b, overflow, fault);
            return result;
        }
        SaturatingAdd | SaturatingSub | SaturatingMul => {
            let (result, overflow) = match (op, signed) {
                (SaturatingAdd, true) => b.ins().sadd_overflow(lhs, rhs),
                (SaturatingAdd, false) => b.ins().uadd_overflow(lhs, rhs),
                (SaturatingSub, true) => b.ins().ssub_overflow(lhs, rhs),
                (SaturatingSub, false) => b.ins().usub_overflow(lhs, rhs),
                (_, true) => b.ins().smul_overflow(lhs, rhs),
                (_, false) => b.ins().umul_overflow(lhs, rhs),
            };
            let int = b.func.dfg.value_type(lhs);
            // CLIF immediates hold the value's bits zero-extended.
            let bounded = |value: i64| (value as u64 & (u64::MAX >> (64 - bits))) as i64;
            let saturated = if signed {
                // Sums and differences leave the range on the side of `lhs`,
                // products on the side of their sign.
                let side = if op == SaturatingMul {
                    b.ins().bxor(lhs, rhs)
                } else {
                    lhs
                };
                let negative = b.ins().icmp_imm(IntCC::SignedLessThan, side, 0);
                let min = b.ins().iconst(int, bounded(i64::MIN >> (64 - bits)));
                let max = b.ins().iconst(int, bounded(i64::MAX >> (64 - bits)));
                b.ins().select(negative, min, max)
            } else if op == SaturatingSub {
                b.ins().iconst(int, 0)
            } else {
                b.ins().iconst(int, bounded(-1))
            };
            return b.ins().select(overflow, saturated, result);
        }
        And => return b.ins().band(lhs, rhs),
        Or => return b.ins().bor(lhs, rhs),
        Xor => return b.ins().bxor(lhs, rhs),
        // CLIF shifts take the amount modulo the width.
        Shl | Shr | CheckedShl | CheckedShr => {
            if matches!(op, CheckedShl | CheckedShr) {
                let invalid = b
                    .ins()
                    .icmp_imm(IntCC::UnsignedGreaterThanOrEqual, rhs, bits);
                let fault = trap(b, Trap::Overflow);
                return_if(b, invalid, fault);
            }
            return match (op, signed) {
                (Shl | CheckedShl, _) => b.ins().ishl(lhs, rhs),
                (_, true) => b.ins().sshr(lhs, rhs),
                (_, false) => b.ins().ushr(lhs, rhs),
            };
        }
        Min if signed => return b.ins().smin(lhs, rhs),
        Min => return b.ins().umin(lhs, rhs),
        Max if signed => return b.ins().smax(lhs, rhs),
        Max => return b.ins().umax(lhs, rhs),
        Div | Rem => {
            let zero = b.ins().icmp_imm(IntCC::Equal, rhs, 0);
            let fault = trap(b, Trap::DivisionByZero);
            return_if(b, zero, fault);
            if signed {
                let min = b.ins().icmp_imm(IntCC::Equal, lhs, i64::MIN >> (64 - bits));
                let minus_one = b.ins().icmp_imm(IntCC::Equal, rhs, -1);
                let overflow = b.ins().band(min, minus_one);
                let fault = trap(b, Trap::DivisionOverflow);
                return_if(b, overflow, fault);
            }
            return match (op, signed) {
                (Div, true) => b.ins().sdiv(lhs, rhs),
                (Div, false) => b.ins().udiv(lhs, rhs),
                (_, true) => b.ins().srem(lhs, rhs),
                (_, false) => b.ins().urem(lhs, rhs),
            };
        }
        Eq => IntCC::Equal,
        Ne => IntCC::NotEqual,
        Lt if signed => IntCC::SignedLessThan,
        Le if signed => IntCC::SignedLessThanOrEqual,
        Gt if signed => IntCC::SignedGreaterThan,
        Ge if signed => IntCC::SignedGreaterThanOrEqual,
        Lt => IntCC::UnsignedLessThan,
        Le => IntCC::UnsignedLessThanOrEqual,
        Gt => IntCC::UnsignedGreaterThan,
        Ge => IntCC::UnsignedGreaterThanOrEqual,
    };
    b.ins().icmp(cc, lhs, rhs)
}

fn float_binary(b: &mut FunctionBuilder, op: Binary, lhs: Value, rhs: Value) -> Value {
    use Binary::*;
    let cc = match op {
        Add => return b.ins().fadd(lhs, rhs),
        Sub => return b.ins().fsub(lhs, rhs),
        Mul => return b.ins().fmul(lhs, rhs),
        Div => return b.ins().fdiv(lhs, rhs),
        Min | Max => {
            // minimumNumber: fmin orders the zeros; a NaN operand yields the
            // other operand.
            let both = if op == Min {
                b.ins().fmin(lhs, rhs)
            } else {
                b.ins().fmax(lhs, rhs)
            };
            let lhs_nan = b.ins().fcmp(FloatCC::Unordered, lhs, lhs);
            let rhs_nan = b.ins().fcmp(FloatCC::Unordered, rhs, rhs);
            let other = b.ins().select(lhs_nan, rhs, both);
            return b.ins().select(rhs_nan, lhs, other);
        }
        Eq => FloatCC::Equal,
        // Unordered or unequal, like Rust's `!=`.
        Ne => FloatCC::NotEqual,
        Lt => FloatCC::LessThan,
        Le => FloatCC::LessThanOrEqual,
        Gt => FloatCC::GreaterThan,
        Ge => FloatCC::GreaterThanOrEqual,
        _ => unreachable!("verified float operation"),
    };
    b.ins().fcmp(cc, lhs, rhs)
}
