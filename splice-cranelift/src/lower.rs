use crate::{BOUNDS, DIV_ZERO, Descriptor, OVERFLOW};
use cranelift_codegen::ir::{
    FuncRef, InstBuilder, MemFlags, Opcode, TrapCode, Value,
    condcodes::IntCC,
    types::{I32, I64},
};
use cranelift_frontend::FunctionBuilder;
use splice::Function;
use splice::hir::{Binary, Instr, Reg};
use std::collections::HashMap;
use std::mem::{offset_of, size_of};
use std::sync::Arc;

mod simd;

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
    let mut open = true;
    for op in f.instructions() {
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
        And => return b.ins().band(lhs, rhs),
        Or => return b.ins().bor(lhs, rhs),
        Xor => return b.ins().bxor(lhs, rhs),
        Shl => return b.ins().ishl(lhs, rhs),
        Shr => return b.ins().sshr(lhs, rhs),
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
