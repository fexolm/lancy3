use super::call;
use inkwell::IntPredicate;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::values::{BasicMetadataValueEnum, FunctionValue, IntValue, PointerValue};
use splice::Function;
use splice::hir::{Binary, Buffer, Instr, Label, Reg};
use std::collections::HashMap;
use std::sync::Arc;

type Built<T> = Result<T, BuilderError>;

#[derive(Clone, Copy)]
struct Slice<'ctx> {
    data: PointerValue<'ctx>,
    len: IntValue<'ctx>,
}

#[derive(Clone, Copy, PartialEq)]
enum Failure {
    Propagate = 0,
    Bounds = 1,
    DivideByZero = 2,
    Overflow = 3,
}

pub(super) struct Body<'a, 'ctx> {
    ctx: &'ctx Context,
    builder: &'a Builder<'ctx>,
    function: FunctionValue<'ctx>,
    status: PointerValue<'ctx>,
    registers: Vec<PointerValue<'ctx>>,
    slices: Vec<Slice<'ctx>>,
    labels: Vec<Option<BasicBlock<'ctx>>>,
    exits: [Option<BasicBlock<'ctx>>; 4],
}

impl<'a, 'ctx> Body<'a, 'ctx> {
    pub(super) fn emit(
        ctx: &'ctx Context,
        builder: &'a Builder<'ctx>,
        f: &Function,
        function: FunctionValue<'ctx>,
        ids: &HashMap<*const Function, usize>,
        declarations: &[FunctionValue<'ctx>],
    ) -> Built<()> {
        builder.position_at_end(ctx.append_basic_block(function, "entry"));
        // Only the entry allocates register slots. LLVM promotes these to SSA,
        // including phi nodes for mutable locals across joins and backedges.
        let registers = (0..f.register_count())
            .map(|_| builder.build_alloca(ctx.i32_type(), ""))
            .collect::<Built<Vec<_>>>()?;
        let mut params = function.get_param_iter();
        let status = params.next().unwrap().into_pointer_value();
        for &ptr in &registers[..f.arity()] {
            builder.build_store(ptr, params.next().unwrap().into_int_value())?;
        }
        let slices = f
            .buffer_kinds()
            .iter()
            .map(|_| Slice {
                data: params.next().unwrap().into_pointer_value(),
                len: params.next().unwrap().into_int_value(),
            })
            .collect();
        // A builder may allocate label IDs without ever marking or using them.
        let mut labels = vec![None; f.label_count()];
        for op in f.instructions() {
            if let Instr::Label(l) = op {
                labels[l.0 as usize] = Some(ctx.append_basic_block(function, ""));
            }
        }
        let mut body = Self {
            ctx,
            builder,
            function,
            status,
            registers,
            slices,
            labels,
            exits: [None; 4],
        };
        let mut open = true;
        for inst in f.instructions() {
            if let Instr::Label(l) = inst {
                let block = body.block(*l);
                if open {
                    builder.build_unconditional_branch(block)?;
                }
                builder.position_at_end(block);
                open = true;
            } else if open {
                body.instruction(inst, ids, declarations)?;
                open = !matches!(
                    inst,
                    Instr::Return(_) | Instr::Jump(_) | Instr::Branch { .. }
                );
            }
        }
        // A dead block at the end of verified HIR may lack a terminator.
        if open {
            builder.build_unreachable()?;
        }
        Ok(())
    }

    fn block(&self, label: Label) -> BasicBlock<'ctx> {
        self.labels[label.0 as usize].expect("verified jump target")
    }

    fn int(&self, n: i32) -> IntValue<'ctx> {
        self.ctx.i32_type().const_int(n as u32 as u64, false)
    }

    fn load(&self, reg: Reg) -> Built<IntValue<'ctx>> {
        Ok(self
            .builder
            .build_load(self.ctx.i32_type(), self.registers[reg.0 as usize], "")?
            .into_int_value())
    }

    fn store(&self, reg: Reg, value: IntValue<'ctx>) -> Built<()> {
        self.builder
            .build_store(self.registers[reg.0 as usize], value)?;
        Ok(())
    }

    // Guards of the same type share one side exit. Failed calls preserve
    // their callee's status and return immediately to the enclosing caller.
    fn exit_if(&mut self, condition: IntValue<'ctx>, fault: Failure) -> Built<()> {
        let existing = self.exits[fault as usize];
        let exit = existing.unwrap_or_else(|| self.ctx.append_basic_block(self.function, "error"));
        let next = self.ctx.append_basic_block(self.function, "");
        self.builder
            .build_conditional_branch(condition, exit, next)?;
        if existing.is_none() {
            self.exits[fault as usize] = Some(exit);
            self.builder.position_at_end(exit);
            if fault != Failure::Propagate {
                self.builder
                    .build_store(self.status, self.int(fault as i32))?;
            }
            self.builder.build_return(Some(&self.int(0)))?;
        }
        self.builder.position_at_end(next);
        Ok(())
    }

    fn element(&mut self, buffer: Buffer, index: Reg) -> Built<PointerValue<'ctx>> {
        let slice = self.slices[buffer.0 as usize];
        let index = self
            .builder
            .build_int_z_extend(self.load(index)?, self.ctx.i64_type(), "")?;
        // Zero extension makes negative indices compare above every legal slice length.
        let invalid = self
            .builder
            .build_int_compare(IntPredicate::UGE, index, slice.len, "")?;
        self.exit_if(invalid, Failure::Bounds)?;
        // SAFETY: Only the edge with index < slice.len reaches this instruction.
        unsafe {
            self.builder
                .build_in_bounds_gep(self.ctx.i32_type(), slice.data, &[index], "")
        }
    }

    fn binary(
        &mut self,
        op: Binary,
        a: IntValue<'ctx>,
        b: IntValue<'ctx>,
    ) -> Built<IntValue<'ctx>> {
        let ir = self.builder;
        use Binary::*;
        match op {
            Add => ir.build_int_add(a, b, ""),
            Sub => ir.build_int_sub(a, b, ""),
            Mul => ir.build_int_mul(a, b, ""),
            And => ir.build_and(a, b, ""),
            Or => ir.build_or(a, b, ""),
            Xor => ir.build_xor(a, b, ""),
            Div | Rem => {
                let zero = ir.build_int_compare(IntPredicate::EQ, b, self.int(0), "")?;
                self.exit_if(zero, Failure::DivideByZero)?;
                let min = ir.build_int_compare(IntPredicate::EQ, a, self.int(i32::MIN), "")?;
                let minus_one = ir.build_int_compare(IntPredicate::EQ, b, self.int(-1), "")?;
                self.exit_if(ir.build_and(min, minus_one, "")?, Failure::Overflow)?;
                if op == Div {
                    ir.build_int_signed_div(a, b, "")
                } else {
                    ir.build_int_signed_rem(a, b, "")
                }
            }
            Shl | Shr => {
                let count = ir.build_and(b, self.int(31), "")?;
                if op == Shl {
                    ir.build_left_shift(a, count, "")
                } else {
                    ir.build_right_shift(a, count, true, "")
                }
            }
            Min | Max => {
                let pred = if op == Min {
                    IntPredicate::SLT
                } else {
                    IntPredicate::SGT
                };
                let take_a = ir.build_int_compare(pred, a, b, "")?;
                Ok(ir.build_select(take_a, a, b, "")?.into_int_value())
            }
            Eq | Ne | Lt | Le | Gt | Ge => {
                let pred = match op {
                    Eq => IntPredicate::EQ,
                    Ne => IntPredicate::NE,
                    Lt => IntPredicate::SLT,
                    Le => IntPredicate::SLE,
                    Gt => IntPredicate::SGT,
                    Ge => IntPredicate::SGE,
                    _ => unreachable!(),
                };
                let flag = ir.build_int_compare(pred, a, b, "")?;
                ir.build_int_s_extend(flag, self.ctx.i32_type(), "")
            }
        }
    }

    fn instruction(
        &mut self,
        inst: &Instr,
        ids: &HashMap<*const Function, usize>,
        declarations: &[FunctionValue<'ctx>],
    ) -> Built<()> {
        let ir = self.builder;
        match inst {
            Instr::Const(dst, n) => self.store(*dst, self.int(*n))?,
            Instr::Copy(dst, src) => self.store(*dst, self.load(*src)?)?,
            Instr::Bin(dst, op, a, b) => {
                let result = self.binary(*op, self.load(*a)?, self.load(*b)?)?;
                self.store(*dst, result)?;
            }
            Instr::BufferLen { dst, buffer } => {
                let len = self.slices[buffer.0 as usize].len;
                self.store(*dst, ir.build_int_truncate(len, self.ctx.i32_type(), "")?)?;
            }
            Instr::Load { dst, buffer, index } => {
                let ptr = self.element(*buffer, *index)?;
                self.store(
                    *dst,
                    ir.build_load(self.ctx.i32_type(), ptr, "")?
                        .into_int_value(),
                )?;
            }
            Instr::Store {
                buffer,
                index,
                value,
            } => {
                let ptr = self.element(*buffer, *index)?;
                ir.build_store(ptr, self.load(*value)?)?;
            }
            Instr::Jump(label) => {
                ir.build_unconditional_branch(self.block(*label))?;
            }
            Instr::Branch { cond, yes, no } => {
                let take =
                    ir.build_int_compare(IntPredicate::NE, self.load(*cond)?, self.int(0), "")?;
                ir.build_conditional_branch(take, self.block(*yes), self.block(*no))?;
            }
            Instr::Return(reg) => {
                ir.build_return(Some(&self.load(*reg)?))?;
            }
            Instr::Call {
                dst,
                function,
                args,
                buffers,
            } => {
                let callee = declarations[ids[&Arc::as_ptr(function)]];
                let mut actuals =
                    Vec::<BasicMetadataValueEnum>::with_capacity(callee.count_params() as usize);
                actuals.push(self.status.into());
                for &r in args {
                    actuals.push(self.load(r)?.into());
                }
                for buffer in buffers {
                    let slice = self.slices[buffer.0 as usize];
                    actuals.push(slice.data.into());
                    actuals.push(slice.len.into());
                }
                let result = call(ir, callee, &actuals)?;
                if function.needs_status() {
                    let status = ir
                        .build_load(self.ctx.i32_type(), self.status, "")?
                        .into_int_value();
                    let failed = ir.build_int_compare(IntPredicate::NE, status, self.int(0), "")?;
                    self.exit_if(failed, Failure::Propagate)?;
                }
                self.store(*dst, result)?;
            }
            Instr::Label(_) => unreachable!("handled at block boundary"),
        }
        Ok(())
    }
}
