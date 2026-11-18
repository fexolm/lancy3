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
    pub(super) fn start(
        ctx: &'ctx Context,
        builder: &'a Builder<'ctx>,
        function: FunctionValue<'ctx>,
        arity: usize,
        register_count: usize,
        buffer_count: usize,
    ) -> Built<Self> {
        builder.position_at_end(ctx.append_basic_block(function, "entry"));
        // Only the entry allocates register slots. LLVM promotes these to SSA,
        // including phi nodes for mutable locals across joins and backedges.
        let registers = (0..register_count)
            .map(|_| builder.build_alloca(ctx.i32_type(), ""))
            .collect::<Built<Vec<_>>>()?;
        let mut params = function.get_param_iter();
        let status = params.next().unwrap().into_pointer_value();
        for &ptr in &registers[..arity] {
            builder.build_store(ptr, params.next().unwrap().into_int_value())?;
        }
        let slices = (0..buffer_count)
            .map(|_| Slice {
                data: params.next().unwrap().into_pointer_value(),
                len: params.next().unwrap().into_int_value(),
            })
            .collect();
        Ok(Self {
            ctx,
            builder,
            function,
            status,
            registers,
            slices,
            labels: Vec::new(),
            exits: [None; 4],
        })
    }

    pub(super) fn emit(
        ctx: &'ctx Context,
        builder: &'a Builder<'ctx>,
        f: &Function,
        function: FunctionValue<'ctx>,
        ids: &HashMap<*const Function, usize>,
        declarations: &[FunctionValue<'ctx>],
    ) -> Built<()> {
        let mut body = Self::start(
            ctx,
            builder,
            function,
            f.arity(),
            f.register_count(),
            f.buffer_kinds().len(),
        )?;
        // A builder may allocate label IDs without marking or using them.
        body.labels.resize(f.label_count(), None);
        for op in f.instructions() {
            if let Instr::Label(l) = op {
                body.labels[l.0 as usize] = Some(ctx.append_basic_block(function, ""));
            }
        }
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

    pub(super) fn new_local(&mut self, allocas: &Builder<'ctx>) -> Built<Reg> {
        // A second builder can insert slots in the entry even while emitting
        // code for a loop or a guarded continuation in a different block.
        let entry = self.function.get_first_basic_block().unwrap();
        if let Some(first) = entry.get_first_instruction() {
            allocas.position_before(&first);
        } else {
            allocas.position_at_end(entry);
        }
        let id = Reg(self.registers.len() as u32);
        self.registers
            .push(allocas.build_alloca(self.ctx.i32_type(), "")?);
        Ok(id)
    }

    pub(super) fn register_count(&self) -> usize {
        self.registers.len()
    }
    pub(super) fn label_count(&self) -> usize {
        self.labels.len()
    }
    pub(super) fn new_label(&mut self) -> Label {
        let label = Label(self.labels.len() as u32);
        self.labels.push(None);
        label
    }
    pub(super) fn prepare_label(&mut self, label: Label) {
        self.labels[label.0 as usize]
            .get_or_insert_with(|| self.ctx.append_basic_block(self.function, ""));
    }
    pub(super) fn begin_block(&self, label: Label) {
        self.builder.position_at_end(self.block(label));
    }
    pub(super) fn unreachable(&self) -> Built<()> {
        self.builder.build_unreachable().map(|_| ())
    }
    pub(super) fn needs_status(&self) -> bool {
        self.exits.iter().any(Option::is_some)
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

    pub(super) fn constant(&self, dst: Reg, n: i32) -> Built<()> {
        self.store(dst, self.int(n))
    }
    pub(super) fn assign(&self, dst: Reg, src: Reg) -> Built<()> {
        self.store(dst, self.load(src)?)
    }
    pub(super) fn arithmetic(&mut self, dst: Reg, op: Binary, a: Reg, b: Reg) -> Built<()> {
        let result = self.binary(op, self.load(a)?, self.load(b)?)?;
        self.store(dst, result)
    }
    pub(super) fn length(&self, dst: Reg, buffer: Buffer) -> Built<()> {
        let len = self.slices[buffer.index()].len;
        self.store(
            dst,
            self.builder
                .build_int_truncate(len, self.ctx.i32_type(), "")?,
        )
    }
    pub(super) fn read_element(&mut self, dst: Reg, buffer: Buffer, index: Reg) -> Built<()> {
        let ptr = self.element(buffer, index)?;
        self.store(
            dst,
            self.builder
                .build_load(self.ctx.i32_type(), ptr, "")?
                .into_int_value(),
        )
    }
    pub(super) fn write_element(&mut self, buffer: Buffer, index: Reg, value: Reg) -> Built<()> {
        let ptr = self.element(buffer, index)?;
        self.builder.build_store(ptr, self.load(value)?)?;
        Ok(())
    }
    pub(super) fn jump(&self, label: Label) -> Built<()> {
        self.builder.build_unconditional_branch(self.block(label))?;
        Ok(())
    }
    pub(super) fn branch(&self, cond: Reg, yes: Label, no: Label) -> Built<()> {
        let take =
            self.builder
                .build_int_compare(IntPredicate::NE, self.load(cond)?, self.int(0), "")?;
        self.builder
            .build_conditional_branch(take, self.block(yes), self.block(no))?;
        Ok(())
    }
    pub(super) fn ret(&self, value: Reg) -> Built<()> {
        self.builder.build_return(Some(&self.load(value)?))?;
        Ok(())
    }
    pub(super) fn call_function(
        &mut self,
        dst: Reg,
        callee: FunctionValue<'ctx>,
        needs_status: bool,
        args: &[Reg],
        buffers: &[Buffer],
    ) -> Built<()> {
        let ir = self.builder;
        let mut actuals =
            Vec::<BasicMetadataValueEnum>::with_capacity(callee.count_params() as usize);
        actuals.push(self.status.into());
        for &r in args {
            actuals.push(self.load(r)?.into());
        }
        for buffer in buffers {
            let slice = self.slices[buffer.index()];
            actuals.push(slice.data.into());
            actuals.push(slice.len.into());
        }
        let result = call(ir, callee, &actuals)?;
        if needs_status {
            let status = ir
                .build_load(self.ctx.i32_type(), self.status, "")?
                .into_int_value();
            let failed = ir.build_int_compare(IntPredicate::NE, status, self.int(0), "")?;
            self.exit_if(failed, Failure::Propagate)?;
        }
        self.store(dst, result)
    }

    fn instruction(
        &mut self,
        inst: &Instr,
        ids: &HashMap<*const Function, usize>,
        declarations: &[FunctionValue<'ctx>],
    ) -> Built<()> {
        match *inst {
            Instr::Const(d, n) => self.constant(d, n),
            Instr::Copy(d, v) => self.assign(d, v),
            Instr::Bin(d, op, a, b) => self.arithmetic(d, op, a, b),
            Instr::BufferLen { dst, buffer } => self.length(dst, buffer),
            Instr::Load { dst, buffer, index } => self.read_element(dst, buffer, index),
            Instr::Store {
                buffer,
                index,
                value,
            } => self.write_element(buffer, index, value),
            Instr::Jump(l) => self.jump(l),
            Instr::Branch { cond, yes, no } => self.branch(cond, yes, no),
            Instr::Return(v) => self.ret(v),
            Instr::Call {
                dst,
                ref function,
                ref args,
                ref buffers,
            } => self.call_function(
                dst,
                declarations[ids[&Arc::as_ptr(function)]],
                function.needs_status(),
                args,
                buffers,
            ),
            Instr::Label(_) => unreachable!("handled at block boundary"),
        }
    }
}
