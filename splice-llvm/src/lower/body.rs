mod debug;
mod simd;

use super::call;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::types::BasicTypeEnum;
use inkwell::values::{BasicMetadataValueEnum, BasicValue, FunctionValue, IntValue, PointerValue};
use inkwell::{IntPredicate, module::Module};
use splice::hir::{
    Binary, Buffer, BufferKind, ContextSlice, Instr, Label, Reg, RustContextTarget, ValueType,
};
use splice::{ContextArguments, Function};
use std::collections::HashMap;
use std::sync::Arc;

type Built<T> = Result<T, BuilderError>;

pub(super) struct Callees<'a, 'ctx> {
    pub ids: &'a HashMap<*const Function, usize>,
    pub declarations: &'a [FunctionValue<'ctx>],
}

#[derive(Clone, Copy)]
struct Slice<'ctx> {
    data: PointerValue<'ctx>,
    len: IntValue<'ctx>,
    element: splice::simd::Element,
}

#[derive(Clone, Copy, PartialEq)]
enum Failure {
    Propagate = 0,
    Bounds = 1,
    DivideByZero = 2,
    Overflow = 3,
    IntegerOverflow = 4,
    RustPanic = 5,
}

pub(super) struct Body<'a, 'ctx> {
    ctx: &'ctx Context,
    builder: &'a Builder<'ctx>,
    module: &'a Module<'ctx>,
    function: FunctionValue<'ctx>,
    status: PointerValue<'ctx>,
    registers: Vec<(ValueType, PointerValue<'ctx>)>,
    slices: Vec<Slice<'ctx>>,
    context_scratch: Option<(u32, PointerValue<'ctx>)>,
    static_rust_slice_operands: bool,
    labels: Vec<Option<BasicBlock<'ctx>>>,
    exits: [Option<BasicBlock<'ctx>>; 6],
}

impl<'a, 'ctx> Body<'a, 'ctx> {
    pub(super) fn start(
        ctx: &'ctx Context,
        builder: &'a Builder<'ctx>,
        module: &'a Module<'ctx>,
        function: FunctionValue<'ctx>,
        arity: usize,
        types: impl IntoIterator<Item = ValueType>,
        kinds: &[BufferKind],
    ) -> Built<Self> {
        builder.position_at_end(ctx.append_basic_block(function, "entry"));
        // Only the entry allocates register slots. LLVM promotes these to SSA,
        // including phi nodes for mutable locals across joins and backedges.
        let registers = types
            .into_iter()
            .map(|ty| Ok((ty, builder.build_alloca(simd::llvm_type(ctx, ty), "")?)))
            .collect::<Built<Vec<_>>>()?;
        let mut params = function.get_param_iter();
        let status = params.next().unwrap().into_pointer_value();
        for &(_, ptr) in &registers[..arity] {
            builder.build_store(ptr, params.next().unwrap().into_int_value())?;
        }
        let slices = kinds
            .iter()
            .map(|kind| Slice {
                data: params.next().unwrap().into_pointer_value(),
                len: params.next().unwrap().into_int_value(),
                element: kind.element(),
            })
            .collect();
        Ok(Self {
            ctx,
            builder,
            module,
            function,
            status,
            registers,
            slices,
            context_scratch: None,
            static_rust_slice_operands: false,
            labels: Vec::new(),
            exits: [None; 6],
        })
    }

    pub(super) fn emit(
        ctx: &'ctx Context,
        builder: &'a Builder<'ctx>,
        module: &'a Module<'ctx>,
        f: &Function,
        function: FunctionValue<'ctx>,
        callees: &Callees<'_, 'ctx>,
        debug_id: Option<usize>,
    ) -> Built<bool> {
        let mut body = Self::start(
            ctx,
            builder,
            module,
            function,
            f.arity(),
            (0..f.register_count()).map(|i| f.reg_type(Reg(i as u32))),
            f.buffer_kinds(),
        )?;
        // A builder may allocate label IDs without marking or using them.
        body.labels.resize(f.label_count(), None);
        for op in f.instructions() {
            if let Instr::Label(l) = op {
                body.labels[l.0 as usize] = Some(ctx.append_basic_block(function, ""));
            }
        }
        let mut open = true;
        let snapshot = debug_id.map(|_| body.debug_frame(f)).transpose()?;
        for (index, inst) in f.instructions().iter().enumerate() {
            if let Instr::Label(l) = inst {
                let block = body.block(*l);
                if open {
                    builder.build_unconditional_branch(block)?;
                }
                builder.position_at_end(block);
                open = true;
            } else if open && let Instr::Debug(point) = inst {
                if let (Some(id), Some(snapshot)) = (debug_id, snapshot) {
                    body.debug_point(f, point, snapshot, id, index)?;
                }
            } else if open {
                body.instruction(inst, callees)?;
                open = !inst.terminates();
            }
        }
        // A dead block at the end of verified HIR may lack a terminator.
        if open {
            builder.build_unreachable()?;
        }
        Ok(body.static_rust_slice_operands())
    }

    fn block(&self, label: Label) -> BasicBlock<'ctx> {
        self.labels[label.0 as usize].expect("verified jump target")
    }

    pub(super) fn new_local(&mut self, allocas: &Builder<'ctx>, ty: ValueType) -> Built<Reg> {
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
            .push((ty, allocas.build_alloca(simd::llvm_type(self.ctx, ty), "")?));
        Ok(id)
    }

    pub(super) fn reg_type(&self, r: Reg) -> Option<ValueType> {
        self.registers.get(r.0 as usize).map(|slot| slot.0)
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

    pub(super) fn static_rust_slice_operands(&self) -> bool {
        self.static_rust_slice_operands
    }

    fn int(&self, n: i32) -> IntValue<'ctx> {
        self.ctx.i32_type().const_int(n as u32 as u64, false)
    }

    fn load(&self, reg: Reg) -> Built<IntValue<'ctx>> {
        Ok(self
            .builder
            .build_load(self.ctx.i32_type(), self.registers[reg.0 as usize].1, "")?
            .into_int_value())
    }

    fn store(&self, reg: Reg, value: IntValue<'ctx>) -> Built<()> {
        self.builder
            .build_store(self.registers[reg.0 as usize].1, value)?;
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
        self.checked_element(slice, self.ctx.i32_type().into(), index)
    }

    pub(super) fn trap_bounds(&self) -> Built<()> {
        self.builder
            .build_store(self.status, self.int(Failure::Bounds as i32))?;
        self.builder.build_return(Some(&self.int(0)))?;
        Ok(())
    }

    fn checked_element(
        &mut self,
        slice: Slice<'ctx>,
        element: BasicTypeEnum<'ctx>,
        index: IntValue<'ctx>,
    ) -> Built<PointerValue<'ctx>> {
        // The unsigned comparison also rejects negative signed indices, as every slice fits i32.
        let invalid = self
            .builder
            .build_int_compare(IntPredicate::UGE, index, slice.len, "")?;
        self.exit_if(invalid, Failure::Bounds)?;
        // SAFETY: Only the edge with index < slice.len reaches this instruction.
        unsafe {
            self.builder
                .build_in_bounds_gep(element, slice.data, &[index], "")
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
            CheckedAdd | CheckedSub | CheckedMul => {
                let wide_type = self.ctx.i64_type();
                let a = ir.build_int_s_extend(a, wide_type, "")?;
                let b = ir.build_int_s_extend(b, wide_type, "")?;
                let wide = match op {
                    CheckedAdd => ir.build_int_add(a, b, "")?,
                    CheckedSub => ir.build_int_sub(a, b, "")?,
                    CheckedMul => ir.build_int_mul(a, b, "")?,
                    _ => unreachable!(),
                };
                let result = ir.build_int_truncate(wide, self.ctx.i32_type(), "")?;
                let extended = ir.build_int_s_extend(result, wide_type, "")?;
                let invalid = ir.build_int_compare(IntPredicate::NE, wide, extended, "")?;
                self.exit_if(invalid, Failure::IntegerOverflow)?;
                Ok(result)
            }
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
            Shl | Shr | CheckedShl | CheckedShr => {
                let count = if matches!(op, CheckedShl | CheckedShr) {
                    let invalid = ir.build_int_compare(IntPredicate::UGE, b, self.int(32), "")?;
                    self.exit_if(invalid, Failure::IntegerOverflow)?;
                    b
                } else {
                    ir.build_and(b, self.int(31), "")?
                };
                if matches!(op, Shl | CheckedShl) {
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
        let (ty, ptr) = self.registers[src.0 as usize];
        let value = self
            .builder
            .build_load(simd::llvm_type(self.ctx, ty), ptr, "")?;
        self.builder
            .build_store(self.registers[dst.0 as usize].1, value)?;
        Ok(())
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

    pub(super) fn rust_call(
        &mut self,
        dst: Reg,
        import: splice::hir::RustCall,
        args: &[Reg],
    ) -> Built<()> {
        // The typed extern "C" entry remains resident in this process. Its
        // wrapper catches unwinding Rust panics before returning to the JIT.
        let wide = self.ctx.i64_type();
        let word = self.ctx.i32_type();
        let ty = wide.fn_type(&[word.into(), word.into()], false);
        let addr = wide
            .const_int((import.symbol.resolve)() as usize as u64, false)
            .const_to_pointer(self.ctx.ptr_type(inkwell::AddressSpace::default()));
        let mut words = [self.int(0); 2];
        for (out, &arg) in words.iter_mut().zip(args) {
            *out = self.load(arg)?;
        }
        let value = self
            .builder
            .build_indirect_call(ty, addr, &words.map(Into::into), "")?
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();
        self.rust_result(dst, value)
    }

    pub(super) fn rust_context_call(
        &mut self,
        dst: Reg,
        target: RustContextTarget,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> Built<()> {
        let wide = self.ctx.i64_type();
        let word = self.ctx.i32_type();
        let pointer = self.ctx.ptr_type(inkwell::AddressSpace::default());
        self.static_rust_slice_operands |=
            matches!(target, RustContextTarget::Static(_)) && !slices.is_empty();
        let (data, ordinal, entry, type_, first, second) =
            if let RustContextTarget::Borrowed { context, operation } = target
                && slices.is_empty()
                && args.len() <= 2
            {
                let mut words = [self.int(0); 2];
                for (word, &arg) in words.iter_mut().zip(args) {
                    *word = self.load(arg)?;
                }
                (
                    self.slices[context.index()].data,
                    operation,
                    splice::invoke_runtime_context as *const (),
                    word.into(),
                    words[0].into(),
                    words[1].into(),
                )
            } else {
                let packet = self.context_packet(args, slices)?;
                let (data, ordinal, entry) = match target {
                    RustContextTarget::Borrowed { context, operation } => (
                        self.slices[context.index()].data,
                        operation,
                        splice::invoke_packet_runtime_context as *const (),
                    ),
                    RustContextTarget::Static(symbol) => (packet, 0, symbol.entry() as *const ()),
                };
                (
                    data,
                    ordinal,
                    entry,
                    pointer.into(),
                    packet.into(),
                    word.const_zero().into(),
                )
            };
        let ty = wide.fn_type(&[pointer.into(), word.into(), type_, word.into()], false);
        let address = wide
            .const_int(entry as usize as u64, false)
            .const_to_pointer(pointer);
        let operands = [
            data.into(),
            word.const_int(u64::from(ordinal), false).into(),
            first,
            second,
        ];
        let value = self
            .builder
            .build_indirect_call(ty, address, &operands, "")?
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();
        self.rust_result(dst, value)
    }

    pub(super) fn rust_packet_call(
        &mut self,
        dst: Reg,
        import: &'static splice::hir::RustPacketSymbol,
        args: &[Reg],
    ) -> Built<()> {
        let packet = self.context_packet(args, &[])?;
        let pointer = self.ctx.ptr_type(inkwell::AddressSpace::default());
        let wide = self.ctx.i64_type();
        let word = self.ctx.i32_type();
        let address = |value: usize| {
            wide.const_int(value as u64, false)
                .const_to_pointer(pointer)
        };
        let ty = wide.fn_type(&[pointer.into(), pointer.into(), word.into()], false);
        let entry = address(splice_target::abi::invoke_packet_import as *const () as usize);
        let symbol = address(std::ptr::from_ref(import) as usize);
        let count = word.const_int(u64::from(import.arity), false);
        let value = self
            .builder
            .build_indirect_call(ty, entry, &[symbol.into(), packet.into(), count.into()], "")?
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();
        self.rust_result(dst, value)
    }

    fn context_packet(
        &mut self,
        args: &[Reg],
        slices: &[ContextSlice],
    ) -> Built<PointerValue<'ctx>> {
        let word = self.ctx.i32_type();
        let layout = ContextArguments::packet_layout(args.len() as u32, slices.len() as u32)
            .expect("validated Rust context packet");
        let packet = self.context_scratch(layout.bytes.max(8))?;
        let slot = |offset: u32| {
            // SAFETY: the shared alloca is aligned and large enough for the
            // verified packet. Only the initialized fields become Rust views.
            unsafe {
                self.builder.build_in_bounds_gep(
                    self.ctx.i8_type(),
                    packet,
                    &[word.const_int(u64::from(offset), false)],
                    "",
                )
            }
        };
        for (index, &arg) in args.iter().enumerate() {
            self.builder
                .build_store(slot(index as u32 * 4)?, self.load(arg)?)?;
        }
        for (index, &ContextSlice { buffer, writable }) in slices.iter().enumerate() {
            let slice = self.slices[buffer.index()];
            let len = self.builder.build_int_truncate(slice.len, word, "")?;
            let tag = word.const_int(
                u64::from(ContextArguments::view_tag(slice.element, writable)),
                false,
            );
            let offset = layout.slices_at + index as u32 * ContextArguments::SLICE_BYTES;
            self.builder.build_store(slot(offset)?, slice.data)?;
            self.builder
                .build_store(slot(offset + ContextArguments::LENGTH_OFFSET)?, len)?;
            self.builder
                .build_store(slot(offset + ContextArguments::ELEMENT_OFFSET)?, tag)?;
        }
        Ok(packet)
    }

    fn context_scratch(&mut self, bytes: u32) -> Built<PointerValue<'ctx>> {
        let wide = self.ctx.i64_type();
        let words = wide.const_int(u64::from(bytes.div_ceil(8)), false);
        if let Some((capacity, packet)) = &mut self.context_scratch {
            if bytes > *capacity {
                // There is one entry alloca. Each synchronous call uses the
                // storage until it returns, then the next call can reuse it.
                assert!(
                    packet
                        .as_instruction_value()
                        .expect("packet alloca")
                        .set_operand(0, words)
                );
                *capacity = bytes;
            }
            return Ok(*packet);
        }
        let allocas = self.ctx.create_builder();
        let entry = self
            .function
            .get_first_basic_block()
            .expect("kernel entry block");
        if let Some(first) = entry.get_first_instruction() {
            allocas.position_before(&first);
        } else {
            allocas.position_at_end(entry);
        }
        let packet = allocas.build_array_alloca(wide, words, "rust_context_packet")?;
        self.context_scratch = Some((bytes, packet));
        Ok(packet)
    }

    fn rust_result(&mut self, dst: Reg, value: IntValue<'ctx>) -> Built<()> {
        let max = self.ctx.i64_type().const_int(u64::from(u32::MAX), false);
        let failed = self
            .builder
            .build_int_compare(IntPredicate::UGT, value, max, "")?;
        self.exit_if(failed, Failure::RustPanic)?;
        self.store(
            dst,
            self.builder
                .build_int_truncate(value, self.ctx.i32_type(), "")?,
        )
    }

    fn instruction(&mut self, inst: &Instr, callees: &Callees<'_, 'ctx>) -> Built<()> {
        match *inst {
            Instr::Debug(_) => unreachable!("debug metadata is lowered at source points"),
            Instr::Simd {
                dst,
                inst,
                ref args,
            } => self.simd(dst, inst, args),
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
            Instr::TrapBounds => self.trap_bounds(),
            Instr::RustCall {
                dst,
                import,
                ref args,
            } => self.rust_call(dst, import, args),
            Instr::RustPacketCall {
                dst,
                import,
                ref args,
            } => self.rust_packet_call(dst, import, args),
            Instr::RustContextCall {
                dst,
                target,
                ref args,
                ref slices,
            } => self.rust_context_call(dst, target, args, slices),
            Instr::Call {
                dst,
                ref function,
                ref args,
                ref buffers,
            } => self.call_function(
                dst,
                callees.declarations[callees.ids[&Arc::as_ptr(function)]],
                function.needs_status(),
                args,
                buffers,
            ),
            Instr::Label(_) => unreachable!("handled at block boundary"),
        }
    }
}
