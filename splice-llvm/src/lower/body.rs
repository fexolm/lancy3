mod debug;
mod intrinsic;
mod simd;

pub(super) use simd::{int_type, scalar_type};

use super::call;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::types::BasicTypeEnum;
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FloatValue, FunctionValue, IntValue,
    PointerValue,
};
use inkwell::{FloatPredicate, IntPredicate, module::Module};
use splice_core::Trap;
use splice_core::hir::{
    Binary, BufferKind, Constant, Element, Label, PacketStore, PacketValue, Reg, RustCallPlan,
    ScalarType, Slice as SliceValue, ValueType,
};

type Built<T> = Result<T, Error>;

/// Failure to emit LLVM IR for a function body.
#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error(transparent)]
    Builder(#[from] BuilderError),
    /// The loaded LLVM does not provide this intrinsic for these types.
    #[error("LLVM does not provide {0}")]
    MissingIntrinsic(&'static str),
}

#[derive(Clone, Copy)]
struct Slice<'ctx> {
    data: PointerValue<'ctx>,
    len: IntValue<'ctx>,
    element: ScalarType,
}

/// A side exit: forward the status a failed callee already stored, or
/// store a new trap code.
#[derive(Clone, Copy, PartialEq)]
enum Failure {
    Propagate,
    Trap(Trap),
}

impl Failure {
    const COUNT: usize = Trap::ALL.len() + 1;

    fn index(self) -> usize {
        match self {
            Self::Propagate => 0,
            Self::Trap(trap) => trap.code() as usize,
        }
    }
}

/// The module that bodies are emitted into, for one build target.
#[derive(Clone, Copy)]
pub(super) struct Unit<'a, 'ctx> {
    pub ctx: &'ctx Context,
    pub module: &'a Module<'ctx>,
    pub machine: crate::target::Machine,
}

pub(super) struct Body<'a, 'ctx> {
    ctx: &'ctx Context,
    machine: crate::target::Machine,
    builder: &'a Builder<'ctx>,
    module: &'a Module<'ctx>,
    function: FunctionValue<'ctx>,
    status: PointerValue<'ctx>,
    registers: Vec<(ValueType, PointerValue<'ctx>)>,
    static_rust_slice_operands: bool,
    labels: Vec<Option<BasicBlock<'ctx>>>,
    exits: [Option<BasicBlock<'ctx>>; Failure::COUNT],
    /// A Rust call can report its own trap code.
    calls_rust: bool,
}

/// Declare intrinsic `name` overloaded on `types`, if the loaded LLVM has it.
fn intrinsic<'ctx>(
    module: &Module<'ctx>,
    name: &'static str,
    types: &[BasicTypeEnum<'ctx>],
) -> Built<FunctionValue<'ctx>> {
    inkwell::intrinsics::Intrinsic::find(name)
        .and_then(|intrinsic| intrinsic.get_declaration(module, types))
        .ok_or(Error::MissingIntrinsic(name))
}

impl<'a, 'ctx> Body<'a, 'ctx> {
    fn intrinsic(
        &self,
        name: &'static str,
        types: &[BasicTypeEnum<'ctx>],
    ) -> Built<FunctionValue<'ctx>> {
        intrinsic(self.module, name, types)
    }

    pub(super) fn start(
        unit: Unit<'a, 'ctx>,
        builder: &'a Builder<'ctx>,
        function: FunctionValue<'ctx>,
        arity: usize,
        types: impl IntoIterator<Item = ValueType>,
        kinds: &[BufferKind],
    ) -> Built<Self> {
        let Unit {
            ctx,
            module,
            machine,
        } = unit;
        builder.position_at_end(ctx.append_basic_block(function, "entry"));
        // Only the entry allocates register slots. LLVM promotes these to SSA,
        // including phi nodes for mutable locals across joins and backedges.
        let registers = types
            .into_iter()
            .map(|ty| {
                let llvm = simd::llvm_type(ctx, machine.vector_bytes, ty);
                Ok((ty, builder.build_alloca(llvm, "")?))
            })
            .collect::<Built<Vec<_>>>()?;
        let mut params = function.get_param_iter();
        let status = params.next().unwrap().into_pointer_value();
        // A buffer parameter is the slice of its two registers, which follow
        // the scalars: the data pointer and the `Index` length.
        for &(_, ptr) in &registers[..arity + 2 * kinds.len()] {
            builder.build_store(ptr, params.next().unwrap())?;
        }
        Ok(Self {
            ctx,
            machine,
            builder,
            module,
            function,
            status,
            registers,
            static_rust_slice_operands: false,
            labels: Vec::new(),
            exits: [None; Failure::COUNT],
            calls_rust: false,
        })
    }

    fn block(&self, label: Label) -> BasicBlock<'ctx> {
        self.labels[label.0 as usize].expect("verified jump target")
    }

    /// A second builder can insert slots in the entry even while emitting
    /// code for a loop or a guarded continuation in a different block.
    fn position_in_entry(&self, allocas: &Builder<'ctx>) {
        let entry = self.function.get_first_basic_block().unwrap();
        if let Some(first) = entry.get_first_instruction() {
            allocas.position_before(&first);
        } else {
            allocas.position_at_end(entry);
        }
    }

    pub(super) fn new_local(&mut self, allocas: &Builder<'ctx>, ty: ValueType) -> Built<Reg> {
        self.position_in_entry(allocas);
        let id = Reg(self.registers.len() as u32);
        self.registers.push((
            ty,
            allocas.build_alloca(simd::llvm_type(self.ctx, self.machine.vector_bytes, ty), "")?,
        ));
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
        self.builder.build_unreachable()?;
        Ok(())
    }
    pub(super) fn needs_status(&self) -> bool {
        self.calls_rust || self.exits.iter().any(Option::is_some)
    }

    pub(super) fn static_rust_slice_operands(&self) -> bool {
        self.static_rust_slice_operands
    }

    fn int(&self, n: i32) -> IntValue<'ctx> {
        self.ctx.i32_type().const_int(n as u32 as u64, false)
    }

    fn scalar_type(&self, reg: Reg) -> ScalarType {
        match self.registers[reg.0 as usize].0 {
            ValueType::Scalar(ty) => ty,
            _ => unreachable!("validated scalar register"),
        }
    }

    fn value(&self, reg: Reg) -> Built<BasicValueEnum<'ctx>> {
        let (ty, ptr) = self.registers[reg.0 as usize];
        let ty = simd::llvm_type(self.ctx, self.machine.vector_bytes, ty);
        Ok(self.builder.build_load(ty, ptr, "")?)
    }

    /// An integer or `bool` register.
    fn load(&self, reg: Reg) -> Built<IntValue<'ctx>> {
        Ok(self.value(reg)?.into_int_value())
    }

    fn store(&self, reg: Reg, value: impl BasicValue<'ctx>) -> Built<()> {
        self.builder
            .build_store(self.registers[reg.0 as usize].1, value)?;
        Ok(())
    }

    /// A `usize` memory offset as `i64`, clamped to `i64::MAX`: every slice
    /// length is below it (a Rust slice spans at most `isize::MAX` bytes),
    /// so a larger offset is out of bounds either way and adding a lane
    /// count does not wrap.
    fn offset(&self, reg: Reg) -> Built<IntValue<'ctx>> {
        let wide = self.ctx.i64_type();
        let umin = self.intrinsic("llvm.umin", &[wide.into()])?;
        let max = wide.const_int(i64::MAX as u64, false);
        Ok(self
            .builder
            .build_call(umin, &[self.load(reg)?.into(), max.into()], "")?
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value())
    }

    /// The result of a function that stops with a status.
    fn failed_result(&self) -> BasicValueEnum<'ctx> {
        self.function
            .get_type()
            .get_return_type()
            .expect("kernels return a value")
            .const_zero()
    }

    // Guards of the same type share one side exit. Failed calls preserve
    // their callee's status and return immediately to the enclosing caller.
    fn exit_if(&mut self, condition: IntValue<'ctx>, fault: Failure) -> Built<()> {
        let exit = self.exit(fault)?;
        let next = self.ctx.append_basic_block(self.function, "");
        self.builder
            .build_conditional_branch(condition, exit, next)?;
        self.builder.position_at_end(next);
        Ok(())
    }

    /// The side exit for `fault`, created on first use.
    fn exit(&mut self, fault: Failure) -> Built<BasicBlock<'ctx>> {
        if let Some(exit) = self.exits[fault.index()] {
            return Ok(exit);
        }
        let here = self.builder.get_insert_block().expect("positioned builder");
        let exit = self.ctx.append_basic_block(self.function, "error");
        self.exits[fault.index()] = Some(exit);
        self.builder.position_at_end(exit);
        if let Failure::Trap(trap) = fault {
            self.builder
                .build_store(self.status, self.int(trap.code() as i32))?;
        }
        self.builder.build_return(Some(&self.failed_result()))?;
        self.builder.position_at_end(here);
        Ok(exit)
    }

    /// The pointer, length and element of slice `s`.
    fn slice(&self, s: SliceValue) -> Built<Slice<'ctx>> {
        let ValueType::Ptr(kind) = self.registers[s.ptr.0 as usize].0 else {
            unreachable!("validated slice");
        };
        Ok(Slice {
            data: self.value(s.ptr)?.into_pointer_value(),
            len: self.load(s.len)?,
            element: kind.scalar().expect("a validated slice of scalars"),
        })
    }

    fn element(&mut self, s: SliceValue, index: Reg) -> Built<PointerValue<'ctx>> {
        let slice = self.slice(s)?;
        let ty = slice.element;
        let ty = simd::scalar_type(self.ctx, ty);
        self.checked_element(slice, ty, self.load(index)?)
    }

    pub(super) fn trap_bounds(&self) -> Built<()> {
        self.builder
            .build_store(self.status, self.int(Trap::Bounds.code() as i32))?;
        self.builder.build_return(Some(&self.failed_result()))?;
        Ok(())
    }

    fn checked_element(
        &mut self,
        slice: Slice<'ctx>,
        element: BasicTypeEnum<'ctx>,
        index: IntValue<'ctx>,
    ) -> Built<PointerValue<'ctx>> {
        let invalid = self
            .builder
            .build_int_compare(IntPredicate::UGE, index, slice.len, "")?;
        self.exit_if(invalid, Failure::Trap(Trap::Bounds))?;
        // SAFETY: Only the edge with index < slice.len reaches this instruction.
        Ok(unsafe {
            self.builder
                .build_in_bounds_gep(element, slice.data, &[index], "")?
        })
    }

    fn binary(
        &mut self,
        op: Binary,
        ty: ScalarType,
        a: BasicValueEnum<'ctx>,
        b: BasicValueEnum<'ctx>,
    ) -> Built<BasicValueEnum<'ctx>> {
        if ty.is_float() {
            return self.float_binary(op, a.into_float_value(), b.into_float_value());
        }
        let (a, b) = (a.into_int_value(), b.into_int_value());
        let ir = self.builder;
        let int = a.get_type();
        let signed = ty.is_signed();
        let constant = |n: i64| int.const_int(n as u64, signed);
        use Binary::*;
        Ok(match op {
            Add => ir.build_int_add(a, b, "")?.into(),
            Sub => ir.build_int_sub(a, b, "")?.into(),
            Mul => ir.build_int_mul(a, b, "")?.into(),
            CheckedAdd | CheckedSub | CheckedMul => {
                let name = match (op, signed) {
                    (CheckedAdd, true) => "llvm.sadd.with.overflow",
                    (CheckedAdd, false) => "llvm.uadd.with.overflow",
                    (CheckedSub, true) => "llvm.ssub.with.overflow",
                    (CheckedSub, false) => "llvm.usub.with.overflow",
                    (_, true) => "llvm.smul.with.overflow",
                    (_, false) => "llvm.umul.with.overflow",
                };
                let function = self.intrinsic(name, &[int.into()])?;
                let pair = ir
                    .build_call(function, &[a.into(), b.into()], "")?
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_struct_value();
                let result = ir.build_extract_value(pair, 0, "")?;
                let overflow = ir.build_extract_value(pair, 1, "")?.into_int_value();
                self.exit_if(overflow, Failure::Trap(Trap::Overflow))?;
                result
            }
            SaturatingAdd | SaturatingSub => {
                let name = match (op, signed) {
                    (SaturatingAdd, true) => "llvm.sadd.sat",
                    (SaturatingAdd, false) => "llvm.uadd.sat",
                    (_, true) => "llvm.ssub.sat",
                    (_, false) => "llvm.usub.sat",
                };
                let function = self.intrinsic(name, &[int.into()])?;
                ir.build_call(function, &[a.into(), b.into()], "")?
                    .try_as_basic_value()
                    .unwrap_basic()
            }
            SaturatingMul => {
                let name = if signed {
                    "llvm.smul.with.overflow"
                } else {
                    "llvm.umul.with.overflow"
                };
                let function = self.intrinsic(name, &[int.into()])?;
                let pair = ir
                    .build_call(function, &[a.into(), b.into()], "")?
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_struct_value();
                let product = ir.build_extract_value(pair, 0, "")?.into_int_value();
                let overflow = ir.build_extract_value(pair, 1, "")?.into_int_value();
                let saturated = if signed {
                    // A product leaves the range on the side of its sign.
                    let sign = ir.build_xor(a, b, "")?;
                    let negative =
                        ir.build_int_compare(IntPredicate::SLT, sign, int.const_zero(), "")?;
                    let min = constant(i64::MIN >> (64 - ty.bits()));
                    let max = constant(i64::MAX >> (64 - ty.bits()));
                    ir.build_select(negative, min, max, "")?.into_int_value()
                } else {
                    int.const_all_ones()
                };
                ir.build_select(overflow, saturated, product, "")?
            }
            And => ir.build_and(a, b, "")?.into(),
            Or => ir.build_or(a, b, "")?.into(),
            Xor => ir.build_xor(a, b, "")?.into(),
            Div | Rem => {
                let zero = ir.build_int_compare(IntPredicate::EQ, b, int.const_zero(), "")?;
                self.exit_if(zero, Failure::Trap(Trap::DivisionByZero))?;
                if signed {
                    let min = constant(i64::MIN >> (64 - ty.bits()));
                    let min = ir.build_int_compare(IntPredicate::EQ, a, min, "")?;
                    let minus_one =
                        ir.build_int_compare(IntPredicate::EQ, b, int.const_all_ones(), "")?;
                    self.exit_if(
                        ir.build_and(min, minus_one, "")?,
                        Failure::Trap(Trap::DivisionOverflow),
                    )?;
                }
                match (op, signed) {
                    (Div, true) => ir.build_int_signed_div(a, b, "")?,
                    (Div, false) => ir.build_int_unsigned_div(a, b, "")?,
                    (_, true) => ir.build_int_signed_rem(a, b, "")?,
                    (_, false) => ir.build_int_unsigned_rem(a, b, "")?,
                }
                .into()
            }
            Shl | Shr | CheckedShl | CheckedShr => {
                let width = constant(i64::from(ty.bits()));
                let count = if matches!(op, CheckedShl | CheckedShr) {
                    let invalid = ir.build_int_compare(IntPredicate::UGE, b, width, "")?;
                    self.exit_if(invalid, Failure::Trap(Trap::Overflow))?;
                    b
                } else {
                    ir.build_and(b, constant(i64::from(ty.bits() - 1)), "")?
                };
                if matches!(op, Shl | CheckedShl) {
                    ir.build_left_shift(a, count, "")?.into()
                } else {
                    ir.build_right_shift(a, count, signed, "")?.into()
                }
            }
            Min | Max => {
                let pred = match (op, signed) {
                    (Min, true) => IntPredicate::SLT,
                    (Min, false) => IntPredicate::ULT,
                    (_, true) => IntPredicate::SGT,
                    (_, false) => IntPredicate::UGT,
                };
                let take_a = ir.build_int_compare(pred, a, b, "")?;
                ir.build_select(take_a, a, b, "")?
            }
            Eq | Ne | Lt | Le | Gt | Ge => {
                let pred = match (op, signed) {
                    (Eq, _) => IntPredicate::EQ,
                    (Ne, _) => IntPredicate::NE,
                    (Lt, true) => IntPredicate::SLT,
                    (Lt, false) => IntPredicate::ULT,
                    (Le, true) => IntPredicate::SLE,
                    (Le, false) => IntPredicate::ULE,
                    (Gt, true) => IntPredicate::SGT,
                    (Gt, false) => IntPredicate::UGT,
                    (_, true) => IntPredicate::SGE,
                    (_, false) => IntPredicate::UGE,
                };
                ir.build_int_compare(pred, a, b, "")?.into()
            }
        })
    }

    fn float_binary(
        &self,
        op: Binary,
        a: FloatValue<'ctx>,
        b: FloatValue<'ctx>,
    ) -> Built<BasicValueEnum<'ctx>> {
        let ir = self.builder;
        use Binary::*;
        Ok(match op {
            Add => ir.build_float_add(a, b, "")?.into(),
            Sub => ir.build_float_sub(a, b, "")?.into(),
            Mul => ir.build_float_mul(a, b, "")?.into(),
            Div => ir.build_float_div(a, b, "")?.into(),
            Min | Max => {
                // minimumNumber: llvm.minimum orders the zeros; a NaN
                // operand yields the other operand.
                let name = if op == Min {
                    "llvm.minimum"
                } else {
                    "llvm.maximum"
                };
                let function = self.intrinsic(name, &[a.get_type().into()])?;
                let both = ir
                    .build_call(function, &[a.into(), b.into()], "")?
                    .try_as_basic_value()
                    .unwrap_basic();
                let a_nan = ir.build_float_compare(FloatPredicate::UNO, a, a, "")?;
                let b_nan = ir.build_float_compare(FloatPredicate::UNO, b, b, "")?;
                let other = ir.build_select(a_nan, b, both.into_float_value(), "")?;
                ir.build_select(b_nan, a.into(), other, "")?
            }
            Eq | Ne | Lt | Le | Gt | Ge => {
                let pred = match op {
                    Eq => FloatPredicate::OEQ,
                    Ne => FloatPredicate::UNE,
                    Lt => FloatPredicate::OLT,
                    Le => FloatPredicate::OLE,
                    Gt => FloatPredicate::OGT,
                    _ => FloatPredicate::OGE,
                };
                ir.build_float_compare(pred, a, b, "")?.into()
            }
            _ => unreachable!("validated float operation"),
        })
    }

    /// Rust's `value as to`.
    fn convert(
        &self,
        value: BasicValueEnum<'ctx>,
        from: ScalarType,
        to: ScalarType,
    ) -> Built<BasicValueEnum<'ctx>> {
        let ir = self.builder;
        let target = simd::scalar_type(self.ctx, to);
        Ok(match (from.is_float(), to.is_float()) {
            _ if from == to => value,
            (false, false) => {
                let value = value.into_int_value();
                let target = target.into_int_type();
                let (bits, wide) = (from.bits(), to.bits());
                if bits > wide {
                    ir.build_int_truncate(value, target, "")?.into()
                } else if bits == wide {
                    value.into()
                } else if from.is_signed() {
                    ir.build_int_s_extend(value, target, "")?.into()
                } else {
                    ir.build_int_z_extend(value, target, "")?.into()
                }
            }
            (false, true) if from.is_signed() => ir
                .build_signed_int_to_float(value.into_int_value(), target.into_float_type(), "")?
                .into(),
            (false, true) => ir
                .build_unsigned_int_to_float(value.into_int_value(), target.into_float_type(), "")?
                .into(),
            (true, true) => ir
                .build_float_cast(value.into_float_value(), target.into_float_type(), "")?
                .into(),
            (true, false) => {
                let name = if to.is_signed() {
                    "llvm.fptosi.sat"
                } else {
                    "llvm.fptoui.sat"
                };
                let function = self.intrinsic(name, &[target, value.get_type()])?;
                ir.build_call(function, &[value.into()], "")?
                    .try_as_basic_value()
                    .unwrap_basic()
            }
        })
    }

    pub(super) fn constant(&self, dst: Reg, c: Constant) -> Built<()> {
        let width = std::num::NonZero::new(c.ty().bits()).expect("a nonzero width");
        let bits = self.ctx.custom_width_int_type(width).unwrap();
        let bits = bits.const_int(c.bits(), false);
        let ty = simd::scalar_type(self.ctx, c.ty());
        // The builder folds a cast of a constant to a constant.
        let value = self.builder.build_bit_cast(bits, ty, "")?;
        self.store(dst, value)
    }
    pub(super) fn cast(&self, dst: Reg, src: Reg, to: ScalarType) -> Built<()> {
        let value = self.convert(self.value(src)?, self.scalar_type(src), to)?;
        self.store(dst, value)
    }
    pub(super) fn assign(&self, dst: Reg, src: Reg) -> Built<()> {
        let (ty, ptr) = self.registers[src.0 as usize];
        let value = self.builder.build_load(
            simd::llvm_type(self.ctx, self.machine.vector_bytes, ty),
            ptr,
            "",
        )?;
        self.builder
            .build_store(self.registers[dst.0 as usize].1, value)?;
        Ok(())
    }
    pub(super) fn arithmetic(&mut self, dst: Reg, op: Binary, a: Reg, b: Reg) -> Built<()> {
        let ty = self.scalar_type(a);
        let result = self.binary(op, ty, self.value(a)?, self.value(b)?)?;
        self.store(dst, result)
    }
    /// `dst = &slice[s..e]` for `e = min(end, len)` and `s = min(start, e)`
    /// (see `KernelBuilder::slice_range`).
    pub(super) fn slice_range(
        &mut self,
        dst: SliceValue,
        slice: SliceValue,
        start: Reg,
        end: Reg,
    ) -> Built<()> {
        let ValueType::Ptr(kind) = self.registers[slice.ptr.0 as usize].0 else {
            unreachable!("validated slice pointer");
        };
        let index = ScalarType::Index;
        let len = self.value(slice.len)?;
        let end = if end == slice.len {
            len
        } else {
            self.binary(Binary::Min, index, self.value(end)?, len)?
        };
        let start = self.binary(Binary::Min, index, self.value(start)?, end)?;
        let element = self.element_type(kind.element());
        let base = self.value(slice.ptr)?.into_pointer_value();
        // SAFETY: `start <= len`, so the address is in the allocation of
        // `slice` or one past it; the verifier guarantees that `slice.len`
        // is the length of the memory `slice.ptr` addresses.
        let address = unsafe {
            self.builder
                .build_in_bounds_gep(element, base, &[start.into_int_value()], "")?
        };
        self.store(dst.ptr, address)?;
        let length = self.binary(Binary::Sub, index, end, start)?;
        self.store(dst.len, length)
    }
    /// The LLVM type of one element: a scalar, or a slice element laid
    /// out as `SliceDescriptor`, `{ ptr, i64 }`.
    fn element_type(&self, element: Element) -> BasicTypeEnum<'ctx> {
        match element {
            Element::Scalar(ty) => simd::scalar_type(self.ctx, ty),
            Element::Slice(_) => {
                let ptr = self.ctx.ptr_type(inkwell::AddressSpace::default());
                let len = self.ctx.i64_type();
                self.ctx
                    .struct_type(&[ptr.into(), len.into()], false)
                    .into()
            }
            // Bytes: the frontend addresses one scalar field at a time.
            Element::Struct(ty) => self.ctx.i8_type().array_type(ty.size as u32).into(),
        }
    }
    /// The address of scalar `field` of struct element `index` of `s`,
    /// after a bounds check, and the field's type.
    fn field(
        &mut self,
        s: SliceValue,
        index: Reg,
        field: usize,
    ) -> Built<(PointerValue<'ctx>, ScalarType)> {
        let ValueType::Ptr(kind) = self.registers[s.ptr.0 as usize].0 else {
            unreachable!("validated slice pointer");
        };
        let (offset, ty) = kind.element().field(field).expect("a validated field");
        let element = self.element_type(kind.element());
        let (data, len) = (self.value(s.ptr)?.into_pointer_value(), self.load(s.len)?);
        let index = self.load(index)?;
        let invalid = self
            .builder
            .build_int_compare(IntPredicate::UGE, index, len, "")?;
        self.exit_if(invalid, Failure::Trap(Trap::Bounds))?;
        let offset = self.ctx.i64_type().const_int(offset as u64, false);
        // SAFETY: only the edge with `index < len` reaches the address, the
        // verifier guarantees that `len` is the length at `data`, and
        // checked the field to lie within its element.
        let at = unsafe {
            let at = self
                .builder
                .build_in_bounds_gep(element, data, &[index], "")?;
            self.builder
                .build_in_bounds_gep(self.ctx.i8_type(), at, &[offset], "")?
        };
        Ok((at, ty))
    }
    pub(super) fn read_field(
        &mut self,
        dst: Reg,
        s: SliceValue,
        index: Reg,
        field: usize,
    ) -> Built<()> {
        let (at, ty) = self.field(s, index, field)?;
        let value = if ty == ScalarType::Bool {
            // A `bool` is a byte that holds 0 or 1; an `i1` in memory would
            // leave the bits above it unspecified.
            let byte = self.builder.build_load(self.ctx.i8_type(), at, "")?;
            self.builder
                .build_int_truncate(byte.into_int_value(), self.ctx.bool_type(), "")?
                .into()
        } else {
            self.builder
                .build_load(simd::scalar_type(self.ctx, ty), at, "")?
        };
        self.store(dst, value)
    }
    pub(super) fn write_field(
        &mut self,
        s: SliceValue,
        index: Reg,
        field: usize,
        value: Reg,
    ) -> Built<()> {
        let (at, ty) = self.field(s, index, field)?;
        let value = self.value(value)?;
        if ty == ScalarType::Bool {
            let byte =
                self.builder
                    .build_int_z_extend(value.into_int_value(), self.ctx.i8_type(), "")?;
            self.builder.build_store(at, byte)?;
        } else {
            self.builder.build_store(at, value)?;
        }
        Ok(())
    }
    /// `dst = s[index]` of a slice of slices, after a bounds check.
    pub(super) fn load_slice(&mut self, dst: SliceValue, s: SliceValue, index: Reg) -> Built<()> {
        let ValueType::Ptr(kind) = self.registers[s.ptr.0 as usize].0 else {
            unreachable!("validated slice pointer");
        };
        let element = self.element_type(kind.element());
        let (data, len) = (self.value(s.ptr)?.into_pointer_value(), self.load(s.len)?);
        let index = self.load(index)?;
        let invalid = self
            .builder
            .build_int_compare(IntPredicate::UGE, index, len, "")?;
        self.exit_if(invalid, Failure::Trap(Trap::Bounds))?;
        // SAFETY: only the edge with `index < len` reaches the address, and
        // the verifier guarantees that `len` is the length at `data`.
        let at = unsafe {
            self.builder
                .build_in_bounds_gep(element, data, &[index], "")?
        };
        let ptr = self.ctx.ptr_type(inkwell::AddressSpace::default());
        let inner = self.builder.build_load(ptr, at, "")?;
        self.store(dst.ptr, inner)?;
        const { assert!(Element::SLICE_LEN_OFFSET == 8) };
        let at = self.builder.build_struct_gep(element, at, 1, "")?;
        let count = self.builder.build_load(self.ctx.i64_type(), at, "")?;
        self.store(dst.len, count)
    }
    pub(super) fn read_element(&mut self, dst: Reg, s: SliceValue, index: Reg) -> Built<()> {
        let ptr = self.element(s, index)?;
        let ty = simd::scalar_type(self.ctx, self.scalar_type(dst));
        let value = self.builder.build_load(ty, ptr, "")?;
        self.store(dst, value)
    }
    pub(super) fn write_element(&mut self, s: SliceValue, index: Reg, value: Reg) -> Built<()> {
        let ptr = self.element(s, index)?;
        self.builder.build_store(ptr, self.value(value)?)?;
        Ok(())
    }
    pub(super) fn jump(&self, label: Label) -> Built<()> {
        self.builder.build_unconditional_branch(self.block(label))?;
        Ok(())
    }
    pub(super) fn branch(&self, cond: Reg, yes: Label, no: Label) -> Built<()> {
        self.builder
            .build_conditional_branch(self.load(cond)?, self.block(yes), self.block(no))?;
        Ok(())
    }
    pub(super) fn ret(&self, value: Reg) -> Built<()> {
        self.builder.build_return(Some(&self.value(value)?))?;
        Ok(())
    }
    pub(super) fn call_function(
        &mut self,
        dst: Reg,
        callee: FunctionValue<'ctx>,
        needs_status: bool,
        args: &[Reg],
        buffers: &[SliceValue],
    ) -> Built<()> {
        let ir = self.builder;
        let mut actuals =
            Vec::<BasicMetadataValueEnum>::with_capacity(callee.count_params() as usize);
        actuals.push(self.status.into());
        for &r in args {
            actuals.push(self.value(r)?.into());
        }
        for &s in buffers {
            actuals.push(self.value(s.ptr)?.into());
            actuals.push(self.value(s.len)?.into());
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

    /// A Rust call as [`splice_core::hir::RustTarget::plan`] lays it out.
    pub(super) fn rust_call(&mut self, dst: Reg, plan: RustCallPlan, args: &[Reg]) -> Built<()> {
        let wide = self.ctx.i64_type();
        let word = self.ctx.i32_type();
        let pointer = self.ctx.ptr_type(inkwell::AddressSpace::default());
        // A static Rust function may keep a slice address for its lifetime.
        self.static_rust_slice_operands |= plan.context.is_none()
            && plan
                .stores
                .iter()
                .any(|store| matches!(store.value, PacketValue::Data(_)));
        let packet = self.packet(plan.bytes)?;
        for &PacketStore { at, value } in &plan.stores {
            let value: BasicValueEnum<'ctx> = match value {
                PacketValue::Word(n) => self.word(args[n])?.into(),
                PacketValue::Data(ptr) => self.value(ptr)?,
                PacketValue::Length(len) => self.load(len)?.into(),
                PacketValue::Tag(tag) => word.const_int(u64::from(tag), false).into(),
                PacketValue::Symbol(address) => wide.const_int(address as u64, false).into(),
            };
            // SAFETY: the packet alloca holds `plan.bytes`, which covers
            // every store of the plan.
            let slot = unsafe {
                self.builder.build_in_bounds_gep(
                    self.ctx.i8_type(),
                    packet,
                    &[word.const_int(u64::from(at), false)],
                    "",
                )?
            };
            self.builder.build_store(slot, value)?;
        }
        let receiver = match plan.context {
            Some(context) => self.value(context.ptr)?.into_pointer_value(),
            None => packet,
        };
        let ty = wide.fn_type(
            &[pointer.into(), word.into(), pointer.into(), word.into()],
            false,
        );
        let entry = wide
            .const_int(plan.entry.entry as usize as u64, false)
            .const_to_pointer(pointer);
        let operands = [
            receiver.into(),
            word.const_int(u64::from(plan.operation), false).into(),
            packet.into(),
            word.const_zero().into(),
        ];
        let status = self
            .builder
            .build_indirect_call(ty, entry, &operands, "")?
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();
        self.rust_status(status)?;
        let (result, _) = self.registers[dst.0 as usize];
        let result = simd::llvm_type(self.ctx, self.machine.vector_bytes, result);
        let value = self.builder.build_load(result, packet, "")?;
        self.store(dst, value)
    }

    /// A scalar register as the 8-byte word of a Rust call operand: its
    /// bits zero-extended.
    fn word(&self, reg: Reg) -> Built<IntValue<'ctx>> {
        let wide = self.ctx.i64_type();
        let value = self.value(reg)?;
        let bits = match self.scalar_type(reg) {
            ScalarType::F32 => self
                .builder
                .build_bit_cast(value, self.ctx.i32_type(), "")?
                .into_int_value(),
            ScalarType::F64 => {
                return Ok(self
                    .builder
                    .build_bit_cast(value, wide, "")?
                    .into_int_value());
            }
            _ => value.into_int_value(),
        };
        Ok(self
            .builder
            .build_int_z_extend_or_bit_cast(bits, wide, "")?)
    }

    /// An 8-byte aligned packet of `bytes` for one Rust call, allocated in
    /// the entry block so that LLVM sees a fixed frame slot.
    fn packet(&self, bytes: u32) -> Built<PointerValue<'ctx>> {
        let wide = self.ctx.i64_type();
        let allocas = self.ctx.create_builder();
        let entry = self
            .function
            .get_first_basic_block()
            .expect("kernel entry block");
        match entry.get_first_instruction() {
            Some(first) => allocas.position_before(&first),
            None => allocas.position_at_end(entry),
        }
        Ok(allocas.build_alloca(wide.array_type(bytes.div_ceil(8)), "rust_packet")?)
    }

    /// Report the trap code in the upper word of a Rust call's status.
    fn rust_status(&mut self, value: IntValue<'ctx>) -> Built<()> {
        self.calls_rust = true;
        let wide = self.ctx.i64_type();
        let max = wide.const_int(u64::from(u32::MAX), false);
        let failed = self
            .builder
            .build_int_compare(IntPredicate::UGT, value, max, "")?;
        let exit = self.ctx.append_basic_block(self.function, "rust_error");
        let next = self.ctx.append_basic_block(self.function, "");
        self.builder.build_conditional_branch(failed, exit, next)?;
        self.builder.position_at_end(exit);
        let code = self
            .builder
            .build_right_shift(value, wide.const_int(32, false), false, "")?;
        let code = self
            .builder
            .build_int_truncate(code, self.ctx.i32_type(), "")?;
        self.builder.build_store(self.status, code)?;
        self.builder.build_return(Some(&self.failed_result()))?;
        self.builder.position_at_end(next);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, intrinsic};
    use inkwell::context::Context;

    #[test]
    fn missing_intrinsics_are_build_errors() {
        let ctx = Context::create();
        let module = ctx.create_module("intrinsics");
        let vector = ctx.i32_type().vec_type(4).into();
        assert!(intrinsic(&module, "llvm.umax", &[vector]).is_ok());
        let missing = intrinsic(&module, "llvm.splice.missing", &[vector]);
        assert!(matches!(
            missing,
            Err(Error::MissingIntrinsic("llvm.splice.missing"))
        ));
        assert_eq!(
            missing.unwrap_err().to_string(),
            "LLVM does not provide llvm.splice.missing"
        );
    }
}
