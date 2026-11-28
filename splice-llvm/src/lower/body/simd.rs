//! LLVM vectors have a concrete width once the JIT target is known. Mask values
//! use one i1 per data lane, so they feed LLVM comparisons and selects directly.
use super::{Body, Built, Error, Failure};
use inkwell::{
    FloatPredicate, IntPredicate,
    context::Context,
    types::IntType,
    types::{BasicTypeEnum, VectorType},
    values::{BasicMetadataValueEnum, BasicValueEnum, IntValue, VectorValue},
};
use splice_core::{
    hir::{
        Buffer, Reg, ScalarType, Simd, SimdOp, ValueType,
        memory::{self, Address, BeforeTrap, Masked, Slice},
    },
    simd::Element,
};

fn element_type(ctx: &Context, e: Element) -> BasicTypeEnum<'_> {
    match e {
        Element::U16 => ctx.i16_type().into(),
        Element::I32 | Element::U32 => ctx.i32_type().into(),
        Element::I64 => ctx.i64_type().into(),
        Element::F32 => ctx.f32_type().into(),
        Element::F64 => ctx.f64_type().into(),
    }
}

fn vector_type(ctx: &Context, bytes: usize, e: Element) -> VectorType<'_> {
    vector_type_with_lanes(ctx, e, e.lanes(bytes) as u32)
}

fn vector_type_with_lanes(ctx: &Context, e: Element, lanes: u32) -> VectorType<'_> {
    match element_type(ctx, e) {
        BasicTypeEnum::IntType(t) => t.vec_type(lanes),
        BasicTypeEnum::FloatType(t) => t.vec_type(lanes),
        _ => unreachable!(),
    }
}

/// The LLVM type of a scalar register: a `bool` is an `i1`.
pub(in crate::lower) fn scalar_type(ctx: &Context, ty: ScalarType) -> BasicTypeEnum<'_> {
    match ty {
        ScalarType::Bool => ctx.bool_type().into(),
        ScalarType::I32 | ScalarType::U32 => ctx.i32_type().into(),
        ScalarType::I64 | ScalarType::U64 | ScalarType::Index => ctx.i64_type().into(),
        ScalarType::F32 => ctx.f32_type().into(),
        ScalarType::F64 => ctx.f64_type().into(),
    }
}

pub(super) fn llvm_type(ctx: &Context, bytes: usize, ty: ValueType) -> BasicTypeEnum<'_> {
    match ty {
        ValueType::Scalar(ty) => scalar_type(ctx, ty),
        ValueType::Vector(e) => vector_type(ctx, bytes, e).into(),
        ValueType::Mask(e) => ctx.bool_type().vec_type(e.lanes(bytes) as u32).into(),
    }
}

impl<'ctx> Body<'_, 'ctx> {
    fn convert_lanes(
        &self,
        x: VectorValue<'ctx>,
        from: Element,
        to: Element,
    ) -> Built<VectorValue<'ctx>> {
        let ir = self.builder;
        let ty = vector_type_with_lanes(self.ctx, to, x.get_type().get_size());
        Ok(match (from.is_float(), to.is_float()) {
            (true, true) if from == to => x,
            (true, true) => ir.build_float_cast(x, ty, "")?,
            (false, false) if from.bytes() == to.bytes() => x,
            (false, false) if from.bytes() > to.bytes() => ir.build_int_truncate(x, ty, "")?,
            (false, false) if from.is_unsigned() => ir.build_int_z_extend(x, ty, "")?,
            (false, false) => ir.build_int_s_extend(x, ty, "")?,
            (false, true) if from.is_unsigned() => ir.build_unsigned_int_to_float(x, ty, "")?,
            (false, true) => ir.build_signed_int_to_float(x, ty, "")?,
            (true, false) if matches!(from, Element::F32 | Element::F64) => {
                self.saturating_float_to_int(x, to, ty)?
            }
            (true, false) => {
                let name = if to.is_unsigned() {
                    "llvm.fptoui.sat"
                } else {
                    "llvm.fptosi.sat"
                };
                let function = self.intrinsic(name, &[ty.into(), x.get_type().into()])?;
                ir.build_call(function, &[x.into()], "")?
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_vector_value()
            }
        })
    }

    /// `fptosi.sat` and `fptoui.sat` on vectors, which LLVM 19 scalarizes on
    /// x86-64, as compares and selects around a plain conversion: a lane
    /// below the integer range converts its lowest value, a lane at or
    /// above `2^k` takes the largest value, NaN takes zero. A lane whose
    /// plain conversion is poison is always replaced by a select.
    fn saturating_float_to_int(
        &self,
        x: VectorValue<'ctx>,
        to: Element,
        ty: VectorType<'ctx>,
    ) -> Built<VectorValue<'ctx>> {
        let ir = self.builder;
        let lanes = x.get_type().get_size() as usize;
        let float = x.get_type().get_element_type().into_float_type();
        let int = ty.get_element_type().into_int_type();
        let floats = |v: f64| VectorType::const_vector(&vec![float.const_float(v); lanes]);
        let bits = to.bytes() as i32 * 8;
        let signed = !to.is_unsigned();
        // Integers in `lowest..2^k` are the range; both bounds are powers of
        // two, so exact in either float type.
        let k = if signed { bits - 1 } else { bits };
        let lowest = if signed { -(2f64.powi(k)) } else { 0.0 };
        let largest = if signed {
            (1u64 << (bits - 1)) - 1
        } else {
            u64::MAX >> (64 - bits)
        };
        let below = ir.build_float_compare(FloatPredicate::OLT, x, floats(lowest), "")?;
        let clamped = ir
            .build_select(below, floats(lowest), x, "")?
            .into_vector_value();
        let converted = if signed {
            ir.build_float_to_signed_int(clamped, ty, "")?
        } else {
            ir.build_float_to_unsigned_int(clamped, ty, "")?
        };
        let above = ir.build_float_compare(FloatPredicate::OGE, x, floats(2f64.powi(k)), "")?;
        let top = VectorType::const_vector(&vec![int.const_int(largest, false); lanes]);
        let value = ir.build_select(above, top, converted, "")?;
        let nan = ir.build_float_compare(FloatPredicate::UNO, x, x, "")?;
        Ok(ir
            .build_select(nan, ty.const_zero(), value.into_vector_value(), "")?
            .into_vector_value())
    }

    fn vector(&self, reg: Reg) -> Built<VectorValue<'ctx>> {
        let (ty, ptr) = self.registers[reg.0 as usize];
        Ok(self
            .builder
            .build_load(llvm_type(self.ctx, self.machine.vector_bytes, ty), ptr, "")?
            .into_vector_value())
    }

    pub(in crate::lower) fn simd(
        &mut self,
        dst: Option<Reg>,
        inst: Simd,
        args: &[Reg],
    ) -> Built<()> {
        use SimdOp::*;
        let e = inst.element;
        let ir = self.builder;
        let arg = |n: usize| self.vector(args[n]);
        let float = e.is_float();
        let value: BasicValueEnum<'ctx> = match inst.op {
            ByteWidth => self
                .ctx
                .i64_type()
                .const_int(self.machine.vector_bytes as u64, false)
                .into(),
            ElementWidth => self
                .ctx
                .i64_type()
                .const_int(e.lanes(self.machine.vector_bytes) as u64, false)
                .into(),
            SplatScalar | FirstN => {
                let lanes = e.lanes(self.machine.vector_bytes) as u32;
                let ty = self.ctx.i32_type().vec_type(lanes);
                let scalar = if inst.op == FirstN {
                    // Clamp the usize count to the lane count before narrowing it.
                    let count = self.offset(args[0])?;
                    let lanes = self.ctx.i64_type().const_int(u64::from(lanes), false);
                    let fits = ir.build_int_compare(IntPredicate::ULT, count, lanes, "")?;
                    let count = ir.build_select(fits, count, lanes, "")?.into_int_value();
                    ir.build_int_truncate(count, self.ctx.i32_type(), "")?
                } else {
                    self.load(args[0])?
                };
                let first = ir.build_insert_element(ty.get_undef(), scalar, self.int(0), "")?;
                let splat = ir.build_shuffle_vector(first, ty.get_undef(), ty.const_zero(), "")?;
                if inst.op == FirstN {
                    // Compare at the full index width even when the data lanes are narrower.
                    let positions: Vec<_> =
                        (0..ty.get_size()).map(|i| self.int(i as i32)).collect();
                    ir.build_int_compare(
                        IntPredicate::ULT,
                        VectorType::const_vector(&positions),
                        splat,
                        "",
                    )?
                    .into()
                } else {
                    splat.into()
                }
            }
            Splat(_) | Iota => {
                let int = element_type(self.ctx, e.index()).into_int_type();
                let mut lanes = Vec::with_capacity(e.lanes(self.machine.vector_bytes));
                for i in 0..e.lanes(self.machine.vector_bytes) {
                    let val = match inst.op {
                        Splat(bits) => ir.build_bit_cast(
                            int.const_int(bits, false),
                            element_type(self.ctx, e),
                            "",
                        )?,
                        Iota if float => element_type(self.ctx, e)
                            .into_float_type()
                            .const_float(i as f64)
                            .into(),
                        Iota => int.const_int(i as u64, false).into(),
                        _ => unreachable!(),
                    };
                    lanes.push(val);
                }
                VectorType::const_vector(&lanes).into()
            }
            Load(_) | Store(_) | LoadSelect(_) | StoreSelect(_) | Gather(_) | Scatter(_)
            | GatherSelect(_) | ScatterSelect(_) | CompressStore(_) => {
                let Some(loaded) = memory::lower(self, inst, args)? else {
                    return Ok(());
                };
                loaded.into()
            }
            AlignmentPrefix(b) => {
                let slice = self.slices[b.index()];
                let base = ir.build_ptr_to_int(slice.data, self.ctx.i32_type(), "address.low")?;
                // Only the low bits of the address decide the alignment.
                let offset = ir.build_int_truncate(self.load(args[0])?, self.ctx.i32_type(), "")?;
                let scaled = ir.build_left_shift(offset, self.int(2), "")?;
                let addr = ir.build_int_add(base, scaled, "")?;
                let bytes = ir.build_and(
                    ir.build_int_neg(addr, "")?,
                    self.int((self.machine.vector_bytes - 1) as i32),
                    "",
                )?;
                let prefix = ir.build_right_shift(bytes, self.int(2), false, "prefix")?;
                ir.build_int_z_extend(prefix, self.ctx.i64_type(), "")?
                    .into()
            }
            Compress | Expand if self.machine.x86_compress => {
                let data = arg(0)?;
                let (name, fallback) = if inst.op == Compress {
                    (
                        "llvm.x86.avx512.mask.compress",
                        data.get_type().const_zero(),
                    )
                } else {
                    ("llvm.x86.avx512.mask.expand", arg(2)?)
                };
                let intrinsic = self.intrinsic(name, &[data.get_type().into()])?;
                ir.build_call(
                    intrinsic,
                    &[data.into(), fallback.into(), arg(1)?.into()],
                    "",
                )?
                .try_as_basic_value()
                .unwrap_basic()
            }
            Compress => {
                let data = arg(0)?;
                let ty = data.get_type();
                let compress = self.intrinsic("llvm.experimental.vector.compress", &[ty.into()])?;
                ir.build_call(
                    compress,
                    &[data.into(), arg(1)?.into(), ty.const_zero().into()],
                    "compact",
                )?
                .try_as_basic_value()
                .unwrap_basic()
            }
            Expand => {
                let data = arg(0)?;
                let mask = arg(1)?;
                let fallback = arg(2)?;
                let mut src = self.int(0);
                let mut out = fallback;
                for i in 0..e.lanes(self.machine.vector_bytes) as i32 {
                    let pos = self.int(i);
                    let active = ir.build_extract_element(mask, pos, "")?.into_int_value();
                    let fresh = ir.build_extract_element(data, src, "")?.into_int_value();
                    let old = ir
                        .build_extract_element(fallback, pos, "")?
                        .into_int_value();
                    let value = ir.build_select(active, fresh, old, "")?;
                    out = ir.build_insert_element(out, value, pos, "")?;
                    let advance = ir.build_int_z_extend(active, self.ctx.i32_type(), "")?;
                    src = ir.build_int_add(src, advance, "")?;
                }
                out.into()
            }
            Swizzle => {
                let data = arg(0)?;
                let indices = arg(1)?;
                let width = e.lanes(self.machine.vector_bytes) as i32;
                let mut output = data.get_type().get_undef();
                for i in 0..width {
                    let pos = self.int(i);
                    let index = ir.build_extract_element(indices, pos, "")?.into_int_value();
                    let valid =
                        ir.build_int_compare(IntPredicate::ULT, index, self.int(width), "")?;
                    let lane = ir.build_extract_element(data, index, "")?.into_int_value();
                    // The false arm masks the poison produced by an out-of-range extract.
                    let value = ir.build_select(valid, lane, self.int(0), "")?;
                    output = ir.build_insert_element(output, value, pos, "")?;
                }
                output.into()
            }
            Shr(n) => ir
                .build_right_shift(
                    arg(0)?,
                    self.splat(
                        self.int((n & 31) as i32),
                        e.lanes(self.machine.vector_bytes) as u32,
                    )?,
                    !e.is_unsigned(),
                    "",
                )?
                .into(),
            Count => self.mask_count(arg(0)?)?.into(),
            Sqrt | MulAdd | ReduceSum => {
                let x = arg(0)?;
                let name = match inst.op {
                    Sqrt => "llvm.sqrt",
                    MulAdd => "llvm.fma",
                    _ => "llvm.vector.reduce.add",
                };
                let function = self.intrinsic(name, &[x.get_type().into()])?;
                let operands = if inst.op == MulAdd {
                    vec![x.into(), arg(1)?.into(), arg(2)?.into()]
                } else {
                    vec![x.into()]
                };
                ir.build_call(function, &operands, "")?
                    .try_as_basic_value()
                    .unwrap_basic()
            }
            Cast(to) => self.convert_lanes(arg(0)?, e, to)?.into(),
            Widen { to, high } => {
                let x = arg(0)?;
                let start = usize::from(high) * to.lanes(self.machine.vector_bytes);
                let mask: Vec<_> = (start..start + to.lanes(self.machine.vector_bytes))
                    .map(|i| self.int(i as i32))
                    .collect();
                let half = ir.build_shuffle_vector(x, x, VectorType::const_vector(&mask), "")?;
                self.convert_lanes(half, e, to)?.into()
            }
            Narrow(to) => {
                let a = self.convert_lanes(arg(0)?, e, to)?;
                let b = self.convert_lanes(arg(1)?, e, to)?;
                let mask: Vec<_> = (0..to.lanes(self.machine.vector_bytes))
                    .map(|i| self.int(i as i32))
                    .collect();
                ir.build_shuffle_vector(a, b, VectorType::const_vector(&mask), "")?
                    .into()
            }
            RotateLeft(n) => {
                let value = arg(0)?;
                let count = self.splat(
                    self.int((n & 31) as i32),
                    e.lanes(self.machine.vector_bytes) as u32,
                )?;
                let intrinsic = self.intrinsic("llvm.fshl", &[value.get_type().into()])?;
                ir.build_call(intrinsic, &[value.into(), value.into(), count.into()], "")?
                    .try_as_basic_value()
                    .unwrap_basic()
            }
            Add | Sub | Mul | Div => {
                let (a, b) = (arg(0)?, arg(1)?);
                match (inst.op, float) {
                    (Add, true) => ir.build_float_add(a, b, ""),
                    (Sub, true) => ir.build_float_sub(a, b, ""),
                    (Mul, true) => ir.build_float_mul(a, b, ""),
                    (Div, true) => ir.build_float_div(a, b, ""),
                    (Add, false) => ir.build_int_add(a, b, ""),
                    (Sub, false) => ir.build_int_sub(a, b, ""),
                    (Mul, false) => ir.build_int_mul(a, b, ""),
                    _ => unreachable!(),
                }?
                .into()
            }
            Min | Max => {
                let (a, b) = (arg(0)?, arg(1)?);
                let name = match (inst.op, e.is_unsigned()) {
                    (Min, true) => "llvm.umin",
                    (Max, true) => "llvm.umax",
                    (Min, false) => "llvm.smin",
                    _ => "llvm.smax",
                };
                let intrinsic = self.intrinsic(name, &[a.get_type().into()])?;
                ir.build_call(intrinsic, &[a.into(), b.into()], "")?
                    .try_as_basic_value()
                    .unwrap_basic()
            }
            Eq | Ne | Lt | Le | Gt | Ge => {
                let (a, b) = (arg(0)?, arg(1)?);
                if float {
                    let pred = match inst.op {
                        Eq => FloatPredicate::OEQ,
                        Ne => FloatPredicate::UNE,
                        Lt => FloatPredicate::OLT,
                        Le => FloatPredicate::OLE,
                        Gt => FloatPredicate::OGT,
                        Ge => FloatPredicate::OGE,
                        _ => unreachable!(),
                    };
                    ir.build_float_compare(pred, a, b, "")?.into()
                } else {
                    let unsigned = e.is_unsigned();
                    let pred = match inst.op {
                        Eq => IntPredicate::EQ,
                        Ne => IntPredicate::NE,
                        Lt if unsigned => IntPredicate::ULT,
                        Le if unsigned => IntPredicate::ULE,
                        Gt if unsigned => IntPredicate::UGT,
                        Ge if unsigned => IntPredicate::UGE,
                        Lt => IntPredicate::SLT,
                        Le => IntPredicate::SLE,
                        Gt => IntPredicate::SGT,
                        Ge => IntPredicate::SGE,
                        _ => unreachable!(),
                    };
                    ir.build_int_compare(pred, a, b, "")?.into()
                }
            }
            And | MaskAnd => ir.build_and(arg(0)?, arg(1)?, "")?.into(),
            Or | MaskOr => ir.build_or(arg(0)?, arg(1)?, "")?.into(),
            Xor | MaskXor => ir.build_xor(arg(0)?, arg(1)?, "")?.into(),
            Not | MaskNot => ir.build_not(arg(0)?, "")?.into(),
            AddMasked => {
                let sum = if float {
                    ir.build_float_add(arg(0)?, arg(2)?, "")?
                } else {
                    ir.build_int_add(arg(0)?, arg(2)?, "")?
                };
                ir.build_select(arg(1)?, sum, arg(3)?, "")?
            }
            Select => ir.build_select(arg(0)?, arg(1)?, arg(2)?, "")?,
            Shuffle(order) => {
                let mask: Vec<_> = (0..e.lanes(self.machine.vector_bytes))
                    .map(|i| {
                        self.ctx.i32_type().const_int(
                            order.index(i, e.lanes(self.machine.vector_bytes)) as u64,
                            false,
                        )
                    })
                    .collect();
                ir.build_shuffle_vector(arg(0)?, arg(1)?, VectorType::const_vector(&mask), "")?
                    .into()
            }
            ExtractWrapping | ReplaceWrapping => {
                let index = ir.build_and(
                    self.load(args[1])?,
                    self.int(e.lanes(self.machine.vector_bytes) as i32 - 1),
                    "",
                )?;
                if inst.op == ExtractWrapping {
                    ir.build_extract_element(arg(0)?, index, "")?
                } else {
                    ir.build_insert_element(arg(0)?, self.load(args[2])?, index, "")?
                        .into()
                }
            }
            Last => ir.build_extract_element(
                arg(0)?,
                self.ctx
                    .i32_type()
                    .const_int((e.lanes(self.machine.vector_bytes) - 1) as u64, false),
                "",
            )?,
            PrefixSum => {
                let mut sums = arg(0)?;
                let zero = sums.get_type().const_zero();
                let mut shift = 1;
                while shift < e.lanes(self.machine.vector_bytes) {
                    let indices: Vec<_> = (0..e.lanes(self.machine.vector_bytes))
                        .map(|i| {
                            self.ctx.i32_type().const_int(
                                i.checked_sub(shift)
                                    .unwrap_or(e.lanes(self.machine.vector_bytes))
                                    as u64,
                                false,
                            )
                        })
                        .collect();
                    let low = ir.build_shuffle_vector(
                        sums,
                        zero,
                        VectorType::const_vector(&indices),
                        "",
                    )?;
                    sums = ir.build_int_add(sums, low, "")?;
                    shift *= 2;
                }
                sums.into()
            }
            Any | All => {
                let bits = self
                    .ctx
                    .custom_width_int_type(
                        (e.lanes(self.machine.vector_bytes) as u32)
                            .try_into()
                            .unwrap(),
                    )
                    .unwrap();
                let mask = ir.build_bit_cast(arg(0)?, bits, "")?.into_int_value();
                if inst.op == Any {
                    ir.build_int_compare(IntPredicate::NE, mask, bits.const_zero(), "")?
                } else {
                    ir.build_int_compare(IntPredicate::EQ, mask, bits.const_all_ones(), "")?
                }
                .into()
            }
        };
        ir.build_store(
            self.registers[dst.expect("value-producing SIMD op").0 as usize].1,
            value,
        )?;
        Ok(())
    }

    fn mask_count(&self, mask: VectorValue<'ctx>) -> Built<IntValue<'ctx>> {
        let bits = self.mask_bits(mask)?;
        let pop = self.intrinsic("llvm.ctpop", &[bits.get_type().into()])?;
        let count = self
            .builder
            .build_call(pop, &[bits.into()], "selected")?
            .try_as_basic_value()
            .unwrap_basic()
            .into_int_value();
        Ok(self
            .builder
            .build_int_z_extend(count, self.ctx.i32_type(), "")?)
    }

    fn splat(
        &self,
        value: inkwell::values::IntValue<'ctx>,
        lanes: u32,
    ) -> Built<VectorValue<'ctx>> {
        let ty = value.get_type().vec_type(lanes);
        let one = self
            .builder
            .build_insert_element(ty.get_undef(), value, self.int(0), "")?;
        Ok(self.builder.build_shuffle_vector(
            one,
            ty.get_undef(),
            self.ctx.i32_type().vec_type(lanes).const_zero(),
            "",
        )?)
    }
}

impl<'ctx> Body<'_, 'ctx> {
    fn lanes(&self, e: Element) -> u32 {
        e.lanes(self.machine.vector_bytes) as u32
    }

    /// The address of each lane at `at`: one pointer, or a vector of them.
    fn lane_address(
        &self,
        s: Slice,
        at: Address<IntValue<'ctx>, VectorValue<'ctx>>,
    ) -> Built<BasicValueEnum<'ctx>> {
        let ir = self.builder;
        let el = element_type(self.ctx, s.element);
        let data = self.slices[s.buffer.index()].data;
        Ok(match at {
            // SAFETY: plain GEP computes the address without accessing
            // memory or asserting in-bounds; `memory::lower` checks each use.
            Address::Offset(offset) => unsafe { ir.build_gep(el, data, &[offset], "")? }.into(),
            Address::Indices(indices) => {
                // Valid slice indices fit in a signed i32. Keep 32-bit VSIB
                // offsets for x86 gather/scatter. A disabled negative GEP is
                // never dereferenced.
                let offsets = ir.build_int_z_extend_or_bit_cast(
                    indices,
                    self.index_type(s.element).vec_type(self.lanes(s.element)),
                    "offsets",
                )?;
                // Inkwell's build_gep accepts scalar indices. LLVM GEP also
                // accepts a vector of integer indices and yields a vector of
                // pointers; the masked intrinsics guard every access.
                use inkwell::{
                    llvm_sys::core::LLVMBuildGEP2, types::AsTypeRef, values::AsValueRef,
                };
                // SAFETY: the positioned builder, scalar pointer, vector index
                // and element type share a context. LLVM consumes this
                // one-index array during the call. GEP sign-extends the
                // 32-bit offsets; an active lane's index is below the slice
                // length, which `validate_buffers` bounds by `i32::MAX`, so
                // the extended offset is the index itself.
                unsafe {
                    VectorValue::new(LLVMBuildGEP2(
                        ir.as_mut_ptr(),
                        el.as_type_ref(),
                        data.as_value_ref(),
                        &mut offsets.as_value_ref(),
                        1,
                        c"addresses".as_ptr(),
                    ))
                }
                .into()
            }
        })
    }

    /// The integer type of the offsets of an indexed access.
    fn index_type(&self, e: Element) -> IntType<'ctx> {
        if e.index() == Element::I64 {
            self.ctx.i64_type()
        } else {
            self.ctx.i32_type()
        }
    }

    /// A mask as an integer with bit `i` for lane `i` (x64 is little endian).
    fn mask_bits(&self, mask: VectorValue<'ctx>) -> Built<IntValue<'ctx>> {
        let word = self
            .ctx
            .custom_width_int_type(mask.get_type().get_size().try_into().unwrap())
            .unwrap();
        Ok(self
            .builder
            .build_bit_cast(mask, word, "mask.bits")?
            .into_int_value())
    }

    fn all_lanes(&self, e: Element) -> VectorValue<'ctx> {
        VectorType::const_vector(&vec![
            self.ctx.bool_type().const_all_ones();
            self.lanes(e) as usize
        ])
    }

    fn masked_intrinsic(
        &self,
        name: &'static str,
        s: Slice,
        address: BasicValueEnum<'ctx>,
    ) -> Built<(
        inkwell::values::FunctionValue<'ctx>,
        BasicMetadataValueEnum<'ctx>,
    )> {
        let ty = vector_type(self.ctx, self.machine.vector_bytes, s.element);
        let decl = self.intrinsic(name, &[ty.into(), address.get_type()])?;
        Ok((decl, self.int(s.element.bytes() as i32).into()))
    }
}

impl<'ctx> memory::Primitives for Body<'_, 'ctx> {
    type Scalar = IntValue<'ctx>;
    type Vector = VectorValue<'ctx>;
    type Error = Error;

    fn read_offset(&mut self, r: Reg) -> Built<Self::Scalar> {
        self.offset(r)
    }
    fn read_vector(&mut self, r: Reg) -> Built<Self::Vector> {
        self.vector(r)
    }
    fn require_full(
        &mut self,
        s: Slice,
        offset: Self::Scalar,
        before_trap: BeforeTrap<'_, Self>,
    ) -> Built<()> {
        // `offset` is clamped to i32::MAX, so the end does not wrap.
        let lanes = self
            .ctx
            .i64_type()
            .const_int(u64::from(self.lanes(s.element)), false);
        let end = self.builder.build_int_add(offset, lanes, "")?;
        let len = self.slices[s.buffer.index()].len;
        let outside = self
            .builder
            .build_int_compare(IntPredicate::UGT, end, len, "")?;
        self.trap_if(outside, before_trap)
    }
    fn room(&mut self, s: Slice, offset: Self::Scalar) -> Built<Self::Scalar> {
        // Slice lengths fit i32; the subtraction is guarded for an invalid offset.
        let ir = self.builder;
        let len = self.slices[s.buffer.index()].len;
        let room = ir
            .build_select(
                ir.build_int_compare(IntPredicate::ULT, offset, len, "")?,
                ir.build_int_sub(len, offset, "")?,
                self.ctx.i64_type().const_zero(),
                "remaining",
            )?
            .into_int_value();
        Ok(ir.build_int_truncate(room, self.ctx.i32_type(), "")?)
    }
    fn first_lanes(&mut self, e: Element, count: Self::Scalar) -> Built<Self::Vector> {
        let lanes = self.lanes(e);
        let seq: Vec<_> = (0..lanes).map(|i| self.int(i as i32)).collect();
        Ok(self.builder.build_int_compare(
            IntPredicate::ULT,
            VectorType::const_vector(&seq),
            self.splat(count, lanes)?,
            "inside",
        )?)
    }
    fn outside(&mut self, s: Slice, indices: Self::Vector) -> Built<Self::Vector> {
        let ir = self.builder;
        let ty = self.index_type(s.element);
        let lanes = self.lanes(s.element);
        let offsets = ir.build_int_z_extend_or_bit_cast(indices, ty.vec_type(lanes), "")?;
        let len =
            ir.build_int_truncate_or_bit_cast(self.slices[s.buffer.index()].len, ty, "length")?;
        Ok(ir.build_int_compare(
            IntPredicate::UGE,
            offsets,
            self.splat(len, lanes)?,
            "outside",
        )?)
    }
    fn length_at_most(&mut self, buffer: Buffer, n: u64) -> Built<Self::Scalar> {
        let n = self.ctx.i64_type().const_int(n, false);
        Ok(self.builder.build_int_compare(
            IntPredicate::ULE,
            self.slices[buffer.index()].len,
            n,
            "",
        )?)
    }
    fn and(&mut self, a: Self::Scalar, b: Self::Scalar) -> Built<Self::Scalar> {
        Ok(self.builder.build_and(a, b, "")?)
    }
    fn mask_and(&mut self, a: Self::Vector, b: Self::Vector) -> Built<Self::Vector> {
        Ok(self.builder.build_and(a, b, "")?)
    }
    fn mask_xor(&mut self, a: Self::Vector, b: Self::Vector) -> Built<Self::Vector> {
        Ok(self.builder.build_xor(a, b, "")?)
    }
    fn any(&mut self, mask: Self::Vector) -> Built<Self::Scalar> {
        let bits = self.mask_bits(mask)?;
        Ok(self.builder.build_int_compare(
            IntPredicate::NE,
            bits,
            bits.get_type().const_zero(),
            "",
        )?)
    }
    fn before_first(&mut self, _: Element, mask: Self::Vector) -> Built<Self::Vector> {
        // lowbit(mask) - 1 has every bit below the first lane, and all bits
        // when the mask is empty.
        let ir = self.builder;
        let bits = self.mask_bits(mask)?;
        let low = ir.build_and(bits, ir.build_int_neg(bits, "")?, "first")?;
        let before = ir.build_int_sub(low, bits.get_type().const_int(1, false), "before")?;
        Ok(ir
            .build_bit_cast(before, mask.get_type(), "")?
            .into_vector_value())
    }
    fn exceeds(&mut self, mask: Self::Vector, count: Self::Scalar) -> Built<Self::Scalar> {
        let lanes = self.mask_count(mask)?;
        Ok(self
            .builder
            .build_int_compare(IntPredicate::UGT, lanes, count, "")?)
    }
    fn trap_if(&mut self, condition: Self::Scalar, before_trap: BeforeTrap<'_, Self>) -> Built<()> {
        let fault = Failure::Trap(splice_core::Trap::Bounds);
        let Some(before_trap) = before_trap else {
            return self.exit_if(condition, fault);
        };
        let fail = self.ctx.append_basic_block(self.function, "before.trap");
        let next = self.ctx.append_basic_block(self.function, "");
        self.builder
            .build_conditional_branch(condition, fail, next)?;
        self.builder.position_at_end(fail);
        before_trap(self)?;
        let exit = self.exit(fault)?;
        self.builder.build_unconditional_branch(exit)?;
        self.builder.position_at_end(next);
        Ok(())
    }
    fn load(
        &mut self,
        s: Slice,
        at: Address<Self::Scalar, Self::Vector>,
        masked: Option<Masked<Self::Vector>>,
    ) -> Built<Self::Vector> {
        let ty = vector_type(self.ctx, self.machine.vector_bytes, s.element);
        let address = self.lane_address(s, at)?;
        if let (Address::Offset(_), None) = (at, masked) {
            let value = self
                .builder
                .build_load(ty, address.into_pointer_value(), "")?
                .into_vector_value();
            value
                .as_instruction()
                .unwrap()
                .set_alignment(s.element.bytes() as u32)
                .expect("element alignment");
            return Ok(value);
        }
        let (mask, passthrough) = match masked {
            Some(Masked { mask, passthrough }) => (mask, passthrough),
            None => (self.all_lanes(s.element), ty.const_zero()),
        };
        let name = match at {
            Address::Offset(_) => "llvm.masked.load",
            Address::Indices(_) => "llvm.masked.gather",
        };
        let (decl, align) = self.masked_intrinsic(name, s, address)?;
        let call = self.builder.build_call(
            decl,
            &[address.into(), align, mask.into(), passthrough.into()],
            "",
        )?;
        Ok(call.try_as_basic_value().unwrap_basic().into_vector_value())
    }
    fn store(
        &mut self,
        s: Slice,
        at: Address<Self::Scalar, Self::Vector>,
        mask: Option<Self::Vector>,
        value: Self::Vector,
    ) -> Built<()> {
        let address = self.lane_address(s, at)?;
        if let (Address::Offset(_), None) = (at, mask) {
            self.builder
                .build_store(address.into_pointer_value(), value)?
                .set_alignment(s.element.bytes() as u32)
                .expect("element alignment");
            return Ok(());
        }
        let mask = mask.unwrap_or_else(|| self.all_lanes(s.element));
        let name = match at {
            Address::Offset(_) => "llvm.masked.store",
            Address::Indices(_) => "llvm.masked.scatter",
        };
        let (decl, align) = self.masked_intrinsic(name, s, address)?;
        self.builder.build_call(
            decl,
            &[value.into(), address.into(), align, mask.into()],
            "",
        )?;
        Ok(())
    }
    fn compress_store(
        &mut self,
        s: Slice,
        offset: Self::Scalar,
        mask: Self::Vector,
        value: Self::Vector,
        room: Self::Scalar,
    ) -> Built<()> {
        let ir = self.builder;
        let ptr = self.lane_address(s, Address::Offset(offset))?;
        let short = self.exceeds(mask, room)?;
        let full = ir.get_insert_block().unwrap();
        let partial = self
            .ctx
            .append_basic_block(self.function, "compact.partial");
        let write = self.ctx.append_basic_block(self.function, "compact.store");
        ir.build_conditional_branch(short, partial, write)?;
        ir.position_at_end(partial);
        // Keep exactly the first `room` set bits when they do not all fit.
        let mut prefix = mask.get_type().const_zero();
        let mut rank = self.int(0);
        for i in 0..mask.get_type().get_size() {
            let index = self.int(i as i32);
            let on = ir.build_extract_element(mask, index, "")?.into_int_value();
            let fits = ir.build_int_compare(IntPredicate::ULT, rank, room, "")?;
            prefix = ir.build_insert_element(prefix, ir.build_and(on, fits, "")?, index, "")?;
            rank = ir.build_int_add(
                rank,
                ir.build_int_z_extend(on, self.ctx.i32_type(), "")?,
                "",
            )?;
        }
        ir.build_unconditional_branch(write)?;
        ir.position_at_end(write);
        let enabled = ir.build_phi(mask.get_type(), "enabled")?;
        enabled.add_incoming(&[(&mask, full), (&prefix, partial)]);
        let store = self.intrinsic("llvm.masked.compressstore", &[value.get_type().into()])?;
        ir.build_call(
            store,
            &[value.into(), ptr.into(), enabled.as_basic_value().into()],
            "",
        )?;
        Ok(())
    }
}
