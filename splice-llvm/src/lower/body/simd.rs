//! LLVM vectors have a concrete width once the JIT target is known. Mask values
//! use one i1 per data lane, so they feed LLVM comparisons and selects directly.
use super::{Body, Built, Failure, Slice};
use inkwell::{
    FloatPredicate, IntPredicate,
    context::Context,
    types::{BasicTypeEnum, VectorType},
    values::{BasicValueEnum, VectorValue},
};
use splice_core::Trap;
use splice_core::{
    hir::{Reg, Simd, SimdOp, ValueType},
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

fn vector_type(ctx: &Context, e: Element) -> VectorType<'_> {
    vector_type_with_lanes(ctx, e, e.lanes() as u32)
}

fn vector_type_with_lanes(ctx: &Context, e: Element, lanes: u32) -> VectorType<'_> {
    match element_type(ctx, e) {
        BasicTypeEnum::IntType(t) => t.vec_type(lanes),
        BasicTypeEnum::FloatType(t) => t.vec_type(lanes),
        _ => unreachable!(),
    }
}

pub(super) fn llvm_type(ctx: &Context, ty: ValueType) -> BasicTypeEnum<'_> {
    match ty {
        ValueType::Scalar => ctx.i32_type().into(),
        ValueType::Vector(e) => vector_type(ctx, e).into(),
        ValueType::Mask(e) => ctx.bool_type().vec_type(e.lanes() as u32).into(),
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

    fn vector(&self, reg: Reg) -> Built<VectorValue<'ctx>> {
        let (ty, ptr) = self.registers[reg.0 as usize];
        Ok(self
            .builder
            .build_load(llvm_type(self.ctx, ty), ptr, "")?
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
            Lanes => self.int(e.lanes() as i32).into(),
            SplatScalar | FirstN => {
                let ty = self.ctx.i32_type().vec_type(e.lanes() as u32);
                let first =
                    ir.build_insert_element(ty.get_undef(), self.load(args[0])?, self.int(0), "")?;
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
                let mut lanes = Vec::with_capacity(e.lanes());
                for i in 0..e.lanes() {
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
            Load(b) | Store(b) | LoadSelect(b) | StoreSelect(b) | Gather(b) | Scatter(b)
            | GatherSelect(b) | ScatterSelect(b) => {
                let Some(loaded) = self.vector_memory(self.slices[b.index()], inst, args)? else {
                    return Ok(());
                };
                loaded.into()
            }
            AlignmentPrefix(b) => {
                let slice = self.slices[b.index()];
                let base = ir.build_ptr_to_int(slice.data, self.ctx.i32_type(), "address.low")?;
                let scaled = ir.build_left_shift(self.load(args[0])?, self.int(2), "")?;
                let addr = ir.build_int_add(base, scaled, "")?;
                let bytes = ir.build_and(
                    ir.build_int_neg(addr, "")?,
                    self.int((splice_core::simd::vector_bytes() - 1) as i32),
                    "",
                )?;
                ir.build_right_shift(bytes, self.int(2), false, "prefix")?
                    .into()
            }
            Compress | Expand if crate::target::x86_compress(e.lanes()) => {
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
                for i in 0..e.lanes() as i32 {
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
                let width = e.lanes() as i32;
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
            CompressStore(b) => {
                self.compress_store(
                    self.slices[b.index()],
                    self.load(args[0])?,
                    arg(1)?,
                    arg(2)?,
                )?;
                return Ok(());
            }
            Shr(n) => ir
                .build_right_shift(
                    arg(0)?,
                    self.splat(self.int((n & 31) as i32), e.lanes() as u32)?,
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
                let start = usize::from(high) * to.lanes();
                let mask: Vec<_> = (start..start + to.lanes())
                    .map(|i| self.int(i as i32))
                    .collect();
                let half = ir.build_shuffle_vector(x, x, VectorType::const_vector(&mask), "")?;
                self.convert_lanes(half, e, to)?.into()
            }
            Narrow(to) => {
                let a = self.convert_lanes(arg(0)?, e, to)?;
                let b = self.convert_lanes(arg(1)?, e, to)?;
                let mask: Vec<_> = (0..to.lanes()).map(|i| self.int(i as i32)).collect();
                ir.build_shuffle_vector(a, b, VectorType::const_vector(&mask), "")?
                    .into()
            }
            RotateLeft(n) => {
                let value = arg(0)?;
                let count = self.splat(self.int((n & 31) as i32), e.lanes() as u32)?;
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
            Select => ir.build_select(arg(0)?, arg(1)?, arg(2)?, "")?,
            Shuffle(order) => {
                let mask: Vec<_> = (0..e.lanes())
                    .map(|i| {
                        self.ctx
                            .i32_type()
                            .const_int(order.index(i, e.lanes()) as u64, false)
                    })
                    .collect();
                ir.build_shuffle_vector(arg(0)?, arg(1)?, VectorType::const_vector(&mask), "")?
                    .into()
            }
            ExtractWrapping | ReplaceWrapping => {
                let index =
                    ir.build_and(self.load(args[1])?, self.int(e.lanes() as i32 - 1), "")?;
                if inst.op == ExtractWrapping {
                    ir.build_extract_element(arg(0)?, index, "")?
                } else {
                    ir.build_insert_element(arg(0)?, self.load(args[2])?, index, "")?
                        .into()
                }
            }
            Last => ir.build_extract_element(
                arg(0)?,
                self.ctx.i32_type().const_int((e.lanes() - 1) as u64, false),
                "",
            )?,
            PrefixSum => {
                let mut sums = arg(0)?;
                let zero = sums.get_type().const_zero();
                let mut shift = 1;
                while shift < e.lanes() {
                    let indices: Vec<_> = (0..e.lanes())
                        .map(|i| {
                            self.ctx
                                .i32_type()
                                .const_int(i.checked_sub(shift).unwrap_or(e.lanes()) as u64, false)
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
                    .custom_width_int_type((e.lanes() as u32).try_into().unwrap())
                    .unwrap();
                let mask = ir.build_bit_cast(arg(0)?, bits, "")?.into_int_value();
                let flag = if inst.op == Any {
                    ir.build_int_compare(IntPredicate::NE, mask, bits.const_zero(), "")?
                } else {
                    ir.build_int_compare(IntPredicate::EQ, mask, bits.const_all_ones(), "")?
                };
                ir.build_int_s_extend(flag, self.ctx.i32_type(), "")?.into()
            }
        };
        ir.build_store(
            self.registers[dst.expect("value-producing SIMD op").0 as usize].1,
            value,
        )?;
        Ok(())
    }

    fn mask_count(&self, mask: VectorValue<'ctx>) -> Built<inkwell::values::IntValue<'ctx>> {
        let ty = self
            .ctx
            .custom_width_int_type(mask.get_type().get_size().try_into().unwrap())
            .unwrap();
        let bits = self.builder.build_bit_cast(mask, ty, "mask.bits")?;
        let pop = self.intrinsic("llvm.ctpop", &[ty.into()])?;
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

    fn compress_store(
        &mut self,
        slice: Slice<'ctx>,
        offset: inkwell::values::IntValue<'ctx>,
        mask: VectorValue<'ctx>,
        data: VectorValue<'ctx>,
    ) -> Built<()> {
        let ir = self.builder;
        let wide = self.ctx.i64_type();
        let offset = ir.build_int_z_extend(offset, wide, "")?;
        // SAFETY: plain GEP is valid for inactive, out-of-slice offsets. The mask
        // passed to compressstore enables at most the available number of words.
        let ptr = unsafe { ir.build_gep(self.ctx.i32_type(), slice.data, &[offset], "")? };
        let fits_offset = ir.build_int_compare(IntPredicate::ULT, offset, slice.len, "")?;
        let available = ir
            .build_select(
                fits_offset,
                ir.build_int_sub(slice.len, offset, "")?,
                wide.const_zero(),
                "available",
            )?
            .into_int_value();
        let count = ir.build_int_z_extend(self.mask_count(mask)?, wide, "")?;
        let short = ir.build_int_compare(IntPredicate::UGT, count, available, "")?;
        let full = ir.get_insert_block().unwrap();
        let partial = self
            .ctx
            .append_basic_block(self.function, "compact.partial");
        let write = self.ctx.append_basic_block(self.function, "compact.store");
        ir.build_conditional_branch(short, partial, write)?;
        ir.position_at_end(partial);
        // Keep exactly the first 'available' set bits on the bounds-error path.
        let mut prefix = mask.get_type().const_zero();
        let mut rank = wide.const_zero();
        for i in 0..mask.get_type().get_size() {
            let index = self.int(i as i32);
            let on = ir.build_extract_element(mask, index, "")?.into_int_value();
            let room = ir.build_int_compare(IntPredicate::ULT, rank, available, "")?;
            prefix = ir.build_insert_element(prefix, ir.build_and(on, room, "")?, index, "")?;
            rank = ir.build_int_add(rank, ir.build_int_z_extend(on, wide, "")?, "")?;
        }
        ir.build_unconditional_branch(write)?;
        ir.position_at_end(write);
        let enabled = ir.build_phi(mask.get_type(), "enabled")?;
        enabled.add_incoming(&[(&mask, full), (&prefix, partial)]);
        let store = self.intrinsic("llvm.masked.compressstore", &[data.get_type().into()])?;
        ir.build_call(
            store,
            &[data.into(), ptr.into(), enabled.as_basic_value().into()],
            "",
        )?;
        self.exit_if(short, Failure::Trap(Trap::Bounds))
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

    fn vector_memory(
        &mut self,
        slice: Slice<'ctx>,
        inst: Simd,
        args: &[Reg],
    ) -> Built<Option<VectorValue<'ctx>>> {
        use SimdOp::*;
        use inkwell::values::BasicMetadataValueEnum;
        let e = inst.element;
        let ir = self.builder;
        let el = element_type(self.ctx, e);
        let ty = vector_type(self.ctx, e);
        let count = ty.get_size();
        let wide = self.ctx.i64_type();
        let indexed = matches!(
            inst.op,
            Gather(_) | GatherSelect(_) | Scatter(_) | ScatterSelect(_)
        );
        let masked = matches!(
            inst.op,
            LoadSelect(_) | StoreSelect(_) | GatherSelect(_) | ScatterSelect(_)
        );
        let store = matches!(
            inst.op,
            Store(_) | StoreSelect(_) | Scatter(_) | ScatterSelect(_)
        );
        let active = if masked {
            self.vector(args[1])?
        } else {
            VectorType::const_vector(&vec![self.ctx.bool_type().const_all_ones(); count as usize])
        };
        let values = if store || masked {
            self.vector(*args.last().unwrap())?
        } else {
            ty.const_zero()
        };
        let mut full_store_exit = None;
        let (outside, address) = if indexed {
            let input = self.vector(args[0])?;
            // Valid slice indices fit in a signed i32. Keep 32-bit VSIB offsets
            // for x86 gather/scatter. A disabled negative GEP is never dereferenced.
            let index_type = if e.index() == Element::I64 {
                wide
            } else {
                self.ctx.i32_type()
            };
            let offsets =
                ir.build_int_z_extend_or_bit_cast(input, index_type.vec_type(count), "offsets")?;
            let len = ir.build_int_truncate_or_bit_cast(slice.len, index_type, "length")?;
            let outside = ir.build_int_compare(
                IntPredicate::UGE,
                offsets,
                self.splat(len, count)?,
                "outside",
            )?;
            // Inkwell's build_gep accepts scalar indices. LLVM GEP also accepts a
            // vector of integer indices and yields a vector of pointers. Plain GEP
            // permits inactive out-of-range offsets; the intrinsic masks all accesses.
            use inkwell::{llvm_sys::core::LLVMBuildGEP2, types::AsTypeRef, values::AsValueRef};
            // SAFETY: the positioned builder, scalar pointer, vector index and element
            // type share a context. LLVM consumes this one-index array during the call.
            let addr = unsafe {
                VectorValue::new(LLVMBuildGEP2(
                    ir.as_mut_ptr(),
                    el.as_type_ref(),
                    slice.data.as_value_ref(),
                    &mut offsets.as_value_ref(),
                    1,
                    c"addresses".as_ptr(),
                ))
            };
            (outside, BasicValueEnum::from(addr))
        } else {
            let start = ir.build_int_z_extend(self.load(args[0])?, wide, "")?;
            // SAFETY: plain GEP computes the address without accessing memory or
            // asserting in-bounds. The full interval or per-lane mask guards each use.
            let addr = unsafe { ir.build_gep(el, slice.data, &[start], "")? };
            if !masked {
                let end = ir.build_int_add(start, wide.const_int(u64::from(count), false), "")?;
                let outside = ir.build_int_compare(IntPredicate::UGT, end, slice.len, "")?;
                if !store {
                    self.exit_if(outside, Failure::Trap(Trap::Bounds))?;
                    let val = ir.build_load(ty, addr, "")?.into_vector_value();
                    val.as_instruction()
                        .unwrap()
                        .set_alignment(e.bytes() as u32)
                        .expect("element alignment");
                    return Ok(Some(val));
                }
                let fast = self.ctx.append_basic_block(self.function, "full.store");
                let partial = self.ctx.append_basic_block(self.function, "partial.store");
                let done = self.ctx.append_basic_block(self.function, "stored");
                ir.build_conditional_branch(outside, partial, fast)?;
                ir.position_at_end(fast);
                ir.build_store(addr, values)?
                    .set_alignment(e.bytes() as u32)
                    .expect("element alignment");
                ir.build_unconditional_branch(done)?;
                ir.position_at_end(partial);
                full_store_exit = Some(done);
            }
            // Compare lane ordinals with the remaining element count. Slice lengths
            // fit i32, while the subtraction is widened to handle an invalid start.
            let room = ir
                .build_select(
                    ir.build_int_compare(IntPredicate::ULT, start, slice.len, "")?,
                    ir.build_int_sub(slice.len, start, "")?,
                    wide.const_zero(),
                    "remaining",
                )?
                .into_int_value();
            let room = ir.build_int_truncate(room, self.ctx.i32_type(), "")?;
            let seq: Vec<_> = (0..count).map(|i| self.int(i as i32)).collect();
            let outside = ir.build_int_compare(
                IntPredicate::UGE,
                VectorType::const_vector(&seq),
                self.splat(room, count)?,
                "outside",
            )?;
            (outside, addr.into())
        };
        let bad = ir.build_and(active, outside, "bad.lanes")?;
        let mask_word = self
            .ctx
            .custom_width_int_type(count.try_into().unwrap())
            .unwrap();
        // x64 is little endian, so bit i corresponds to lane i.
        let bits = ir
            .build_bit_cast(bad, mask_word, "bad.bits")?
            .into_int_value();
        let failed = ir.build_int_compare(IntPredicate::NE, bits, mask_word.const_zero(), "")?;
        let enabled = if store {
            // (lowbit(bad) - 1) includes every lane strictly before the first
            // fault, and all lanes if bad == 0. This preserves visible partial stores.
            let low = ir.build_and(bits, ir.build_int_neg(bits, "")?, "first.bad")?;
            let prefix = ir.build_int_sub(low, mask_word.const_int(1, false), "prefix")?;
            let prefix = ir
                .build_bit_cast(prefix, active.get_type(), "")?
                .into_vector_value();
            ir.build_and(active, prefix, "enabled")?
        } else {
            self.exit_if(failed, Failure::Trap(Trap::Bounds))?;
            active
        };
        let name = match (indexed, store) {
            (true, true) => "llvm.masked.scatter",
            (true, false) => "llvm.masked.gather",
            (false, true) => "llvm.masked.store",
            (false, false) => "llvm.masked.load",
        };
        let decl = self.intrinsic(name, &[ty.into(), address.get_type()])?;
        let align = BasicMetadataValueEnum::from(self.int(e.bytes() as i32));
        if store {
            ir.build_call(
                decl,
                &[values.into(), address.into(), align, enabled.into()],
                "",
            )?;
            self.exit_if(failed, Failure::Trap(Trap::Bounds))?;
            if let Some(done) = full_store_exit {
                ir.build_unconditional_branch(done)?;
                ir.position_at_end(done);
            }
            Ok(None)
        } else {
            let call = ir.build_call(
                decl,
                &[address.into(), align, enabled.into(), values.into()],
                "",
            )?;
            Ok(Some(
                call.try_as_basic_value().unwrap_basic().into_vector_value(),
            ))
        }
    }
}
