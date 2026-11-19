//! LLVM vectors have a concrete width once the JIT target is known. Mask values
//! use one i1 per data lane, so they feed LLVM comparisons and selects directly.
use super::{Body, Built, Failure, Slice};
use inkwell::{
    FloatPredicate, IntPredicate,
    context::Context,
    types::{BasicTypeEnum, VectorType},
    values::{BasicValueEnum, VectorValue},
};
use splice::{
    hir::{Reg, Simd, SimdOp, ValueType},
    simd::Element,
};

fn element_type(ctx: &Context, e: Element) -> BasicTypeEnum<'_> {
    match e {
        Element::U16 => ctx.i16_type().into(),
        Element::I32 => ctx.i32_type().into(),
        Element::I64 => ctx.i64_type().into(),
        Element::F32 => ctx.f32_type().into(),
        Element::F64 => ctx.f64_type().into(),
    }
}

fn vector_type(ctx: &Context, e: Element) -> VectorType<'_> {
    match element_type(ctx, e) {
        BasicTypeEnum::IntType(t) => t.vec_type(e.lanes() as u32),
        BasicTypeEnum::FloatType(t) => t.vec_type(e.lanes() as u32),
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
                    let unsigned = e == Element::U16;
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

    fn splat(
        &self,
        value: inkwell::values::IntValue<'ctx>,
        lanes: u32,
    ) -> Built<VectorValue<'ctx>> {
        let ty = value.get_type().vec_type(lanes);
        let one = self
            .builder
            .build_insert_element(ty.get_undef(), value, self.int(0), "")?;
        self.builder.build_shuffle_vector(
            one,
            ty.get_undef(),
            self.ctx.i32_type().vec_type(lanes).const_zero(),
            "",
        )
    }

    fn vector_memory(
        &mut self,
        slice: Slice<'ctx>,
        inst: Simd,
        args: &[Reg],
    ) -> Built<Option<VectorValue<'ctx>>> {
        use SimdOp::*;
        use inkwell::{intrinsics::Intrinsic, values::BasicMetadataValueEnum};
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
        let (indices, address) = if indexed {
            let input = self.vector(args[0])?;
            let offsets =
                ir.build_int_z_extend_or_bit_cast(input, wide.vec_type(count), "offsets")?;
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
            (offsets, BasicValueEnum::from(addr))
        } else {
            let start = ir.build_int_z_extend(self.load(args[0])?, wide, "")?;
            // SAFETY: plain GEP computes the address without accessing memory or
            // asserting in-bounds. The full interval or per-lane mask guards each use.
            let addr = unsafe { ir.build_gep(el, slice.data, &[start], "")? };
            if !masked {
                let end = ir.build_int_add(start, wide.const_int(u64::from(count), false), "")?;
                let outside = ir.build_int_compare(IntPredicate::UGT, end, slice.len, "")?;
                if !store {
                    self.exit_if(outside, Failure::Bounds)?;
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
            let seq: Vec<_> = (0..count)
                .map(|i| wide.const_int(u64::from(i), false))
                .collect();
            (
                ir.build_int_add(
                    self.splat(start, count)?,
                    VectorType::const_vector(&seq),
                    "offsets",
                )?,
                addr.into(),
            )
        };
        let outside = ir.build_int_compare(
            IntPredicate::UGE,
            indices,
            self.splat(slice.len, count)?,
            "outside",
        )?;
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
            self.exit_if(failed, Failure::Bounds)?;
            active
        };
        let name = match (indexed, store) {
            (true, true) => "llvm.masked.scatter",
            (true, false) => "llvm.masked.gather",
            (false, true) => "llvm.masked.store",
            (false, false) => "llvm.masked.load",
        };
        let decl = Intrinsic::find(name)
            .unwrap()
            .get_declaration(self.module, &[ty.into(), address.get_type()])
            .unwrap();
        let align = BasicMetadataValueEnum::from(self.int(e.bytes() as i32));
        if store {
            ir.build_call(
                decl,
                &[values.into(), address.into(), align, enabled.into()],
                "",
            )?;
            self.exit_if(failed, Failure::Bounds)?;
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
