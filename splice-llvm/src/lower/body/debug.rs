use super::{Body, Built, simd};
use inkwell::builder::Builder;
use inkwell::values::BasicValueEnum;
use splice_core::{debug, hir::ValueType};

impl<'ctx> Body<'_, 'ctx> {
    /// Save the runtime variables of source point `ordinal` of function `id`
    /// in a snapshot of their own, and mark the point with a stack map record.
    pub(in crate::lower) fn debug_point(
        &self,
        allocas: &Builder<'ctx>,
        slots: &[debug::Slot],
        id: usize,
        ordinal: usize,
    ) -> Built<()> {
        let size = slots
            .last()
            .map_or(0, |slot| slot.offset + slot.ty.storage_bytes())
            .max(1);
        self.position_in_entry(allocas);
        let frame = allocas.build_array_alloca(
            self.ctx.i8_type(),
            self.ctx.i64_type().const_int(size as u64, false),
            "debug_values",
        )?;
        let b = self.builder;
        let save = |offset: usize, value: BasicValueEnum<'ctx>| -> Built<()> {
            let offset = self.ctx.i64_type().const_int(offset as u64, false);
            // SAFETY: the snapshot holds every slot of this point.
            let dst = unsafe { b.build_in_bounds_gep(self.ctx.i8_type(), frame, &[offset], "")? };
            let store = b.build_store(dst, value)?;
            store
                .set_alignment(1)
                .expect("byte alignment is valid for stores");
            store.set_volatile(true).expect("a store can be volatile");
            Ok(())
        };
        for slot in slots {
            match slot.variable.value {
                debug::Value::Register(reg, _) => {
                    let (ty, ptr) = self.registers[reg.0 as usize];
                    let value = b.build_load(
                        simd::llvm_type(self.ctx, self.machine.vector_bytes, ty),
                        ptr,
                        "",
                    )?;
                    let value = if let ValueType::Mask(element) = ty {
                        let ty = simd::llvm_type(
                            self.ctx,
                            self.machine.vector_bytes,
                            ValueType::Vector(element.index()),
                        )
                        .into_vector_type();
                        b.build_int_s_extend(value.into_vector_value(), ty, "")?
                            .into()
                    } else if matches!(slot.ty, debug::Type::Scalar(debug::Scalar::Index)) {
                        b.build_int_s_extend(value.into_int_value(), self.ctx.i64_type(), "")?
                            .into()
                    } else if matches!(slot.ty, debug::Type::Scalar(debug::Scalar::Bool)) {
                        // A DWARF bool is one byte holding 0 or 1.
                        b.build_and(value.into_int_value(), self.int(1), "")?.into()
                    } else {
                        value
                    };
                    save(slot.offset, value)?;
                }
                debug::Value::Slice(buffer) => {
                    let slice = self.slices[buffer.index()];
                    save(slot.offset, slice.data.into())?;
                    save(slot.offset + 8, slice.len.into())?;
                }
                debug::Value::Constant(_) => unreachable!("only runtime values have slots"),
            }
        }
        // The stack map records a static alloca as a frame register plus
        // displacement and places the source point after the snapshot stores.
        let intr = self.intrinsic("llvm.experimental.stackmap", &[])?;
        let id = self
            .ctx
            .i64_type()
            .const_int((id as u64) << 32 | ordinal as u64, false);
        let shadow = self
            .ctx
            .i32_type()
            .const_int(if cfg!(target_arch = "x86_64") { 1 } else { 4 }, false);
        b.build_direct_call(intr, &[id.into(), shadow.into(), frame.into()], "")?;
        Ok(())
    }
}
