use super::{Body, Built, simd};
use inkwell::values::{BasicValueEnum, PointerValue};
use splice_core::{
    Function, debug,
    hir::{Instr, ValueType},
};

impl<'ctx> Body<'_, 'ctx> {
    pub(super) fn debug_frame(&self, function: &Function) -> Built<PointerValue<'ctx>> {
        let size = function
            .instructions()
            .iter()
            .filter_map(|op| match op {
                Instr::Debug(point) => Some(point.storage_bytes(function)),
                _ => None,
            })
            .max()
            .unwrap_or(0)
            .max(1);
        let size = self.ctx.i64_type().const_int(size as u64, false);
        Ok(self
            .builder
            .build_array_alloca(self.ctx.i8_type(), size, "debug_values")?)
    }

    pub(super) fn debug_point(
        &self,
        function: &Function,
        point: &debug::Point,
        frame: PointerValue<'ctx>,
        id: usize,
        index: usize,
    ) -> Built<()> {
        let b = self.builder;
        let save = |offset: usize, value: BasicValueEnum<'ctx>| -> Built<()> {
            let offset = self.ctx.i64_type().const_int(offset as u64, false);
            // SAFETY: debug_frame reserves at least the largest verified
            // point layout. The store fits in that allocation.
            let dst = unsafe { b.build_in_bounds_gep(self.ctx.i8_type(), frame, &[offset], "")? };
            let store = b.build_store(dst, value)?;
            store
                .set_alignment(1)
                .expect("byte alignment is valid for stores");
            store.set_volatile(true).expect("a store can be volatile");
            Ok(())
        };
        for (var, layout, offset) in point.slots(function) {
            match var.value {
                debug::Value::Register(reg, _) => {
                    let (ty, ptr) = self.registers[reg.0 as usize];
                    let value = b.build_load(simd::llvm_type(self.ctx, ty), ptr, "")?;
                    let value = if let ValueType::Mask(element) = ty {
                        let ty = simd::llvm_type(self.ctx, ValueType::Vector(element.index()))
                            .into_vector_type();
                        b.build_int_s_extend(value.into_vector_value(), ty, "")?
                            .into()
                    } else if matches!(layout, debug::Type::Scalar(debug::Scalar::Index)) {
                        b.build_int_s_extend(value.into_int_value(), self.ctx.i64_type(), "")?
                            .into()
                    } else if matches!(layout, debug::Type::Scalar(debug::Scalar::Bool)) {
                        // A DWARF bool is one byte holding 0 or 1.
                        b.build_and(value.into_int_value(), self.int(1), "")?.into()
                    } else {
                        value
                    };
                    save(offset, value)?;
                }
                debug::Value::Slice(buffer) => {
                    let slice = self.slices[buffer.index()];
                    save(offset, slice.data.into())?;
                    save(offset + 8, slice.len.into())?;
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
            .const_int((id as u64) << 32 | index as u64, false);
        let shadow = self
            .ctx
            .i32_type()
            .const_int(if cfg!(target_arch = "x86_64") { 1 } else { 4 }, false);
        b.build_direct_call(intr, &[id.into(), shadow.into(), frame.into()], "")?;
        Ok(())
    }
}
