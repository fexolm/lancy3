//! Lower integer bit operations to native instructions.
use super::{Body, Built};
use splice_core::hir::MathOp;
use splice_core::hir::Reg;

impl<'ctx> Body<'_, 'ctx> {
    /// Store `op` on `args` to `dst`.
    pub(in crate::lower) fn native_intrinsic(
        &mut self,
        dst: Reg,
        op: MathOp,
        args: &[Reg],
    ) -> Built<()> {
        let ir = self.builder;
        let x = self.load(args[0])?;
        let ty = x.get_type();
        let call = |name, operands: &[inkwell::values::BasicMetadataValueEnum<'ctx>]| {
            let function = self.intrinsic(name, &[ty.into()])?;
            Ok::<_, super::Error>(
                ir.build_call(function, operands, "")?
                    .try_as_basic_value()
                    .unwrap_basic()
                    .into_int_value(),
            )
        };
        let value = match op {
            MathOp::CountOnes | MathOp::LeadingZeros | MathOp::TrailingZeros => {
                // Zero is defined: the count is the width.
                let defined = self.ctx.bool_type().const_zero();
                let count = match op {
                    MathOp::CountOnes => call("llvm.ctpop", &[x.into()])?,
                    MathOp::LeadingZeros => call("llvm.ctlz", &[x.into(), defined.into()])?,
                    _ => call("llvm.cttz", &[x.into(), defined.into()])?,
                };
                ir.build_int_cast_sign_flag(count, self.ctx.i32_type(), false, "")?
            }
            MathOp::RotateLeft => {
                // `fshl` takes the count modulo the width, as Rust does.
                let count = self.load(args[1])?;
                let count = ir.build_int_cast_sign_flag(count, ty, false, "")?;
                call("llvm.fshl", &[x.into(), x.into(), count.into()])?
            }
        };
        self.store(dst, value)
    }
}
