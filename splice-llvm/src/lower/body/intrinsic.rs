//! The kernel intrinsics LLVM implements with its own intrinsics.
use super::{Body, Built};
use crate::target::Machine;
use splice_core::hir::{Reg, ScalarType};
use splice_core::intrinsic::Intrinsic;

/// An intrinsic LLVM emits instead of compiling its body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Native {
    /// `count_ones`, `leading_zeros`, `trailing_zeros` of a 32- or 64-bit
    /// integer, as `u32`.
    CountOnes,
    LeadingZeros,
    TrailingZeros,
    /// `rotate_left(x, n: u32)` of a 32- or 64-bit integer.
    RotateLeft,
    /// `sqrt` of a float.
    Sqrt,
    /// `mul_add(a, b, c)` of floats, where the machine fuses it: LLVM would
    /// otherwise call the C library.
    MulAdd,
}

/// How LLVM implements `intrinsic` on `machine`, if it does.
pub(crate) fn native(intrinsic: &Intrinsic<'_>, machine: &Machine) -> Option<Native> {
    use ScalarType as S;
    let signature = intrinsic.signature;
    let result = signature.result;
    let integer = |ty: ScalarType| matches!(ty, S::I32 | S::U32 | S::I64 | S::U64);
    match (intrinsic.name, signature.params) {
        ("count_ones" | "leading_zeros" | "trailing_zeros", &[x])
            if integer(x) && result == S::U32 =>
        {
            Some(match intrinsic.name {
                "count_ones" => Native::CountOnes,
                "leading_zeros" => Native::LeadingZeros,
                _ => Native::TrailingZeros,
            })
        }
        ("rotate_left", &[x, S::U32]) if integer(x) && result == x => Some(Native::RotateLeft),
        ("sqrt", &[x]) if x.is_float() && result == x => Some(Native::Sqrt),
        ("mul_add", &[a, b, c])
            if a.is_float() && a == b && b == c && result == a && machine.fma =>
        {
            Some(Native::MulAdd)
        }
        _ => None,
    }
}

impl<'ctx> Body<'_, 'ctx> {
    /// Store `op` on `args` to `dst`, of type `result`.
    pub(in crate::lower) fn native_intrinsic(
        &mut self,
        dst: Reg,
        op: Native,
        args: &[Reg],
        result: ScalarType,
    ) -> Built<()> {
        let ir = self.builder;
        let x = self.value(args[0])?;
        let ty = x.get_type();
        let call = |name, operands: &[inkwell::values::BasicMetadataValueEnum<'ctx>]| {
            let function = self.intrinsic(name, &[ty])?;
            Ok::<_, super::Error>(
                ir.build_call(function, operands, "")?
                    .try_as_basic_value()
                    .unwrap_basic(),
            )
        };
        let value = match op {
            Native::CountOnes | Native::LeadingZeros | Native::TrailingZeros => {
                // Zero is defined: the count is the width.
                let defined = self.ctx.bool_type().const_zero();
                let count = match op {
                    Native::CountOnes => call("llvm.ctpop", &[x.into()])?,
                    Native::LeadingZeros => call("llvm.ctlz", &[x.into(), defined.into()])?,
                    _ => call("llvm.cttz", &[x.into(), defined.into()])?,
                };
                let u32 = self.ctx.i32_type();
                ir.build_int_cast_sign_flag(count.into_int_value(), u32, false, "")?
                    .into()
            }
            Native::RotateLeft => {
                // `fshl` takes the count modulo the width, as Rust does.
                let count = self.load(args[1])?;
                let count = ir.build_int_cast_sign_flag(count, ty.into_int_type(), false, "")?;
                call("llvm.fshl", &[x.into(), x.into(), count.into()])?
            }
            Native::Sqrt => call("llvm.sqrt", &[x.into()])?,
            Native::MulAdd => {
                let (b, c) = (self.value(args[1])?, self.value(args[2])?);
                call("llvm.fma", &[x.into(), b.into(), c.into()])?
            }
        };
        debug_assert_eq!(self.scalar_type(dst), result);
        self.store(dst, value)
    }
}
