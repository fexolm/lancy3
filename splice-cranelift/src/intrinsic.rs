//! The kernel intrinsics Cranelift implements with its own instructions.
use cranelift_codegen::ir::{InstBuilder, Value, types::I32};
use cranelift_frontend::FunctionBuilder;
use splice_core::hir::ScalarType;
use splice_core::intrinsic::Intrinsic;

/// An intrinsic Cranelift emits instead of compiling its body. Cranelift
/// lowers each of these on every target it supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Native {
    /// `count_ones`, `leading_zeros`, `trailing_zeros` of a 32- or 64-bit
    /// integer, as `u32`.
    CountOnes,
    LeadingZeros,
    TrailingZeros,
    /// `rotate_left(x, n: u32)` of a 32- or 64-bit integer.
    RotateLeft,
}

/// How Cranelift implements `intrinsic`, if it does.
pub(crate) fn native(intrinsic: &Intrinsic<'_>) -> Option<Native> {
    use ScalarType as S;
    let signature = intrinsic.signature;
    let result = signature.result;
    let integer = |ty: ScalarType| matches!(ty, S::I32 | S::U32 | S::I64 | S::U64);
    match (intrinsic.name, signature.params) {
        ("count_ones", &[x]) if integer(x) && result == S::U32 => Some(Native::CountOnes),
        ("leading_zeros", &[x]) if integer(x) && result == S::U32 => Some(Native::LeadingZeros),
        ("trailing_zeros", &[x]) if integer(x) && result == S::U32 => Some(Native::TrailingZeros),
        ("rotate_left", &[x, S::U32]) if integer(x) && result == x => Some(Native::RotateLeft),
        _ => None,
    }
}

/// `op` on `args`.
pub(crate) fn emit(b: &mut FunctionBuilder, op: Native, args: &[Value]) -> Value {
    let x = args[0];
    let count = |b: &mut FunctionBuilder, count: Value| {
        if b.func.dfg.value_type(count) == I32 {
            count
        } else {
            b.ins().ireduce(I32, count)
        }
    };
    match op {
        Native::CountOnes => {
            let c = b.ins().popcnt(x);
            count(b, c)
        }
        Native::LeadingZeros => {
            let c = b.ins().clz(x);
            count(b, c)
        }
        Native::TrailingZeros => {
            let c = b.ins().ctz(x);
            count(b, c)
        }
        // `rotl` takes the count modulo the width, as Rust does.
        Native::RotateLeft => b.ins().rotl(x, args[1]),
    }
}
