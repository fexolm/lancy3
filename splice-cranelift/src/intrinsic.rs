//! The kernel intrinsics Cranelift implements: every [`splice_core::Math`]
//! operation, with Cranelift's own instructions on every target.
use cranelift_codegen::ir::{InstBuilder, Value, types::I32};
use cranelift_frontend::FunctionBuilder;
use splice_core::intrinsic::IntrinsicId;
use splice_core::math::MathOp;

/// How Cranelift implements `id`, if it does.
pub(crate) fn native(id: &IntrinsicId) -> Option<MathOp> {
    splice_core::math::decode(id).map(|(op, _)| op)
}

/// `op` on `args`.
pub(crate) fn emit(b: &mut FunctionBuilder, op: MathOp, args: &[Value]) -> Value {
    let x = args[0];
    // The count of an operand of any width is a `u32`: it fits the low
    // half of a 64-bit operand's type and zero-extends a narrow one's.
    let count = |b: &mut FunctionBuilder, count: Value| match b.func.dfg.value_type(count).bits() {
        32 => count,
        64 => b.ins().ireduce(I32, count),
        _ => b.ins().uextend(I32, count),
    };
    match op {
        MathOp::CountOnes => {
            let c = b.ins().popcnt(x);
            count(b, c)
        }
        MathOp::LeadingZeros => {
            let c = b.ins().clz(x);
            count(b, c)
        }
        MathOp::TrailingZeros => {
            let c = b.ins().ctz(x);
            count(b, c)
        }
        // `rotl` takes the count modulo the width, as Rust does.
        MathOp::RotateLeft => b.ins().rotl(x, args[1]),
    }
}
