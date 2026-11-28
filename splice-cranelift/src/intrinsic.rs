//! Lower integer bit operations to native instructions.
use cranelift_codegen::ir::{InstBuilder, Value, types::I32};
use cranelift_frontend::FunctionBuilder;
use splice_core::hir::MathOp;

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
        // Cranelift's `rotl` accepts an independent scalar integer count.
        // It takes that count modulo the value's width, as Rust does.
        MathOp::RotateLeft => b.ins().rotl(x, args[1]),
    }
}
