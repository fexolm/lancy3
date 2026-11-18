//! Instruction selection produces SSA LIR. Allocation resolves virtual values
//! and block parameters into a separate machine program consumed by x64 emission.
mod ssa;
mod x64;
pub(crate) use x64::scalar::{buffer_header_words, emit_ssa, emit_ssa3};
