//! Lower HIR to SSA LIR, then allocate and emit code for the target.
mod ssa;
mod x64;

pub(crate) use x64::{buffer_header_words, emit};
