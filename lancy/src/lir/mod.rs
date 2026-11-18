//! Low-level representation between verified HIR and native machine code.
//! Batch kernels use an allocated instruction stream. Scalar functions retain
//! HIR instruction order and carry layout and frame information alongside it.

mod batch;
mod scalar;
mod x64;

pub(crate) use x64::batch::emit as emit_batch;
pub(crate) use x64::scalar::{buffer_header_words, emit as emit_scalar, emit3 as emit_scalar3};
