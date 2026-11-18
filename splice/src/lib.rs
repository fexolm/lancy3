//! The Splice frontend stages a kernel directly into its chosen backend.
//! A portable function can also be materialized for inspection and reuse.
#![forbid(unsafe_code)]

extern crate self as splice;

pub mod frontend;
pub mod hir;

pub use hir::{BufferArg, BufferKind, Function, validate_buffers};
pub use splice_macros::kernel;

/// Generate a backend-specific output from a specialized kernel.
pub trait Codegen {
    /// The complete return type of `kernel::build::<Self>()`, including
    /// any error handling chosen by this backend.
    type Output;
    fn compile(function: &Function) -> Self::Output;
    /// Consume the specialized operation stream. Backends that work on a
    /// verified portable function can use [`frontend::materialize`].
    fn build(arity: usize, kinds: &[BufferKind], body: frontend::Body<'_>) -> Self::Output;
}

/// The portable calling convention. Scalars and slices have separate
/// argument lists, each following declaration order. An error stops execution;
/// completed stores remain visible to the caller.
pub trait Executable {
    fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32, String>;
}
