//! The Splice frontend stages a kernel directly into its chosen backend.
//! A portable function can also be materialized for inspection and reuse.
#![forbid(unsafe_code)]

extern crate self as splice;

pub mod frontend;
pub mod hir;

pub use hir::{BufferArg, BufferKind, Function, validate_buffers};
pub use splice_macros::kernel;

/// Result of a Splice operation, with the staging or backend error message.
pub type Result<T> = std::result::Result<T, String>;

/// Generate a backend-specific output from a specialized kernel.
pub trait Codegen {
    /// The value produced by a successful build.
    type Output;
    fn compile(function: &Function) -> Result<Self::Output>;
    /// By default, verify a portable function and pass it to the backend.
    /// Native backends may consume the operation stream directly.
    fn build(arity: usize, kinds: &[BufferKind], body: frontend::Body<'_>) -> Result<Self::Output> {
        Self::compile(&frontend::materialize(arity, kinds, body)?)
    }
}

/// The portable calling convention. Scalars and slices have separate
/// argument lists, each following declaration order. An error stops execution;
/// completed stores remain visible to the caller.
pub trait Executable {
    fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32>;
}
