//! The Splice frontend stages a kernel directly into its chosen backend.
//! A portable function can also be materialized for inspection and reuse.
//!
//! A runtime branch cannot mutate a collection captured from specialization.
//! Pass a writable slice for runtime state.
//! ```compile_fail,E0502
//! use splice::kernel;
//! #[kernel]
//! fn conditional_push(x: i32) -> i32 {
//!     let mut values = vec![0];
//!     if x > 0 { values.push(1); }
//!     values.len() as i32
//! }
//! ```
#![forbid(unsafe_code)]

extern crate self as splice;

pub mod frontend;
pub mod hir;
pub mod simd;

pub use frontend::ComptimeKey;
pub use hir::{BufferArg, BufferElement, BufferKind, Function, validate_buffers};
pub use splice_macros::{ComptimeKey, kernel};

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
        let _span = tracing::debug_span!(target: "splice::compile", "build", backend = std::any::type_name::<Self>()).entered();
        Self::compile(&frontend::materialize(arity, kinds, body)?)
    }
}

/// The portable calling convention. Scalars and slices have separate
/// argument lists, each following declaration order. An error stops execution;
/// completed stores remain visible to the caller.
pub trait Executable {
    fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32>;
}
