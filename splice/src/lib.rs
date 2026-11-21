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

pub mod debug;
pub mod frontend;
pub mod hir;
pub mod simd;

pub use frontend::ComptimeKey;
pub use hir::{BufferArg, BufferElement, BufferKind, Function, validate_buffers};
pub use splice_macros::{ComptimeKey, kernel};

/// Result of a Splice operation, with the staging or backend error message.
pub type Result<T> = std::result::Result<T, String>;

/// Selects the optimization and source-level debug information for a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildOptions {
    /// Preserve source statements, variables and calls for native debuggers.
    Debug,
    /// Use the backend's optimized pipeline.
    Release,
}

impl BuildOptions {
    /// Select the mode from the caller's `cfg!(debug_assertions)`.
    pub const fn for_profile(debug_assertions: bool) -> Self {
        if debug_assertions {
            Self::Debug
        } else {
            Self::Release
        }
    }
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self::for_profile(cfg!(debug_assertions))
    }
}

/// Generate a backend-specific output from a specialized kernel.
pub trait Codegen {
    /// The value produced by a successful build.
    type Output;
    fn compile(function: &Function) -> Result<Self::Output>;
    /// Override the code generation mode. Source points are available when
    /// the portable IR was staged in debug mode.
    fn compile_with_options(function: &Function, _options: BuildOptions) -> Result<Self::Output> {
        Self::compile(function)
    }
    /// By default, verify a portable function and pass it to the backend.
    /// Native backends may consume the operation stream directly.
    fn build(arity: usize, kinds: &[BufferKind], body: frontend::Body<'_>) -> Result<Self::Output> {
        let _span = tracing::debug_span!(target: "splice::compile", "build", backend = std::any::type_name::<Self>()).entered();
        Self::compile(&frontend::materialize(arity, kinds, body)?)
    }
    /// Stage and compile using the requested mode. Backends that only provide
    /// a listing or have their own mode selection may use the default.
    fn build_with_options(
        arity: usize,
        kinds: &[BufferKind],
        body: frontend::Body<'_>,
        _options: BuildOptions,
    ) -> Result<Self::Output> {
        Self::build(arity, kinds, body)
    }
}

/// The portable calling convention. Scalars and slices have separate
/// argument lists, each following declaration order. An error stops execution;
/// completed stores remain visible to the caller.
pub trait Executable {
    fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32>;
}
