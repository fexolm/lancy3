//! The Splice frontend stages a kernel directly into its chosen backend.
//! A portable function can also be materialized for inspection and reuse.
//!
//! Kernels have distinct call argument types, even when their fields match.
//! ```compile_fail,E0271
//! use splice::{Executable, TypedKernel, kernel};
//! #[kernel]
//! fn first(x: i32) -> i32 { x }
//! #[kernel]
//! fn second(x: i32) -> i32 { x }
//! fn call<K: Executable>(code: TypedKernel<first::Signature, K>) {
//!     let _ = code.call(second::CallArgs { x: 1 });
//! }
//! ```
//!
//! Publish associated functions by marking their enclosing block.
//! ```
//! struct Plan;
//! #[splice::kernel]
//! impl Plan {
//!     fn zero() -> i32 { 0 }
//! }
//! ```
//! Marking only the method is rejected. Stable rustdoc does not check error
//! codes, so each rejected example differs from the accepted one above it in
//! the rejected property alone.
//! ```compile_fail
//! struct Plan;
//! impl Plan {
//!     #[splice::kernel]
//!     fn zero() -> i32 { 0 }
//! }
//! ```
//!
//! A published trait implementation requires a published trait declaration,
//! even when the implementation inherits every method.
//! ```
//! #[splice::kernel]
//! trait Plan { fn zero() -> i32 { 0 } }
//! #[splice::kernel]
//! impl Plan for () {}
//! ```
//! ```compile_fail
//! trait Plan { fn zero() -> i32 { 0 } }
//! #[splice::kernel]
//! impl Plan for () {}
//! ```
#![deny(unsafe_code)]

extern crate self as splice;

pub mod debug;
pub mod frontend;
pub mod hir;
mod imports;
pub mod simd;
#[doc(hidden)]
pub use imports::ImportInvocation;
pub use imports::{ImportArgs, ImportTarget, LinkedImport, RustImport};
#[allow(unsafe_code)]
mod runtime_context;
#[doc(hidden)]
pub use runtime_context::{
    ContextArguments, ContextFrame, ContextOperation, ContextPacketLayout, ContextSignature,
    RuntimeContext, StaticRustCall, StaticRustSymbol, invoke_packet_runtime_context,
    invoke_runtime_context,
};

/// Description of an actual Rust struct field for this monomorphization.
pub struct KernelField {
    pub name: &'static str,
    pub offset: usize,
    pub type_name: fn() -> &'static str,
}
pub trait KernelType {
    const FIELDS: &'static [KernelField];
}

mod typed;
pub use typed::{BuildError, CallArgs, CallError, CompileInput, TypedIr, TypedKernel};
#[doc(hidden)]
pub use typed::{build_boundary, call_boundary};

pub use frontend::ComptimeKey;
pub use hir::{BufferArg, BufferElement, BufferKind, Function, validate_buffers};
pub use splice_macros::{ComptimeKey, KernelType, kernel};

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
pub trait Codegen: Sized {
    /// The value produced by a successful build.
    type Output;
    fn compile_raw(function: &Function) -> Result<Self::Output>;
    fn compile<I: CompileInput<Self>>(input: &I) -> std::result::Result<I::Output, I::Error> {
        input.compile_with(None)
    }
    /// Override the code generation mode. Source points are available when
    /// the portable IR was staged in debug mode.
    fn compile_with_options<I: CompileInput<Self>>(
        input: &I,
        options: BuildOptions,
    ) -> std::result::Result<I::Output, I::Error> {
        input.compile_with(Some(options))
    }
    fn compile_raw_with_options(
        function: &Function,
        _options: BuildOptions,
    ) -> Result<Self::Output> {
        Self::compile_raw(function)
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
