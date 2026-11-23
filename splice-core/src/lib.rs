//! The Splice frontend stages a kernel directly into its chosen backend.
//! A portable function can also be materialized for inspection and reuse.
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

mod error;
mod typed;
pub use error::{BuildError, CallError, SignatureError, TargetError, Trap};
#[doc(hidden)]
pub use error::{build_boundary, call_boundary};
pub use typed::{CallArgs, CompileInput, TypedIr, TypedKernel};

pub use frontend::ComptimeKey;
pub use hir::{BufferArg, BufferElement, BufferKind, Function, validate_buffers};
pub use splice_macros::{ComptimeKey, KernelType, kernel};

/// Result of staging or compiling a kernel.
pub type Result<T> = std::result::Result<T, BuildError>;

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
    fn call(
        &self,
        args: &[i32],
        buffers: &mut [BufferArg<'_>],
    ) -> std::result::Result<i32, CallError>;
}
