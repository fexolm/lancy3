//! Splice stages Rust kernels into an enabled code generation backend.
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
//! Only a kernel's own module can give a build its call interface: the
//! output of `second` cannot be retyped as `first`'s.
//! ```compile_fail,E0603
//! use splice::{TypedKernel, kernel, lancy::Lancy};
//! #[kernel]
//! fn first(x: i32) -> i32 { x }
//! #[kernel]
//! fn second(x: i32) -> i32 { -x }
//! let code = second::build::<Lancy>(second::BuildArgs {}).unwrap().into_inner();
//! let _ = TypedKernel::new(code, first::Signature(std::marker::PhantomData));
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

pub use splice_core::{
    Backend, BufferArg, BufferElement, BufferKind, BuildError, BuildOptions, CallArgs, CallError,
    Codegen, CompileInput, ComptimeKey, Executable, Function, KernelScalar, KernelType, Math, Mode,
    Result, SignatureError, Target, TargetError, Trap, TypedIr, TypedKernel, debug, frontend, hir,
    intrinsic, kernel, math, simd, target, validate_buffers,
};

/// Items that generated code names. Not a stable API.
#[doc(hidden)]
pub use splice_core::__private;

/// Lancy, Splice's own JIT (feature `lancy`, on by default).
#[cfg(feature = "lancy")]
pub mod lancy {
    pub use splice_lancy::{Lancy, LancyKernel};
}

/// Cranelift (feature `cranelift`).
#[cfg(feature = "cranelift")]
pub mod cranelift {
    pub use splice_cranelift::{Cranelift, CraneliftKernel};
}

/// LLVM 19 (feature `llvm`). [`llvm::LlvmManualSimd`] keeps only the SIMD the
/// kernel spells out: O3 without loop and SLP vectorization.
#[cfg(feature = "llvm")]
pub mod llvm {
    pub use splice_llvm::{Llvm, LlvmKernel, LlvmManualSimd, to_llvm_ir};
}

#[cfg(doctest)]
#[doc = include_str!("../../README.md")]
mod readme {}

#[cfg(doctest)]
#[doc = include_str!("../../docs/splice-language.md")]
mod language {}

#[cfg(doctest)]
#[doc = include_str!("../../docs/splice-api-cases.md")]
mod api_cases {}
