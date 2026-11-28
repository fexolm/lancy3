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
    Codegen, CompileInput, ComptimeKey, Executable, Function, KernelScalar, KernelType, Mode,
    Result, SignatureError, Target, TargetError, Trap, TypedIr, TypedKernel, debug, frontend, hir,
    target, validate_buffers,
};

pub use splice_macros::{ComptimeKey, KernelType, kernel};

/// Standard kernel operations.
pub mod std {
    pub mod intrinsics;
}

/// Items that generated code names. Not a stable API.
#[doc(hidden)]
pub mod __private {
    use crate::{
        hir::ValueType,
        std::intrinsics::simd::{Lane, LaneElement, Mask},
    };
    use core::marker::PhantomData;
    pub use splice_core::__private::*;

    /// Resolve a concrete Rust type alias during staging.
    pub trait SimdType {
        fn simd_type(self) -> Option<ValueType>;
    }
    impl<T: LaneElement> SimdType for &PhantomData<Lane<T>> {
        fn simd_type(self) -> Option<ValueType> {
            Some(ValueType::Vector(T::ELEMENT))
        }
    }
    impl<T: LaneElement> SimdType for &PhantomData<Mask<T>> {
        fn simd_type(self) -> Option<ValueType> {
            Some(ValueType::Mask(T::ELEMENT))
        }
    }
    impl<T: ?Sized> SimdType for &&PhantomData<T> {
        fn simd_type(self) -> Option<ValueType> {
            None
        }
    }

    /// Restore a captured width inside a Rust object expression. An unrelated
    /// associated function keeps its own result type and evaluation order.
    pub trait ContextWidth<R> {
        fn context_width(self, from_ir: impl FnOnce() -> usize, ordinary: impl FnOnce() -> R) -> R;
    }
    impl<T: LaneElement> ContextWidth<usize> for &PhantomData<Lane<T>> {
        fn context_width(
            self,
            from_ir: impl FnOnce() -> usize,
            _: impl FnOnce() -> usize,
        ) -> usize {
            from_ir()
        }
    }
    impl<T: ?Sized, R> ContextWidth<R> for &&PhantomData<T> {
        fn context_width(self, _: impl FnOnce() -> usize, ordinary: impl FnOnce() -> R) -> R {
            ordinary()
        }
    }
}

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
