#![cfg_attr(feature = "kernel", doc = include_str!("../README.md"))]
#![cfg_attr(
    not(feature = "kernel"),
    doc = "Low-latency CPU JIT with scalar control flow and adaptive SIMD projections."
)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(all(
    target_arch = "x86_64",
    any(target_os = "windows", target_os = "linux")
)))]
compile_error!("lancy currently targets x86-64 Windows and Linux");

extern crate self as lancy;

#[doc(hidden)]
#[cfg(feature = "kernel")]
pub mod frontend;
pub mod hir;
pub mod jit;
mod lir;

pub use hir::{
    Assumption, Buffer, BufferArg, BufferKind, BuildError, BuildValue, Builder, Function,
    FunctionBuilder, Op, Program, Value,
};
pub use jit::{
    AdaptiveBufferKernel, AdaptiveKernel, AdaptiveScalar, KernelStats, MappedKernel, NativeKernel,
    ScalarCaller, ScalarFn, ScalarFn3, ScalarKernel, Tier,
};
#[cfg(feature = "kernel")]
pub use splice_macros::lancy_kernel as kernel;

// Keep the original public paths available to clients and older macro expansions.
#[doc(hidden)]
#[cfg(feature = "kernel")]
pub use frontend::batch as kernel_support;
pub use hir::projection as ir;
pub mod function {
    #[doc(hidden)]
    #[cfg(feature = "kernel")]
    pub use crate::frontend::scalar::{
        ExpansionGuard, Host, RunArg, ScalarConstant, enter_expansion,
    };
    pub use crate::hir::function::*;
}
