#![doc = include_str!("../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(all(
    target_arch = "x86_64",
    any(target_os = "windows", target_os = "linux")
)))]
compile_error!("lancy-jit currently targets x86-64 Windows and Linux");

extern crate self as lancy_jit;

#[doc(hidden)]
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
pub use lancy_macros::kernel;

// Keep the original public paths available to clients and older macro expansions.
#[doc(hidden)]
pub use frontend::batch as kernel_support;
pub use hir::projection as ir;
pub mod function {
    #[doc(hidden)]
    pub use crate::frontend::scalar::{
        ExpansionGuard, Host, RunArg, ScalarConstant, enter_expansion,
    };
    pub use crate::hir::function::*;
}
