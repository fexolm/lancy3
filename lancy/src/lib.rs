#![cfg_attr(feature = "kernel", doc = include_str!("../README.md"))]
#![cfg_attr(
    not(feature = "kernel"),
    doc = "Low-latency CPU JIT with scalar control flow and adaptive specialization."
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
/// Register-based input accepted by the original Lancy builder and macro.
pub mod input;
pub mod jit;
mod lir;

pub use input::{Buffer, BufferArg, BufferKind, Function, FunctionBuilder};
pub use jit::{
    AdaptiveBufferKernel, AdaptiveScalar, ScalarCaller, ScalarFn, ScalarFn3, ScalarKernel, Tier,
};
#[cfg(feature = "kernel")]
pub use splice_macros::lancy_kernel as kernel;

/// Compatibility path for the original instruction builder.
pub mod function {
    #[doc(hidden)]
    #[cfg(feature = "kernel")]
    pub use crate::frontend::scalar::{
        ExpansionGuard, Host, RunArg, ScalarConstant, enter_expansion,
    };
    pub use crate::input::*;
}
