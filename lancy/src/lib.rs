#![doc = include_str!("../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(any(
    all(target_arch = "x86_64", any(windows, target_os = "linux")),
    all(target_arch = "aarch64", target_os = "linux", target_endian = "little")
)))]
compile_error!("lancy supports x86-64 Windows and Linux, and little-endian AArch64 Linux");

mod call_graph;
mod error;
#[cfg(test)]
mod filetests;
/// Typed SSA with block parameters and stable instruction and block IDs.
pub mod hir;
/// Installation and execution of compiled kernels.
pub mod jit;
mod lir;
mod module;
mod numbers;
pub mod support;

pub use lir::aarch64;

pub use error::{CallError, CompileError};
pub use jit::{Inlining, ScalarKernel};
