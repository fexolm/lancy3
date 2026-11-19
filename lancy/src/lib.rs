#![doc = include_str!("../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(all(target_arch = "x86_64", any(windows, target_os = "linux"))))]
compile_error!("lancy currently targets x86-64 Windows and Linux");

mod call_graph;
mod error;
/// Typed SSA with block parameters and stable instruction and block IDs.
pub mod hir;
/// Installation and execution of compiled kernels.
pub mod jit;
mod lir;
mod module;
mod numbers;
pub mod support;

pub use error::{CallError, CompileError};
pub use jit::{Inlining, ScalarKernel};
