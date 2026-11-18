#![doc = include_str!("../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(all(target_arch = "x86_64", any(windows, target_os = "linux"))))]
compile_error!("lancy currently targets x86-64 Windows and Linux");

pub mod hir;
pub mod jit;
mod lir;

pub use hir::{Buffer, BufferArg, BufferKind};
pub use jit::ScalarKernel;
