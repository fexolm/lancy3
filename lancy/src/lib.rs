#![doc = include_str!("../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs, unreachable_pub)]

#[cfg(not(any(
    all(target_arch = "x86_64", any(windows, target_os = "linux")),
    all(target_arch = "aarch64", target_os = "linux", target_endian = "little")
)))]
compile_error!("lancy supports x86-64 Windows and Linux, and little-endian AArch64 Linux");

pub mod debug;
mod error;
#[cfg(test)]
mod filetests;
/// Typed SSA with block parameters and stable instruction and block IDs.
pub mod hir;
mod jit;
mod lir;
mod module;
mod support;

pub use error::{CallError, CompileError, InternalError, Limit, SignatureError, Trap};
pub use jit::{CompileOptions, Inlining, Kernel};
/// Targets: extensions, vector width and calling convention.
pub use splice_target::{self as target, Target};

/// The `SPLICE_SIMD` target that tests compile for unless they name one.
#[cfg(test)]
pub(crate) mod test_target {
    use crate::{CompileOptions, Target};

    pub(crate) fn options() -> CompileOptions {
        CompileOptions::host().expect("SPLICE_SIMD names a target of this machine")
    }

    pub(crate) fn target() -> Target {
        options().target
    }

    pub(crate) fn width() -> usize {
        target().vector_bytes()
    }
}
