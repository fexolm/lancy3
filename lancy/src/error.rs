use crate::hir::IrError;
pub use splice_target::{TargetError, abi::Trap};
use std::io;
use thiserror::Error;

/// Failure to verify, compile, or install a kernel.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum CompileError {
    #[error(transparent)]
    InvalidIr(#[from] IrError),
    /// The function exceeds a limit of the JIT or the target.
    #[error(transparent)]
    Unsupported(#[from] Limit),
    /// The selected SIMD profile cannot be used.
    #[error(transparent)]
    Target(#[from] TargetError),
    /// The operating system refused to install machine code.
    #[error("{operation} failed while installing machine code")]
    Os {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    /// A defect in Lancy's code generation. The user cannot fix it.
    #[error(transparent)]
    Internal(#[from] InternalError),
}

impl CompileError {
    /// A limit that the smaller body compiled without inlining can stay within.
    pub fn is_limit(&self) -> bool {
        matches!(self, Self::Unsupported(limit) if limit.is_size())
    }

    pub(crate) fn internal(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Internal(InternalError(Box::new(error)))
    }
}

impl From<crate::aarch64::Error> for CompileError {
    fn from(error: crate::aarch64::Error) -> Self {
        Self::internal(error)
    }
}

/// A limit of the JIT or the target that a function exceeds.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum Limit {
    #[error("recursive runtime calls require a dynamic stack")]
    Recursion,
    #[error("scalar frame size overflow")]
    Frame,
    #[error("workspace of {slots} slots exceeds the x64 displacement range")]
    Workspace { slots: usize },
    #[error("function exceeds the register allocator's index range")]
    Registers,
    #[error("branch between code offsets {offset} and {target} exceeds the supported span")]
    BranchRange { offset: usize, target: usize },
    #[error("{bytes} bytes of machine code exceed the 2 GiB a function can address")]
    CodeSize { bytes: usize },
}

impl Limit {
    /// Whether a smaller function can stay within this limit.
    pub fn is_size(self) -> bool {
        !matches!(self, Self::Recursion)
    }
}

/// An internal code generation failure, for example in register allocation
/// or instruction encoding. Its source describes the failing component.
#[derive(Debug, Error)]
#[error("internal code generation error")]
pub struct InternalError(#[source] Box<dyn std::error::Error + Send + Sync>);

/// Failure to invoke a kernel, or a trap raised during its execution.
#[non_exhaustive]
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CallError {
    /// The calling thread cannot run the code, for example because its SVE
    /// vector length is shorter than the compiled kernel requires.
    #[error(transparent)]
    Target(#[from] TargetError),
    #[error(transparent)]
    Signature(#[from] SignatureError),
    /// The kernel stopped. A [`Trap::RustPanic`] message is available from
    /// [`splice_target::abi::take_panic_message`] on the calling thread.
    #[error(transparent)]
    Trap(#[from] Trap),
}

/// The arguments or the entry point do not match the kernel's signature.
#[non_exhaustive]
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SignatureError {
    #[error("expected {expected} arguments, got {actual}")]
    Arity { expected: usize, actual: usize },
    #[error("use call_words or call_words_raw for a multiword signature")]
    WideSignature,
    #[error("expected {expected} result words, got {actual}")]
    ResultWords { expected: usize, actual: usize },
    #[error("signature requires a typed invocation")]
    PointerSignature,
}
