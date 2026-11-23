use crate::hir::IrError;
use std::io;
use thiserror::Error;

/// Failure to verify, compile, or install a kernel.
#[derive(Debug, Error)]
pub enum CompileError {
    #[error("invalid SSA: {0}")]
    InvalidIr(#[from] IrError),
    #[error("{0}")]
    SimdProfile(String),
    #[error("recursive runtime calls require a dynamic stack")]
    RecursiveCall,
    #[error("scalar frame size overflow")]
    FrameOverflow,
    #[error("workspace of {slots} slots exceeds the x64 displacement range")]
    WorkspaceTooLarge { slots: usize },
    #[error("function exceeds the register allocator's index range")]
    FunctionTooLarge,
    #[error("branch between code offsets {offset} and {target} exceeds the supported span")]
    BranchOutOfRange { offset: usize, target: usize },
    #[error("register allocation failed: {0}")]
    RegisterAllocation(#[from] regalloc2::RegAllocError),
    #[error("x64 assembly failed: {0}")]
    #[cfg(target_arch = "x86_64")]
    Assembly(#[from] iced_x86::IcedError),
    #[error("AArch64 assembly failed: {0}")]
    Aarch64Assembly(#[from] crate::aarch64::Error),
    #[error("cannot install empty machine code")]
    EmptyCode,
    #[error("{operation} failed while installing machine code: {source}")]
    ExecutableMemory {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
}

impl CompileError {
    /// A target size limit that the smaller body compiled without inlining
    /// can stay within.
    pub(crate) fn is_size_limit(&self) -> bool {
        matches!(
            self,
            Self::FunctionTooLarge
                | Self::BranchOutOfRange { .. }
                | Self::WorkspaceTooLarge { .. }
                | Self::FrameOverflow
        )
    }
}

/// Failure to invoke a kernel, or a trap raised during its execution.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum CallError {
    #[error("the calling thread's SVE vector length is smaller than the compiled kernel requires")]
    VectorLength,
    #[error("expected {expected} arguments, got {actual}")]
    Arity { expected: usize, actual: usize },
    #[error("use call_words or call_words_raw for a multiword signature")]
    WideSignature,
    #[error("expected {expected} result words, got {actual}")]
    ResultWords { expected: usize, actual: usize },
    #[error("signature requires a typed invocation")]
    PointerSignature,
    #[error("kernel slice index out of bounds")]
    OutOfBounds,
    #[error("kernel integer division by zero")]
    DivisionByZero,
    #[error("kernel signed division overflow")]
    DivisionOverflow,
    #[error("kernel integer arithmetic overflow")]
    IntegerOverflow,
    #[error("kernel Rust import panicked")]
    RustPanic,
}
