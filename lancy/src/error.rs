use crate::hir::IrError;
use std::io;
use thiserror::Error;

/// Failure to verify, compile, or install a kernel.
#[derive(Debug, Error)]
pub enum CompileError {
    #[error("invalid SSA: {0}")]
    InvalidIr(#[from] IrError),
    #[error("recursive runtime calls require a dynamic stack")]
    RecursiveCall,
    #[error("scalar frame size overflow")]
    FrameOverflow,
    #[error("workspace of {slots} slots exceeds the x64 displacement range")]
    WorkspaceTooLarge { slots: usize },
    #[error("function exceeds the register allocator's index range")]
    FunctionTooLarge,
    #[error("x64 code offsets {offset} and {target} exceed the supported rel32 span")]
    BranchOutOfRange { offset: usize, target: usize },
    #[error("register allocation failed: {0}")]
    RegisterAllocation(#[from] regalloc2::RegAllocError),
    #[error("x64 assembly failed: {0}")]
    Assembly(#[from] iced_x86::IcedError),
    #[error("cannot install empty machine code")]
    EmptyCode,
    #[error("{operation} failed while installing machine code: {source}")]
    ExecutableMemory {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
}

/// Failure to invoke a kernel, or a trap raised during its execution.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum CallError {
    #[error("expected {expected} arguments, got {actual}")]
    Arity { expected: usize, actual: usize },
    #[error("use call_raw for a pointer signature")]
    PointerSignature,
    #[error("kernel slice index out of bounds")]
    OutOfBounds,
    #[error("kernel integer division by zero")]
    DivisionByZero,
    #[error("kernel signed division overflow")]
    DivisionOverflow,
}
