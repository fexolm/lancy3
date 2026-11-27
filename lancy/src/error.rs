use crate::hir::IrError;
use splice_target::TargetError;
pub use splice_target::abi::Trap;
use std::io;
use thiserror::Error;

/// Failure to verify, compile, or install a kernel.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum CompileError {
    /// The function or a callee fails verification.
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
        /// The system call that failed.
        operation: &'static str,
        /// The operating system's error.
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

impl From<crate::lir::aarch64::EncodeError> for CompileError {
    fn from(error: crate::lir::aarch64::EncodeError) -> Self {
        Self::internal(error)
    }
}

/// A limit of the JIT or the target that a function exceeds.
#[expect(missing_docs, reason = "each variant's #[error] message documents it")]
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
    /// The arguments or result buffer do not fit the signature.
    #[error(transparent)]
    Signature(#[from] SignatureError),
    /// The kernel stopped. A [`Trap::RustPanic`] message is available from
    /// [`splice_target::abi::take_panic_message`] on the calling thread.
    #[error(transparent)]
    Trap(#[from] Trap),
}

/// The arguments or the entry point do not match the kernel's signature.
#[expect(missing_docs, reason = "the fields are the expected and actual counts")]
#[non_exhaustive]
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SignatureError {
    /// [`Kernel::call`](crate::Kernel::call) got the wrong number of
    /// arguments.
    #[error("expected {expected} arguments, got {actual}")]
    Arity { expected: usize, actual: usize },
    /// [`Kernel::call`](crate::Kernel::call) needs a function of `i32`
    /// parameters and an `i32` result.
    #[error("the signature is not i32 parameters and an i32 result; use call_packed")]
    NotI32,
    /// The packed arguments do not fill the parameters' words.
    #[error("expected {expected} argument words, got {actual}")]
    ArgumentWords { expected: usize, actual: usize },
    /// The result buffer does not match the result's words.
    #[error("expected {expected} result words, got {actual}")]
    ResultWords { expected: usize, actual: usize },
    /// The signature passes pointers, or the code builds them from call
    /// scratch, so only the unsafe entry may call it.
    #[error("the signature passes pointers; use call_packed_unchecked")]
    PointerSignature,
}
