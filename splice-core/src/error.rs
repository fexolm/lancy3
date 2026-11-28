//! Errors of the two phases: building a kernel and calling it.
pub use splice_target::{TargetError, abi::Trap};
use std::panic::{AssertUnwindSafe, catch_unwind};
use thiserror::Error;

/// Failure to call a compiled kernel. Stores that completed before the
/// error remain visible to the caller.
#[non_exhaustive]
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CallError {
    /// The kernel stopped with a runtime trap. A panic in Rust code is
    /// reported as [`CallError::RustPanic`], never as `Trap::RustPanic`.
    #[error(transparent)]
    Trap(Trap),
    /// A Rust import, or a kernel that runs as Rust, panicked.
    #[error("Rust code panicked: {0}")]
    RustPanic(String),
    /// The arguments do not match the compiled kernel.
    #[error(transparent)]
    Signature(#[from] SignatureError),
    /// The calling thread cannot run the compiled code.
    #[error(transparent)]
    Target(#[from] TargetError),
}

impl From<Trap> for CallError {
    /// A Rust panic takes the message its native boundary kept on this thread.
    fn from(trap: Trap) -> Self {
        match trap {
            Trap::RustPanic => Self::RustPanic(
                splice_target::abi::take_panic_message().unwrap_or_else(|| trap.to_string()),
            ),
            trap => Self::Trap(trap),
        }
    }
}

/// The arguments of a call do not match the kernel's signature.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum SignatureError {
    #[error("expected {expected} scalar arguments, got {actual}")]
    Scalars { expected: usize, actual: usize },
    /// The word has bits set above the value of its type (see [`crate::KernelScalar`]).
    #[error("scalar argument {index} is not a {ty} word")]
    Word {
        index: usize,
        ty: crate::hir::ScalarType,
    },
    #[error("expected {expected} buffers, got {actual}")]
    Buffers { expected: usize, actual: usize },
    #[error("buffer {buffer} expects {} elements", expected.name())]
    Element {
        buffer: usize,
        expected: crate::hir::ScalarType,
    },
    #[error("buffer {buffer} has the wrong Rust context signature")]
    Context { buffer: usize },
    #[error("buffer {buffer} requires a &str")]
    Utf8 { buffer: usize },
    #[error("buffer {buffer} requires a writable slice")]
    ReadOnly { buffer: usize },
    #[error("buffer {buffer} length exceeds i32::MAX")]
    Length { buffer: usize },
}

/// Failure to build a kernel. No part of a failed build is kept.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum BuildError {
    /// Specialization rejected the kernel body, for example a runtime value
    /// where the body needs a known one, or an exhausted resource budget.
    #[error("{0}")]
    Staging(String),
    /// Rust would panic on a host operation that `comptime!` requires.
    #[error("{0} in comptime!")]
    Comptime(Trap),
    /// Rust code panicked during the build.
    #[error("Rust code panicked during the build: {0}")]
    RustPanic(String),
    /// The staged or portable IR is invalid.
    #[error("invalid kernel IR{}: {reason}", instr.map(|i| format!(" at instruction {i}")).unwrap_or_default())]
    Verify {
        instr: Option<usize>,
        reason: String,
    },
    /// The backend, the builder or the target cannot express the kernel.
    #[error("{0}")]
    Unsupported(String),
    /// The selected SIMD profile cannot be used.
    #[error(transparent)]
    Target(#[from] TargetError),
    /// Code generation failed inside a backend.
    #[error("{backend} failed to compile the kernel")]
    Backend {
        backend: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl BuildError {
    #[doc(hidden)]
    pub fn staging(message: impl Into<String>) -> Self {
        Self::Staging(message.into())
    }

    pub fn verify(reason: impl Into<String>) -> Self {
        Self::Verify {
            instr: None,
            reason: reason.into(),
        }
    }

    pub fn verify_at(instr: usize, reason: impl Into<String>) -> Self {
        Self::Verify {
            instr: Some(instr),
            reason: reason.into(),
        }
    }

    /// Attribute a verification error to an instruction of the caller.
    pub fn at(self, instr: usize) -> Self {
        match self {
            Self::Verify { reason, .. } => Self::verify_at(instr, reason),
            error => error,
        }
    }

    pub fn unsupported(feature: impl Into<String>) -> Self {
        Self::Unsupported(feature.into())
    }

    pub fn backend(
        backend: &'static str,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Backend {
            backend,
            source: source.into(),
        }
    }
}

#[doc(hidden)]
pub fn build_boundary<T>(action: impl FnOnce() -> Result<T, BuildError>) -> Result<T, BuildError> {
    // Returning an error abandons this build. Mutations to the caller's
    // arguments made before an unwind retain their Rust semantics.
    catch_unwind(AssertUnwindSafe(action)).unwrap_or_else(|payload| {
        Err(BuildError::RustPanic(splice_target::abi::panic_message(
            payload,
        )))
    })
}

/// Run a call that may execute Rust code. A panic becomes
/// [`CallError::RustPanic`]; stores made before it remain visible.
#[doc(hidden)]
pub fn call_boundary<T>(action: impl FnOnce() -> Result<T, CallError>) -> Result<T, CallError> {
    catch_unwind(AssertUnwindSafe(action)).unwrap_or_else(|payload| {
        Err(CallError::RustPanic(splice_target::abi::panic_message(
            payload,
        )))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn a_panic_while_building_is_a_typed_error_with_its_message() {
        let error =
            build_boundary(|| -> Result<(), BuildError> { panic!("staging failed") }).unwrap_err();
        assert!(matches!(&error, BuildError::RustPanic(message) if message == "staging failed"));
        assert_eq!(
            error.to_string(),
            "Rust code panicked during the build: staging failed"
        );
    }

    #[test]
    fn build_errors_name_their_cause_once() {
        let backend = BuildError::backend("llvm", "module verification failed");
        assert_eq!(backend.to_string(), "llvm failed to compile the kernel");
        assert_eq!(
            backend.source().unwrap().to_string(),
            "module verification failed"
        );
        assert_eq!(
            BuildError::verify("undefined label").at(3).to_string(),
            "invalid kernel IR at instruction 3: undefined label"
        );
        assert_eq!(
            BuildError::verify("empty scalar function").to_string(),
            "invalid kernel IR: empty scalar function"
        );
        assert_eq!(
            BuildError::Comptime(Trap::Bounds).to_string(),
            "kernel slice index out of bounds in comptime!"
        );
    }
}
