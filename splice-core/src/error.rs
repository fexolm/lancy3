//! Errors of the two phases: building a kernel and calling it.
use crate::simd::Element;
pub use splice_target::{TargetError, abi::Trap};
use std::{
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
};
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
    #[error("expected {expected} buffers, got {actual}")]
    Buffers { expected: usize, actual: usize },
    #[error("buffer {buffer} expects {} elements", expected.name())]
    Element { buffer: usize, expected: Element },
    #[error("buffer {buffer} has the wrong Rust context signature")]
    Context { buffer: usize },
    #[error("buffer {buffer} requires a writable slice")]
    ReadOnly { buffer: usize },
    #[error("buffer {buffer} length exceeds i32::MAX")]
    Length { buffer: usize },
}

macro_rules! phase_error {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct $name(String);
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
        impl std::error::Error for $name {}
        impl From<String> for $name {
            fn from(message: String) -> Self {
                Self(message)
            }
        }
    };
}
phase_error!(BuildError);

#[doc(hidden)]
pub fn build_boundary<T>(action: impl FnOnce() -> crate::Result<T>) -> Result<T, BuildError> {
    // Returning an error abandons this build. Mutations to the caller's
    // arguments made before an unwind retain their Rust semantics.
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(value) => value.map_err(BuildError::from),
        Err(payload) => Err(BuildError::from(splice_target::abi::panic_message(payload))),
    }
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
