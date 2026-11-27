//! Kernel intrinsics: functions and methods marked `#[intrinsic("name")]`.
//!
//! An intrinsic is an ordinary kernel function whose call reaches the
//! backend as the named operation. A backend that implements the name for
//! the call's signature emits its own code and the body is not staged.
//! Otherwise [`BuildOptions::unsupported_intrinsic`](crate::BuildOptions)
//! decides: [`IntrinsicPolicy::Fallback`] logs the intrinsic and the
//! backend and compiles the body, [`IntrinsicPolicy::Error`] fails the
//! build. The name is the user's string; the operand and result types come
//! from the Rust signature.
use crate::frontend::{Body, Signature, Specialization};
use crate::hir::Function;
use std::sync::Arc;

/// What a build does with an intrinsic its backend does not implement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum IntrinsicPolicy {
    /// Compile the intrinsic's body and log a warning with target
    /// `splice::intrinsic`.
    #[default]
    Fallback,
    /// Fail the build with [`BuildError::Unsupported`](crate::BuildError).
    Error,
}

/// A call of an intrinsic: its name and the signature of the called
/// function. An intrinsic takes scalar operands and has no buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Intrinsic<'a> {
    pub name: &'a str,
    pub signature: Signature<'a>,
}

impl<'a> Intrinsic<'a> {
    pub const fn new(name: &'a str, signature: Signature<'a>) -> Self {
        Self { name, signature }
    }
}

impl std::fmt::Display for Intrinsic<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}`(", self.name)?;
        for (i, ty) in self.signature.params.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{ty}")?;
        }
        write!(f, ") -> {}", self.signature.result)
    }
}

/// The body of an intrinsic, staged when the backend does not implement it.
pub enum Fallback<'a> {
    /// The body as the kernel stages it. Calls with the same `key` share
    /// one staged function.
    Staged {
        key: Option<Specialization>,
        body: Body<'a>,
    },
    /// The body recorded in portable IR.
    Function(&'a Arc<Function>),
}

/// Apply `policy` to `intrinsic`, which `backend` does not implement:
/// `Ok` after logging under [`IntrinsicPolicy::Fallback`], an error naming
/// both under [`IntrinsicPolicy::Error`].
pub fn unsupported(
    policy: IntrinsicPolicy,
    backend: &str,
    intrinsic: &Intrinsic<'_>,
) -> crate::Result<()> {
    match policy {
        IntrinsicPolicy::Fallback => {
            tracing::warn!(
                target: "splice::intrinsic",
                intrinsic = intrinsic.name,
                backend,
                "unsupported intrinsic {intrinsic} on backend `{backend}`; compiling its body"
            );
            Ok(())
        }
        IntrinsicPolicy::Error => Err(crate::BuildError::unsupported(format!(
            "intrinsic {intrinsic} on backend `{backend}`"
        ))),
    }
}
