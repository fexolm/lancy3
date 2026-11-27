mod kernel;
mod memory;
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) mod packed_test;
mod workspace;

pub use kernel::Kernel;

#[cfg(test)]
pub(crate) mod tests;

use crate::CompileError;
use crate::hir::{Function, ssa::VerifiedBody, transforms::passes};
use crate::module::Module;
use splice_target::Target;
use splice_target::TargetError;

/// What to compile for and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompileOptions {
    /// The machine to generate code for.
    pub target: Target,
    /// Whether calls are expanded in release code.
    pub inlining: Inlining,
    /// Keep source points, outlined calls and debugger frame layouts.
    /// Debug code is never inlined, whatever `inlining` says.
    pub debug: bool,
}

impl CompileOptions {
    /// Optimized code for `target`, with calls inlined.
    pub fn new(target: Target) -> Self {
        Self {
            target,
            inlining: Inlining::Always,
            debug: false,
        }
    }

    /// Optimized code for [`Target::host`]: the `SPLICE_SIMD` profile, or
    /// the best of this machine.
    pub fn host() -> Result<Self, TargetError> {
        Target::host().map(Self::new)
    }

    /// These options with `inlining`.
    pub fn with_inlining(self, inlining: Inlining) -> Self {
        Self { inlining, ..self }
    }

    /// These options for a debug build.
    pub fn with_debug(self) -> Self {
        Self {
            debug: true,
            ..self
        }
    }
}

/// Controls expansion of runtime calls in HIR.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Inlining {
    /// Expand the entire acyclic graph when the resulting function is representable.
    #[default]
    Always,
    /// Keep the internal call ABI while optimizing each body independently.
    Never,
}

impl Inlining {
    pub(crate) fn should_inline(self, source: &Module<VerifiedBody<'_>>) -> bool {
        self == Self::Always && source.functions.len() > 1
    }
}

/// Lower optimized HIR, retrying with calls if the expanded form exceeds a target limit.
pub(crate) fn optimize_and_lower<T>(
    source: &Module<VerifiedBody<'_>>,
    inlining: Inlining,
    mut lower: impl FnMut(Module<VerifiedBody<'_>>) -> Result<T, CompileError>,
) -> Result<T, CompileError> {
    let _span =
        tracing::debug_span!(target: "lancy::compile", "optimize_and_lower_lancy").entered();
    let mut emit = |body: Function| {
        let module = body.verified_module()?;
        lower(module)
    };
    if inlining.should_inline(source) {
        let expanded = tracing::debug_span!(target: "lancy::compile", "inline_lancy_hir")
            .in_scope(|| passes::inline_all(source, crate::lir::MAX_VALUES))
            .and_then(|mut body| {
                tracing::debug_span!(target: "lancy::compile", "optimize_lancy_hir")
                    .in_scope(|| passes::simplify(&mut body))?;
                emit(body)
            });
        match expanded {
            // Inlining or lowering the expanded body can exceed a size limit.
            Err(error) if error.is_limit() => {}
            other => return other,
        }
    }
    let body = tracing::debug_span!(target: "lancy::compile", "optimize_lancy_hir")
        .in_scope(|| passes::simplify_bodies(source))?;
    emit(body)
}
