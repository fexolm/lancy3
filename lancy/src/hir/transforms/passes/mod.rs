//! HIR optimization passes, shared by all targets.
#[cfg(feature = "experimental-simd")]
mod experimental;
mod inline;
mod local_cfg;
mod loops;
mod prune;
mod sccp;
mod simplify;
#[cfg(test)]
mod tests;

pub(crate) use inline::inline_all;
pub(crate) use simplify::{simplify, simplify_bodies};

use crate::CompileError;
use crate::hir::{Function, ssa::VerifiedBody};
use crate::module::{FunctionIds, Module};

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
    ids: &FunctionIds<Function>,
    inlining: Inlining,
    lower: impl Fn(Module<VerifiedBody<'_>>, &FunctionIds<Function>) -> Result<T, CompileError>,
) -> Result<T, CompileError> {
    let _span =
        tracing::debug_span!(target: "splice::compile", "optimize_and_lower_lancy").entered();
    let emit = |body: Function| {
        let (module, ids) = body.verified_module()?;
        lower(module, &ids)
    };
    if inlining.should_inline(source) {
        let expanded = tracing::debug_span!(target: "splice::compile", "inline_lancy_hir")
            .in_scope(|| inline_all(source, ids))
            .and_then(|mut body| {
                tracing::debug_span!(target: "splice::compile", "optimize_lancy_hir")
                    .in_scope(|| simplify(&mut body))?;
                emit(body)
            });
        match expanded {
            // Lowering can also run out of virtual registers or branch range.
            Err(CompileError::FunctionTooLarge | CompileError::BranchOutOfRange { .. }) => {}
            other => return other,
        }
    }
    let body = tracing::debug_span!(target: "splice::compile", "optimize_lancy_hir")
        .in_scope(|| simplify_bodies(source, ids))?;
    emit(body)
}
