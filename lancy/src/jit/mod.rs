mod kernel;
mod memory;
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) mod packed_test;

pub use kernel::ScalarKernel;

#[cfg(test)]
mod tests;

use crate::CompileError;
use crate::hir::{Function, ssa::VerifiedBody, transforms::passes};
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
    mut lower: impl FnMut(Module<VerifiedBody<'_>>, &FunctionIds<Function>) -> Result<T, CompileError>,
) -> Result<T, CompileError> {
    let _span =
        tracing::debug_span!(target: "splice::compile", "optimize_and_lower_lancy").entered();
    let mut emit = |body: Function| {
        let (module, ids) = body.verified_module()?;
        lower(module, &ids)
    };
    if inlining.should_inline(source) {
        let expanded = tracing::debug_span!(target: "splice::compile", "inline_lancy_hir")
            .in_scope(|| passes::inline_all(source, ids))
            .and_then(|mut body| {
                tracing::debug_span!(target: "splice::compile", "optimize_lancy_hir")
                    .in_scope(|| passes::simplify(&mut body))?;
                emit(body)
            });
        match expanded {
            // Lowering can also run out of virtual registers or branch range.
            Err(CompileError::FunctionTooLarge | CompileError::BranchOutOfRange { .. }) => {}
            other => return other,
        }
    }
    let body = tracing::debug_span!(target: "splice::compile", "optimize_lancy_hir")
        .in_scope(|| passes::simplify_bodies(source, ids))?;
    emit(body)
}
