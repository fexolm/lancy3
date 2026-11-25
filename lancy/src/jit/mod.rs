mod kernel;
mod memory;
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) mod packed_test;

pub use kernel::ScalarKernel;

/// Machine offsets associated with debug points in a single compiled body.
#[derive(Debug)]
pub struct DebugFunction {
    pub start: u64,
    pub end: u64,
    pub workspace_offset: usize,
    pub points: Vec<(u32, u64)>,
    /// The active frame, after the prologue and before the epilogue.
    pub frame: DebugFrame,
}

/// Stack-based unwind rules for the active portion of a compiled body.
#[derive(Debug)]
pub struct DebugFrame {
    pub start: u64,
    pub end: u64,
    pub cfa_from_sp: i32,
    /// (DWARF register, byte offset relative to the CFA).
    pub saved: Vec<(u16, i32)>,
    /// (DWARF register, byte offset relative to the workspace register).
    pub workspace_saved: Vec<(u16, i64)>,
}

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
    pub target: Target,
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

    pub fn with_inlining(self, inlining: Inlining) -> Self {
        Self { inlining, ..self }
    }

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
            .in_scope(|| passes::inline_all(source))
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
