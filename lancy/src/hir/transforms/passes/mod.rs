//! HIR optimization passes, shared by all targets.
mod bounds;
mod dead;
#[cfg(feature = "experimental-simd")]
mod experimental;
mod facts;
mod inline;
mod local_cfg;
mod loops;
mod memory;
mod pipeline;
mod prune;
mod sccp;
mod simplify;
#[cfg(test)]
mod tests;

pub(crate) use inline::inline_all;

/// In test and debug builds, verify a function after a pass, so a pass that
/// breaks HIR is named where it ran instead of failing later in code generation.
fn checkpoint(f: &crate::hir::Function, pass: &str) {
    if cfg!(any(test, debug_assertions))
        && let Err(error) = f.verify()
    {
        panic!("HIR pass `{pass}` produced invalid IR: {error}");
    }
}
pub(crate) use simplify::{simplify, simplify_bodies};
