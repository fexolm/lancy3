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
