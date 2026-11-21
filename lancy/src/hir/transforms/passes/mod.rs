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
mod prune;
mod sccp;
mod simplify;
#[cfg(test)]
mod tests;

pub(crate) use inline::inline_all;
pub(crate) use simplify::{simplify, simplify_bodies};
