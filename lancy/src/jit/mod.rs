mod kernel;
mod memory;
#[cfg(test)]
pub(crate) mod packed_test;

pub use kernel::{Inlining, ScalarKernel};
