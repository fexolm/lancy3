mod kernel;
mod memory;
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) mod packed_test;

pub use crate::hir::transforms::passes::Inlining;
pub use kernel::ScalarKernel;
