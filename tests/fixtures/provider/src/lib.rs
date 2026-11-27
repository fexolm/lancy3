//! Kernels declared in a crate other than the tests that use them: the
//! tests check what `#[kernel]` exports to another crate. A separate
//! package is also the only way to give kernels their own overflow-check
//! profile (see the workspace profiles).
pub mod overflow;
pub mod traits;
