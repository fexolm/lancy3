//! Kernels of a crate whose overflow checks are the opposite of its profile's
//! `debug_assertions` (see the workspace profiles): its kernels must follow
//! the crate's real `-C overflow-checks`, at run time and while staging.
use splice::kernel;

#[kernel]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Staged arithmetic on a specialization argument, reached only when `flag`.
#[kernel]
pub fn add_known(#[comptime] a: i32, flag: bool) -> i32 {
    if flag { a + 1 } else { 0 }
}
