//! Kernel bit operations with Rust integer semantics.

use super::intrinsic;
#[doc(hidden)]
pub use __splice_math as Math;

/// The built-in integer intrinsics. Inside a kernel the frontend also
/// recognizes the corresponding inherent methods on i32, u32, i64 and u64.
pub trait Math {
    fn count_ones(self) -> u32;
    fn leading_zeros(self) -> u32;
    fn trailing_zeros(self) -> u32;
    fn rotate_left(self, n: u32) -> Self
    where
        Self: Sized;
}

macro_rules! integers {
    ($($ty:ty),*) => {$(
        impl Math for $ty {
            intrinsic!(fn count_ones(self) -> u32);
            intrinsic!(fn leading_zeros(self) -> u32);
            intrinsic!(fn trailing_zeros(self) -> u32);
            intrinsic!(fn rotate_left(self, n: u32) -> Self);
        }
    )*};
}
integers!(i32, u32, i64, u64);
