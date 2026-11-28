//! Kernel operations provided by the Splice frontend.

/// Declare a kernel-only Rust function. Its signature is checked by rustc;
/// the kernel frontend translates registered operations into portable IR.
/// Calling the declaration directly from Rust panics.
#[macro_export]
#[doc(hidden)]
macro_rules! __splice_intrinsic {
    ($($signature:tt)*) => {
        #[allow(unused_variables)]
        $($signature)* { ::core::unreachable!("kernel intrinsic called outside a #[kernel]") }
    };
}

#[doc(inline)]
pub use crate::__splice_intrinsic as intrinsic;

// These markers share the import of the Rust item. They only resolve names;
// the proc macro owns all signatures, argument handling and IR construction.
macro_rules! builtin_marker {
    ($name:ident, [$($kind:ident),+], $d:tt) => {
        #[doc(hidden)]
        #[macro_export]
        macro_rules! $name {
            $((@intrinsic $kind, $d yes:expr, $d no:expr) => { $d yes };)+
            ($d($d rest:tt)*) => { $d crate::__private::__splice_choose_rust!($d($d rest)*) };
        }
    };
}
builtin_marker!(__splice_lane, [simd, lane], $);
builtin_marker!(__splice_mask, [simd], $);
builtin_marker!(__splice_math, [math], $);

pub mod math;
pub mod simd;

pub use math::Math;
pub use simd::{Lane, LaneCast, LaneElement, LaneIndex, LaneInteger, Mask, Shuffle};
