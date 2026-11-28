//! Kernel operations provided by the Splice frontend.

// Declare a signature for a built-in operation registered by the frontend.
macro_rules! intrinsic {
    ($($signature:tt)*) => {
        #[allow(unused_variables)]
        $($signature)* { ::core::unreachable!("Splice intrinsic stub executed in Rust") }
    };
}

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
pub mod simd;

pub use simd::{Element, Lane, LaneCast, LaneElement, LaneIndex, Mask, Shuffle};
