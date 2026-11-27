//! Native-width vectors: the kernel types `Lane` and `Mask`, declared as
//! intrinsics that every backend implements. Outside a kernel they only exist
//! as types; host code sizes data with [`lanes`].
//!
//! A kernel can import the vector type inside an individual Rust block.
//! The import is scoped to that block in both ordinary and staged code.
//!
//! ```compile_fail
//! #[splice_core::kernel(crate = splice_core)]
//! fn outside_scope(src: &[u16]) -> i32 {
//!     { use splice_core::simd::Lane; let _inside = Lane::load(src, 0); }
//!     Lane::<u16>::lanes() as i32
//! }
//! ```
use crate::{BufferElement, KernelScalar};
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Not, Shr, Sub, SubAssign,
};

/// Vector width of the default target of this machine (`SPLICE_SIMD`), for
/// host code that sizes data for kernels built for it ([`lanes`]).
/// Compiled kernels use the width of their `BuildOptions::target`.
///
/// # Panics
/// If `SPLICE_SIMD` names a profile this machine cannot run.
pub fn vector_bytes() -> usize {
    match splice_target::Target::host() {
        Ok(target) => target.vector_bytes(),
        Err(error) => panic!("invalid SPLICE_SIMD configuration: {error}"),
    }
}

/// An element type of [`Lane`]: a number type whose
/// [`ScalarType::is_lane`](crate::ScalarType::is_lane) holds.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a SIMD lane element",
    label = "not a lane element",
    note = "lane elements are `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `f32` and `f64`"
)]
pub trait LaneElement: BufferElement + Default + PartialOrd + std::fmt::Debug + 'static {
    /// The element of the lane indices of a vector of this element.
    type Index: LaneIndex;
    /// Zero of this element.
    const ZERO: Self;
    /// The bit pattern a kernel splats (`Lane::splat` of a known value).
    #[doc(hidden)]
    fn bits(self) -> u64;
}
/// An element type that indexes lanes (`Lane::gather`).
pub trait LaneIndex: LaneElement {}
/// An integer element type of [`Lane`]: the lanes have integer arithmetic
/// (`reduce_sum`, `simd_min`, `saturating_add`).
pub trait LaneInteger: LaneElement {}
macro_rules! element {
    ($ty:ty, $idx:ty, $bits:expr) => {
        const _: () = assert!(<$ty as KernelScalar>::TYPE.is_lane());
        impl LaneElement for $ty {
            type Index = $idx;
            const ZERO: Self = 0 as $ty;
            fn bits(self) -> u64 {
                ($bits)(self)
            }
        }
    };
}
macro_rules! integer {
    ($($ty:ty),*) => {$(
        element!($ty, $ty, |v| v as u64);
        impl LaneIndex for $ty {}
        impl LaneInteger for $ty {}
    )*};
}
integer!(i8, u8, i16, u16, i32, u32, i64);
element!(f32, i32, |v: f32| u64::from(v.to_bits()));
element!(f64, i64, f64::to_bits);

/// Lanes of `T` in a vector of this machine's default target
/// ([`vector_bytes`]): host code sizes the data of a kernel built for it
/// with this. A kernel asks `Lane::<T>::lanes()`.
pub fn lanes<T: LaneElement>() -> usize {
    T::TYPE.lanes(vector_bytes())
}

// The narrowest element (`u8`) has the most lanes; each needs a mask bit.
const _: () = assert!(splice_target::MAX_VECTOR_BYTES <= u64::BITS as usize);

// The SIMD types and their intrinsics, declared in `splice-macros`, which
// types a kernel's calls from the declarations (see `simd_api.rs` there).
crate::__simd_api!();

/// Fixed-width-independent permutations of two vectors concatenated end to end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shuffle {
    InterleaveLow,
    InterleaveHigh,
    DeinterleaveEven,
    DeinterleaveOdd,
}
impl Shuffle {
    pub fn index(self, i: usize, lanes: usize) -> usize {
        match self {
            Self::InterleaveLow => i / 2 + (i % 2) * lanes,
            Self::InterleaveHigh => lanes / 2 + i / 2 + (i % 2) * lanes,
            Self::DeinterleaveEven => 2 * i,
            Self::DeinterleaveOdd => 2 * i + 1,
        }
    }
}
/// The lane conversions `Lane::cast`, `widen_low`, `widen_high` and
/// `narrow` allow: `T` converts to `U` (`as`).
pub trait LaneCast<T: LaneElement>: LaneElement {}
impl<T: LaneElement> LaneCast<T> for T {}
macro_rules! lane_cast_pair {
    ($a:ty, $b:ty) => {
        impl LaneCast<$b> for $a {}
        impl LaneCast<$a> for $b {}
    };
}
lane_cast_pair!(i32, u32);
lane_cast_pair!(i32, f32);
lane_cast_pair!(u32, f32);
lane_cast_pair!(i64, f64);
lane_cast_pair!(u16, i32);
lane_cast_pair!(u16, u32);
lane_cast_pair!(i32, i64);
lane_cast_pair!(u32, i64);
lane_cast_pair!(f32, f64);
