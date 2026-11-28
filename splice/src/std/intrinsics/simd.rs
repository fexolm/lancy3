//! Native-width vectors: the kernel types `Lane` and `Mask`, declared as
//! intrinsics that every backend implements. Outside a kernel they only exist
//! as types; host code sizes data using the build [`crate::Target`].
//!
//! A kernel can import the vector type inside an individual Rust block.
//! The import is scoped to that block in both ordinary and staged code.
//!
//! ```compile_fail
//! #[splice::kernel]
//! fn outside_scope(src: &[u16]) -> i32 {
//!     { use splice::std::intrinsics::simd::Lane; let _inside = Lane::load(src, 0); }
//!     Lane::<u16>::element_width() as i32
//! }
//! ```
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Not, Shr, Sub, SubAssign,
};

use core::marker::PhantomData;
pub use splice_core::simd::{Element, LaneElement, LaneIndex, Shuffle};

// The companion marker follows use aliases in the macro namespace.
#[doc(hidden)]
pub use __splice_lane as Lane;
#[doc(hidden)]
pub use __splice_mask as Mask;

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

/// One native-width block of elements, a kernel value: the build target
/// decides the number of lanes. Use `element_width()` to advance a slice cursor.
#[derive(Clone, Copy)]
pub struct Lane<T: LaneElement>(PhantomData<T>);
/// One predicate per lane of the corresponding data vector.
#[derive(Clone, Copy)]
pub struct Mask<T: LaneElement>(PhantomData<T>);

impl<T: LaneElement> Mask<T> {
    intrinsic! {
        /// Whether any lane is active.
        pub fn any(self) -> bool
    }
    intrinsic! {
        /// Whether every lane is active.
        pub fn all(self) -> bool
    }
    intrinsic! {
        /// `yes` in the active lanes, `no` in the others.
        pub fn select<U: LaneElement<Index = T::Index>>(self, yes: Lane<U>, no: Lane<U>) -> Lane<U>
    }
}
impl Mask<i32> {
    intrinsic! {
        /// Number of active lanes.
        pub fn count(self) -> i32
    }
}
impl<T: LaneElement> Not for Mask<T> {
    type Output = Self;
    intrinsic! {
        fn not(self) -> Self
    }
}
impl<T: LaneElement> BitAnd for Mask<T> {
    type Output = Self;
    intrinsic! {
        fn bitand(self, rhs: Self) -> Self
    }
}
impl<T: LaneElement> BitOr for Mask<T> {
    type Output = Self;
    intrinsic! {
        fn bitor(self, rhs: Self) -> Self
    }
}
impl<T: LaneElement> BitXor for Mask<T> {
    type Output = Self;
    intrinsic! {
        fn bitxor(self, rhs: Self) -> Self
    }
}

impl<T: LaneElement> Lane<T> {
    intrinsic! {
        /// Lanes of the build target's vector.
        pub fn element_width() -> usize
    }
    intrinsic! {
        /// Bytes occupied by a vector on the kernel target.
        pub fn byte_width() -> usize
    }
    intrinsic! {
        /// Activate the first min(count, element_width()) lanes. Useful for the last partial block.
        pub fn first_n(count: usize) -> Mask<T>
    }
    intrinsic! {
        /// `v` in every lane: a known value, or a runtime `i32`.
        pub fn splat(v: T) -> Self
    }
    intrinsic! {
        /// Lane `i` holds `i`.
        pub fn iota() -> Self
    }
    intrinsic! {
        /// `src[offset..offset + element_width()]`; out of bounds traps.
        pub fn load(src: &[T], offset: usize) -> Self
    }
    intrinsic! {
        /// Store the lanes at `dst[offset..]`; out of bounds traps.
        pub fn store(self, dst: &mut [T], offset: usize)
    }
    intrinsic! {
        /// The active lanes from `src[offset + i]`, `fallback` in the others;
        /// only active lanes are bounds-checked.
        pub fn load_select<M: LaneElement<Index = T::Index>>(
            src: &[T],
            offset: usize,
            mask: Mask<M>,
            fallback: Self,
        ) -> Self
    }
    intrinsic! {
        /// Store the active lanes at `dst[offset + i]`.
        pub fn store_select<M: LaneElement<Index = T::Index>>(
            self,
            dst: &mut [T],
            offset: usize,
            mask: Mask<M>,
        )
    }
    intrinsic! {
        /// Lane `i` from `src[indices[i]]`.
        pub fn gather(src: &[T], indices: Lane<T::Index>) -> Self
    }
    intrinsic! {
        /// The active lanes from `src[indices[i]]`, `fallback` in the others.
        pub fn gather_select<M: LaneElement<Index = T::Index>>(
            src: &[T],
            mask: Mask<M>,
            indices: Lane<T::Index>,
            fallback: Self,
        ) -> Self
    }
    intrinsic! {
        /// Write lane `i` to `dst[indices[i]]` in lane order.
        pub fn scatter(self, dst: &mut [T], indices: Lane<T::Index>)
    }
    intrinsic! {
        /// Write the active lanes to `dst[indices[i]]` in lane order.
        pub fn scatter_select<M: LaneElement<Index = T::Index>>(
            self,
            dst: &mut [T],
            mask: Mask<M>,
            indices: Lane<T::Index>,
        )
    }
    intrinsic! {
        /// `self + rhs` in the active lanes, `fallback` in the others.
        pub fn add_masked<M: LaneElement<Index = T::Index>>(
            self,
            mask: Mask<M>,
            rhs: Self,
            fallback: Self,
        ) -> Self
    }
    intrinsic! {
        pub fn simd_eq(self, rhs: Self) -> Mask<T>
    }
    intrinsic! {
        pub fn simd_ne(self, rhs: Self) -> Mask<T>
    }
    intrinsic! {
        pub fn simd_lt(self, rhs: Self) -> Mask<T>
    }
    intrinsic! {
        pub fn simd_le(self, rhs: Self) -> Mask<T>
    }
    intrinsic! {
        pub fn simd_gt(self, rhs: Self) -> Mask<T>
    }
    intrinsic! {
        pub fn simd_ge(self, rhs: Self) -> Mask<T>
    }
    intrinsic! {
        /// Convert each lane to `U` of the same size (`as`).
        pub fn cast<U: LaneElement>(self) -> Lane<U>
        where
            T: LaneCast<U>
    }
    intrinsic! {
        /// Convert the low half of the lanes to `U` of twice the size.
        pub fn widen_low<U: LaneElement>(self) -> Lane<U>
        where
            T: LaneCast<U>
    }
    intrinsic! {
        /// Convert the high half of the lanes to `U` of twice the size.
        pub fn widen_high<U: LaneElement>(self) -> Lane<U>
        where
            T: LaneCast<U>
    }
    intrinsic! {
        /// Convert the lanes of `self`, then of `high`, to `U` of half the size.
        pub fn narrow<U: LaneElement>(self, high: Self) -> Lane<U>
        where
            T: LaneCast<U>
    }
}

impl Lane<u32> {
    intrinsic! {
        /// Rotate each lane left by a known count modulo 32.
        pub fn rotate_left(self, count: u32) -> Self
    }
    intrinsic! {
        pub fn simd_min(self, rhs: Self) -> Self
    }
    intrinsic! {
        pub fn simd_max(self, rhs: Self) -> Self
    }
}

impl Lane<i32> {
    intrinsic! {
        /// The wrapping sum of the lanes.
        pub fn reduce_sum(self) -> i32
    }
    intrinsic! {
        /// Rotate each lane's bits left by a known count modulo 32.
        pub fn rotate_left(self, count: u32) -> Self
    }
    intrinsic! {
        pub fn simd_min(self, rhs: Self) -> Self
    }
    intrinsic! {
        pub fn simd_max(self, rhs: Self) -> Self
    }
    intrinsic! {
        /// Elements from `offset` until `src[offset..]` is vector-aligned.
        pub fn alignment_prefix(src: &[i32], offset: usize) -> usize
    }
    intrinsic! {
        /// The active lanes in order, zero in the remaining lanes.
        pub fn compress(self, mask: Mask<i32>) -> Self
    }
    intrinsic! {
        /// The lanes in order into the active lanes, `fallback` in the others.
        pub fn expand(self, mask: Mask<i32>, fallback: Self) -> Self
    }
    intrinsic! {
        /// Lane `i` from lane `indices[i]`, zero for an out-of-range index.
        pub fn swizzle(self, indices: Self) -> Self
    }
    intrinsic! {
        /// Store the active lanes in order at `dst[offset..]`: the stores that
        /// fit happen, then an out-of-bounds lane traps.
        pub fn compress_store(self, dst: &mut [i32], offset: usize, mask: Mask<i32>)
    }
    intrinsic! {
        pub fn interleave_low(self, rhs: Self) -> Self
    }
    intrinsic! {
        pub fn interleave_high(self, rhs: Self) -> Self
    }
    intrinsic! {
        pub fn deinterleave_even(self, rhs: Self) -> Self
    }
    intrinsic! {
        pub fn deinterleave_odd(self, rhs: Self) -> Self
    }
    intrinsic! {
        /// Inclusive wrapping prefix sums of the lanes.
        pub fn prefix_sum(self) -> Self
    }
    intrinsic! {
        /// Lane `index` modulo `element_width()`.
        pub fn extract_wrapping(self, index: i32) -> i32
    }
    intrinsic! {
        /// `self` with lane `index` modulo `element_width()` set to `value`.
        pub fn replace_wrapping(self, index: i32, value: i32) -> Self
    }
    intrinsic! {
        /// The last lane.
        pub fn last(self) -> i32
    }
}

impl Lane<f32> {
    intrinsic! {
        pub fn sqrt(self) -> Self
    }
    intrinsic! {
        /// `self * multiplier + addend`, rounded once.
        pub fn mul_add(self, multiplier: Self, addend: Self) -> Self
    }
}
impl Lane<f64> {
    intrinsic! {
        pub fn sqrt(self) -> Self
    }
    intrinsic! {
        /// `self * multiplier + addend`, rounded once.
        pub fn mul_add(self, multiplier: Self, addend: Self) -> Self
    }
}

impl<T: LaneElement> Add for Lane<T> {
    type Output = Self;
    intrinsic! {
        fn add(self, rhs: Self) -> Self
    }
}
impl<T: LaneElement> Sub for Lane<T> {
    type Output = Self;
    intrinsic! {
        fn sub(self, rhs: Self) -> Self
    }
}
impl<T: LaneElement> Mul for Lane<T> {
    type Output = Self;
    intrinsic! {
        fn mul(self, rhs: Self) -> Self
    }
}
impl Div for Lane<f32> {
    type Output = Self;
    intrinsic! {
        fn div(self, rhs: Self) -> Self
    }
}
impl Div for Lane<f64> {
    type Output = Self;
    intrinsic! {
        fn div(self, rhs: Self) -> Self
    }
}
impl<T: LaneElement + BitAnd<Output = T>> BitAnd for Lane<T> {
    type Output = Self;
    intrinsic! {
        fn bitand(self, rhs: Self) -> Self
    }
}
impl<T: LaneElement + BitOr<Output = T>> BitOr for Lane<T> {
    type Output = Self;
    intrinsic! {
        fn bitor(self, rhs: Self) -> Self
    }
}
impl<T: LaneElement + BitXor<Output = T>> BitXor for Lane<T> {
    type Output = Self;
    intrinsic! {
        fn bitxor(self, rhs: Self) -> Self
    }
}
impl<T: LaneElement + Not<Output = T>> Not for Lane<T> {
    type Output = Self;
    intrinsic! {
        fn not(self) -> Self
    }
}
impl Shr<u32> for Lane<i32> {
    type Output = Self;
    intrinsic! {
        /// Arithmetic shift by a known count modulo 32.
        fn shr(self, count: u32) -> Self
    }
}
impl Shr<u32> for Lane<u32> {
    type Output = Self;
    intrinsic! {
        /// Logical shift by a known count modulo 32.
        fn shr(self, count: u32) -> Self
    }
}

// `x op= y` is `x = x op y`: a kernel stages the operator's intrinsic.
impl<T: LaneElement> AddAssign for Lane<T> {
    intrinsic! {
        fn add_assign(&mut self, rhs: Self)
    }
}
impl<T: LaneElement> SubAssign for Lane<T> {
    intrinsic! {
        fn sub_assign(&mut self, rhs: Self)
    }
}
impl<T: LaneElement> MulAssign for Lane<T> {
    intrinsic! {
        fn mul_assign(&mut self, rhs: Self)
    }
}
impl DivAssign for Lane<f32> {
    intrinsic! {
        fn div_assign(&mut self, rhs: Self)
    }
}
impl DivAssign for Lane<f64> {
    intrinsic! {
        fn div_assign(&mut self, rhs: Self)
    }
}
impl<T: LaneElement + BitAnd<Output = T>> BitAndAssign for Lane<T> {
    intrinsic! {
        fn bitand_assign(&mut self, rhs: Self)
    }
}
impl<T: LaneElement + BitOr<Output = T>> BitOrAssign for Lane<T> {
    intrinsic! {
        fn bitor_assign(&mut self, rhs: Self)
    }
}
impl<T: LaneElement + BitXor<Output = T>> BitXorAssign for Lane<T> {
    intrinsic! {
        fn bitxor_assign(&mut self, rhs: Self)
    }
}
