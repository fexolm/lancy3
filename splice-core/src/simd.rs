//! Element types and permutations used by portable SIMD instructions.
use crate::{BufferElement, KernelScalar};

/// An element type of a kernel vector (`Lane`): a number type whose
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
/// An integer element type of a kernel vector (`Lane`): the lanes have integer arithmetic
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

// The narrowest element (`u8`) has the most lanes; each needs a mask bit.
const _: () = assert!(splice_target::MAX_VECTOR_BYTES <= u64::BITS as usize);

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
