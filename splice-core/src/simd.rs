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
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Not, Shr, Sub, SubAssign,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Element {
    U16,
    U32,
    I32,
    I64,
    F32,
    F64,
}

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

impl Element {
    pub const fn bytes(self) -> usize {
        match self {
            Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
    /// Lanes in a vector of `vector_bytes`.
    pub const fn lanes(self, vector_bytes: usize) -> usize {
        vector_bytes / self.bytes()
    }
    pub const fn index(self) -> Self {
        match self {
            Self::F32 => Self::I32,
            Self::F64 => Self::I64,
            other => other,
        }
    }
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
    pub const fn is_unsigned(self) -> bool {
        matches!(self, Self::U16 | Self::U32)
    }
    pub const fn name(self) -> &'static str {
        match self {
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

mod sealed {
    pub trait Sealed {}
}
/// An element type of [`Lane`].
pub trait LaneElement:
    sealed::Sealed + Copy + Default + PartialOrd + std::fmt::Debug + 'static
{
    /// The element of the lane indices of a vector of this element.
    type Index: LaneIndex;
    const ELEMENT: Element;
    /// Zero of this element.
    const ZERO: Self;
    /// The bit pattern a kernel splats (`Lane::splat` of a known value).
    #[doc(hidden)]
    fn bits(self) -> u64;
}
/// An element type that indexes lanes (`Lane::gather`).
pub trait LaneIndex: LaneElement {}
macro_rules! element {
    ($ty:ty, $kind:ident, $idx:ty, $bits:expr) => {
        impl sealed::Sealed for $ty {}
        impl LaneElement for $ty {
            type Index = $idx;
            const ELEMENT: Element = Element::$kind;
            const ZERO: Self = 0 as $ty;
            fn bits(self) -> u64 {
                ($bits)(self)
            }
        }
    };
}
macro_rules! integer {
    ($ty:ty,$kind:ident) => {
        element!($ty, $kind, $ty, |v| v as u64);
        impl LaneIndex for $ty {}
    };
}
integer!(u16, U16);
integer!(u32, U32);
integer!(i32, I32);
integer!(i64, I64);
element!(f32, F32, i32, |v: f32| u64::from(v.to_bits()));
element!(f64, F64, i64, f64::to_bits);

/// Lanes of `T` in a vector of this machine's default target
/// ([`vector_bytes`]): host code sizes the data of a kernel built for it
/// with this. A kernel asks `Lane::<T>::lanes()`.
pub fn lanes<T: LaneElement>() -> usize {
    T::ELEMENT.lanes(vector_bytes())
}

// The narrowest element (`u16`) has the most lanes; each needs a mask bit.
const _: () = assert!(splice_target::MAX_VECTOR_BYTES / Element::U16.bytes() <= u32::BITS as usize);

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
