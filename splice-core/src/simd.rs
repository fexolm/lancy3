//! Element types and permutations used by portable SIMD instructions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Element {
    U16,
    U32,
    I32,
    I64,
    F32,
    F64,
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
/// An element type of a kernel vector.
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

// The narrowest element (`u16`) has the most lanes; each needs a mask bit.
const _: () = assert!(splice_target::MAX_VECTOR_BYTES / Element::U16.bytes() <= u32::BITS as usize);

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
