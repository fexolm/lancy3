//! Native-width vectors. The ordinary Rust and staged versions use the same number of lanes.
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
use std::{
    marker::PhantomData,
    ops::{Add, BitAnd, BitOr, BitXor, Mul, Not, Sub},
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

pub fn vector_bytes() -> usize {
    splice_target::vector_bytes()
}

impl Element {
    pub const fn bytes(self) -> usize {
        match self {
            Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
    pub fn lanes(self) -> usize {
        vector_bytes() / self.bytes()
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
pub trait LaneElement:
    sealed::Sealed + Copy + Default + PartialOrd + std::fmt::Debug + 'static
{
    type Storage: Copy + AsRef<[Self]> + AsMut<[Self]>;
    type Index: LaneIndex;
    const ELEMENT: Element;
    const ZERO: Self;
    fn fill(v: Self) -> Self::Storage;
    fn number(n: usize) -> Self;
    fn bits(self) -> u64;
    fn sum(self, rhs: Self) -> Self;
    fn difference(self, rhs: Self) -> Self;
    fn product(self, rhs: Self) -> Self;
}
pub trait LaneIndex: LaneElement {
    fn offset(self) -> usize;
}
macro_rules! element {
    ($ty:ty, $kind:ident, $idx:ty, $bits:expr, $add:expr, $sub:expr, $mul:expr) => {
        impl sealed::Sealed for $ty {}
        impl LaneElement for $ty {
            type Storage = [$ty; 64 / size_of::<$ty>()];
            type Index = $idx;
            const ELEMENT: Element = Element::$kind;
            const ZERO: Self = 0 as $ty;
            fn fill(v: Self) -> Self::Storage {
                [v; 64 / size_of::<$ty>()]
            }
            fn number(n: usize) -> Self {
                n as Self
            }
            fn bits(self) -> u64 {
                ($bits)(self)
            }
            fn sum(self, r: Self) -> Self {
                ($add)(self, r)
            }
            fn difference(self, r: Self) -> Self {
                ($sub)(self, r)
            }
            fn product(self, r: Self) -> Self {
                ($mul)(self, r)
            }
        }
    };
}
macro_rules! integer {
    ($ty:ty,$kind:ident) => {
        element!(
            $ty,
            $kind,
            $ty,
            |v| v as u64,
            <$ty>::wrapping_add,
            <$ty>::wrapping_sub,
            <$ty>::wrapping_mul
        );
        impl LaneIndex for $ty {
            fn offset(self) -> usize {
                self as usize
            }
        }
    };
}
integer!(u16, U16);
integer!(u32, U32);
integer!(i32, I32);
integer!(i64, I64);
element!(
    f32,
    F32,
    i32,
    |v: f32| u64::from(v.to_bits()),
    Add::add,
    Sub::sub,
    Mul::mul
);
element!(f64, F64, i64, f64::to_bits, Add::add, Sub::sub, Mul::mul);

/// One native-width block of elements. Use lanes() to advance a slice cursor.
#[derive(Clone, Copy)]
pub struct Lane<T: LaneElement>(T::Storage);
/// One predicate per lane of the corresponding data vector.
#[derive(Clone, Copy)]
pub struct Mask<T: LaneElement>(u32, PhantomData<T>);
impl<T: LaneElement> Mask<T> {
    fn from_fn(f: impl Fn(usize) -> bool) -> Self {
        let mut bits = 0;
        for i in 0..Lane::<T>::lanes() {
            bits |= u32::from(f(i)) << i;
        }
        Self(bits, PhantomData)
    }
    fn active(self, i: usize) -> bool {
        self.0 & (1 << i) != 0
    }
    pub fn any(self) -> bool {
        self.0 != 0
    }
    pub fn all(self) -> bool {
        self.0 == u32::MAX >> (32 - Lane::<T>::lanes())
    }
    pub fn select<U: LaneElement<Index = T::Index>>(self, yes: Lane<U>, no: Lane<U>) -> Lane<U> {
        Lane::from_fn(|i| if self.active(i) { yes.at(i) } else { no.at(i) })
    }
}
impl<T: LaneElement> Not for Mask<T> {
    type Output = Self;
    fn not(self) -> Self {
        Self::from_fn(|i| !self.active(i))
    }
}
macro_rules! mask_binary {
    ($trait:ident,$op:ident) => {
        impl<T: LaneElement> $trait for Mask<T> {
            type Output = Self;
            fn $op(self, r: Self) -> Self {
                Self(self.0.$op(r.0), PhantomData)
            }
        }
    };
}
mask_binary!(BitAnd, bitand);
mask_binary!(BitOr, bitor);
mask_binary!(BitXor, bitxor);

impl<T: LaneElement> Lane<T> {
    pub fn lanes() -> usize {
        T::ELEMENT.lanes()
    }
    /// Activate the first min(count, lanes()) lanes. Useful for the last partial block.
    pub fn first_n(count: usize) -> Mask<T> {
        Mask::from_fn(|i| i < count)
    }
    pub fn splat(v: T) -> Self {
        Self(T::fill(v))
    }
    fn at(&self, i: usize) -> T {
        self.0.as_ref()[i]
    }
    fn from_fn(mut f: impl FnMut(usize) -> T) -> Self {
        let mut data = T::fill(T::default());
        for (i, v) in data.as_mut()[..Self::lanes()].iter_mut().enumerate() {
            *v = f(i);
        }
        Self(data)
    }
    pub fn iota() -> Self {
        Self::from_fn(T::number)
    }
    pub fn load(src: &[T], offset: usize) -> Self {
        Self::from_fn(|i| src[offset.checked_add(i).expect("SIMD index overflow")])
    }
    pub fn store(self, dst: &mut [T], offset: usize) {
        for i in 0..Self::lanes() {
            dst[offset.checked_add(i).expect("SIMD index overflow")] = self.at(i);
        }
    }
    pub fn load_select<M: LaneElement<Index = T::Index>>(
        src: &[T],
        offset: usize,
        mask: Mask<M>,
        fallback: Self,
    ) -> Self {
        Self::from_fn(|i| {
            if mask.active(i) {
                src[offset.checked_add(i).expect("SIMD index overflow")]
            } else {
                fallback.at(i)
            }
        })
    }
    pub fn store_select<M: LaneElement<Index = T::Index>>(
        self,
        dst: &mut [T],
        offset: usize,
        mask: Mask<M>,
    ) {
        for i in 0..Self::lanes() {
            if mask.active(i) {
                dst[offset.checked_add(i).expect("SIMD index overflow")] = self.at(i);
            }
        }
    }
    pub fn gather(src: &[T], indices: Lane<T::Index>) -> Self {
        Self::from_fn(|i| src[indices.at(i).offset()])
    }
    pub fn gather_select<M: LaneElement<Index = T::Index>>(
        src: &[T],
        mask: Mask<M>,
        indices: Lane<T::Index>,
        fallback: Self,
    ) -> Self {
        Self::from_fn(|i| {
            if mask.active(i) {
                src[indices.at(i).offset()]
            } else {
                fallback.at(i)
            }
        })
    }
    pub fn scatter(self, dst: &mut [T], indices: Lane<T::Index>) {
        for i in 0..Self::lanes() {
            dst[indices.at(i).offset()] = self.at(i);
        }
    }
    pub fn scatter_select<M: LaneElement<Index = T::Index>>(
        self,
        dst: &mut [T],
        mask: Mask<M>,
        indices: Lane<T::Index>,
    ) {
        for i in 0..Self::lanes() {
            if mask.active(i) {
                dst[indices.at(i).offset()] = self.at(i);
            }
        }
    }
    pub fn add_masked<M: LaneElement<Index = T::Index>>(
        self,
        mask: Mask<M>,
        rhs: Self,
        fallback: Self,
    ) -> Self {
        mask.select(self + rhs, fallback)
    }
}
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
impl Mask<i32> {
    /// Number of active lanes.
    pub fn count(self) -> i32 {
        self.0.count_ones() as i32
    }
}

impl std::ops::Shr<u32> for Lane<i32> {
    type Output = Self;
    fn shr(self, count: u32) -> Self {
        Self::from_fn(|i| self.at(i).wrapping_shr(count))
    }
}

impl std::ops::Shr<u32> for Lane<u32> {
    type Output = Self;
    fn shr(self, count: u32) -> Self {
        Self::from_fn(|i| self.at(i).wrapping_shr(count))
    }
}

impl Lane<u32> {
    pub fn rotate_left(self, count: u32) -> Self {
        Self::from_fn(|i| self.at(i).rotate_left(count))
    }
    pub fn simd_min(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.at(i).min(rhs.at(i)))
    }
    pub fn simd_max(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.at(i).max(rhs.at(i)))
    }
}

impl Lane<i32> {
    /// Sum all lanes with wrapping addition.
    pub fn reduce_sum(self) -> i32 {
        (0..Self::lanes()).fold(0i32, |sum, i| sum.wrapping_add(self.at(i)))
    }
    /// Rotate the bits of each lane left by the same count, modulo 32.
    pub fn rotate_left(self, count: u32) -> Self {
        Self::from_fn(|i| self.at(i).rotate_left(count))
    }

    /// Signed minimum of each pair of lanes.
    pub fn simd_min(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.at(i).min(rhs.at(i)))
    }

    /// Signed maximum of each pair of lanes.
    pub fn simd_max(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.at(i).max(rhs.at(i)))
    }

    /// Number of words before the next vector-aligned address. Does not access memory.
    pub fn alignment_prefix(src: &[i32], offset: usize) -> usize {
        let addr = src.as_ptr().wrapping_add(offset).addr();
        (addr.wrapping_neg() & (vector_bytes() - 1)) / size_of::<i32>()
    }
    /// Pack selected lanes, preserving order and zeroing the remaining lanes.
    pub fn compress(self, mask: Mask<i32>) -> Self {
        let mut bits = mask.0;
        Self::from_fn(|_| {
            if bits == 0 {
                return 0;
            }
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            self.at(i)
        })
    }
    /// Fill selected positions from consecutive low lanes; keep the other
    /// positions from `fallback`.
    pub fn expand(self, mask: Mask<i32>, fallback: Self) -> Self {
        let mut src = 0;
        Self::from_fn(|i| {
            if mask.active(i) {
                let word = self.at(src);
                src += 1;
                word
            } else {
                fallback.at(i)
            }
        })
    }
    /// Look up each lane by its index; out-of-range indices produce zero.
    pub fn swizzle(self, indices: Self) -> Self {
        Self::from_fn(|i| {
            let idx = indices.at(i) as usize;
            if idx < Self::lanes() { self.at(idx) } else { 0 }
        })
    }
    /// Write enabled lanes contiguously, in lane order. An empty mask touches no memory.
    pub fn compress_store(self, dst: &mut [i32], mut offset: usize, mask: Mask<i32>) {
        for i in 0..Self::lanes() {
            if mask.active(i) {
                dst[offset] = self.at(i);
                offset += 1;
            }
        }
    }
    fn shuffle(self, other: Self, order: Shuffle) -> Self {
        let n = Self::lanes();
        Self::from_fn(|i| {
            let j = order.index(i, n);
            if j < n { self.at(j) } else { other.at(j - n) }
        })
    }
    pub fn interleave_low(self, rhs: Self) -> Self {
        self.shuffle(rhs, Shuffle::InterleaveLow)
    }
    pub fn interleave_high(self, rhs: Self) -> Self {
        self.shuffle(rhs, Shuffle::InterleaveHigh)
    }
    pub fn deinterleave_even(self, rhs: Self) -> Self {
        self.shuffle(rhs, Shuffle::DeinterleaveEven)
    }
    pub fn deinterleave_odd(self, rhs: Self) -> Self {
        self.shuffle(rhs, Shuffle::DeinterleaveOdd)
    }
    /// Wrapping sum of lanes zero through i in output lane i.
    pub fn prefix_sum(self) -> Self {
        let mut sum = 0i32;
        Self::from_fn(|i| {
            sum = sum.wrapping_add(self.at(i));
            sum
        })
    }
    /// Read a lane, wrapping the index modulo the active lane count.
    pub fn extract_wrapping(self, index: i32) -> i32 {
        self.at(index as usize & (Self::lanes() - 1))
    }
    /// Replace a lane, wrapping the index modulo the active lane count.
    pub fn replace_wrapping(self, index: i32, value: i32) -> Self {
        let index = index as usize & (Self::lanes() - 1);
        Self::from_fn(|i| if i == index { value } else { self.at(i) })
    }
    pub fn last(self) -> i32 {
        self.at(Self::lanes() - 1)
    }
}

macro_rules! compare {
    ($name:ident, $op:tt) => {
        impl<T: LaneElement> Lane<T> {
            pub fn $name(self, rhs: Self) -> Mask<T> {
                Mask::from_fn(|i| self.at(i) $op rhs.at(i))
            }
        }
    };
}
compare!(simd_eq, ==);
compare!(simd_ne, !=);
compare!(simd_lt, <);
compare!(simd_le, <=);
compare!(simd_gt, >);
compare!(simd_ge, >=);
macro_rules! arithmetic {
    ($trait:ident,$op:ident,$f:ident) => {
        impl<T: LaneElement> $trait for Lane<T> {
            type Output = Self;
            fn $op(self, r: Self) -> Self {
                Self::from_fn(|i| self.at(i).$f(r.at(i)))
            }
        }
    };
}
arithmetic!(Add, add, sum);
arithmetic!(Sub, sub, difference);
arithmetic!(Mul, mul, product);
macro_rules! bitwise {
    ($trait:ident,$op:ident) => {
        impl<T: LaneElement + $trait<Output = T>> $trait for Lane<T> {
            type Output = Self;
            fn $op(self, r: Self) -> Self {
                Self::from_fn(|i| self.at(i).$op(r.at(i)))
            }
        }
    };
}
bitwise!(BitAnd, bitand);
bitwise!(BitOr, bitor);
bitwise!(BitXor, bitxor);
impl<T: LaneElement + Not<Output = T>> Not for Lane<T> {
    type Output = Self;
    fn not(self) -> Self {
        Self::from_fn(|i| !self.at(i))
    }
}

macro_rules! assign {
    ($trait:ident,$method:ident,$op:tt,$bound:ident) => {
        impl<T:LaneElement> std::ops::$trait for Lane<T> where Lane<T>:$bound<Output=Lane<T>> {
            fn $method(&mut self,rhs:Self){ *self=*self $op rhs; }
        }
    }
}
assign!(AddAssign,add_assign,+,Add);
assign!(SubAssign,sub_assign,-,Sub);
assign!(MulAssign,mul_assign,*,Mul);
assign!(BitAndAssign,bitand_assign,&,BitAnd);
assign!(BitOrAssign,bitor_assign,|,BitOr);
assign!(BitXorAssign,bitxor_assign,^,BitXor);

impl<T: LaneElement + std::ops::Div<Output = T>> Lane<T> {
    fn quotient(self, rhs: Self) -> Self {
        Self::from_fn(|i| self.at(i) / rhs.at(i))
    }
}
macro_rules! float_div {
    ($ty:ty) => {
        impl std::ops::Div for Lane<$ty> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                self.quotient(rhs)
            }
        }
        impl std::ops::DivAssign for Lane<$ty> {
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }
    };
}
float_div!(f32);
float_div!(f64);

macro_rules! float_math {
    ($ty:ty) => {
        impl Lane<$ty> {
            pub fn sqrt(self) -> Self {
                Self::from_fn(|i| self.at(i).sqrt())
            }
            /// Multiply and add with one rounding per lane.
            pub fn mul_add(self, multiplier: Self, addend: Self) -> Self {
                Self::from_fn(|i| self.at(i).mul_add(multiplier.at(i), addend.at(i)))
            }
        }
    };
}
float_math!(f32);
float_math!(f64);

/// Numeric conversion of one lane, using the corresponding Rust cast.
pub trait LaneCast<T: LaneElement>: LaneElement {
    #[doc(hidden)]
    fn cast_lane(self) -> T;
}

impl<T: LaneElement> LaneCast<T> for T {
    fn cast_lane(self) -> T {
        self
    }
}

macro_rules! lane_cast_pair {
    ($a:ty, $b:ty) => {
        impl LaneCast<$b> for $a {
            fn cast_lane(self) -> $b {
                self as $b
            }
        }
        impl LaneCast<$a> for $b {
            fn cast_lane(self) -> $a {
                self as $a
            }
        }
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

impl<T: LaneElement> Lane<T> {
    /// Convert lanes without changing their width.
    pub fn cast<U: LaneElement>(self) -> Lane<U>
    where
        T: LaneCast<U>,
    {
        assert_eq!(size_of::<T>(), size_of::<U>());
        Lane::from_fn(|i| self.at(i).cast_lane())
    }
    /// Convert the lower half into lanes twice as wide.
    pub fn widen_low<U: LaneElement>(self) -> Lane<U>
    where
        T: LaneCast<U>,
    {
        assert_eq!(size_of::<T>() * 2, size_of::<U>());
        Lane::from_fn(|i| self.at(i).cast_lane())
    }
    /// Convert the upper half into lanes twice as wide.
    pub fn widen_high<U: LaneElement>(self) -> Lane<U>
    where
        T: LaneCast<U>,
    {
        assert_eq!(size_of::<T>() * 2, size_of::<U>());
        Lane::from_fn(|i| self.at(i + Lane::<U>::lanes()).cast_lane())
    }
    /// Convert and concatenate the two vectors into lanes half as wide.
    pub fn narrow<U: LaneElement>(self, high: Self) -> Lane<U>
    where
        T: LaneCast<U>,
    {
        assert_eq!(size_of::<T>(), size_of::<U>() * 2);
        let n = Self::lanes();
        Lane::from_fn(|i| {
            if i < n {
                self.at(i).cast_lane()
            } else {
                high.at(i - n).cast_lane()
            }
        })
    }
}
