//! [`Math`]: integer bit operations that every backend implements as
//! intrinsics, with the signatures and results of Rust's integer methods.
use crate::hir::ScalarType;
use crate::intrinsic::IntrinsicId;

/// Bit operations of the kernel integer types. A kernel calls them as
/// methods (`x.count_ones()`) with the trait in scope; each call reaches
/// the backend as an intrinsic. Outside a kernel they panic: Rust code
/// calls the integer's own method.
#[crate::intrinsic]
pub trait Math: Sized {
    /// The number of one bits (Rust's `count_ones`).
    fn count_ones(self) -> u32;
    /// The number of leading zero bits (Rust's `leading_zeros`).
    fn leading_zeros(self) -> u32;
    /// The number of trailing zero bits (Rust's `trailing_zeros`).
    fn trailing_zeros(self) -> u32;
    /// Rotate left by `n` modulo the bit width (Rust's `rotate_left`).
    fn rotate_left(self, n: u32) -> Self;
}

macro_rules! math {
    ($($ty:ty),*) => {$(
        #[crate::intrinsic]
        impl Math for $ty {
            fn count_ones(self) -> u32;
            fn leading_zeros(self) -> u32;
            fn trailing_zeros(self) -> u32;
            fn rotate_left(self, n: u32) -> Self;
        }
    )*};
}
math!(i32, u32, i64, u64);

/// The intrinsics of every [`Math`] implementation.
pub(crate) const INTRINSICS: [&[IntrinsicId]; 4] = [
    <i32 as Math>::__SPLICE_INTRINSICS,
    <u32 as Math>::__SPLICE_INTRINSICS,
    <i64 as Math>::__SPLICE_INTRINSICS,
    <u64 as Math>::__SPLICE_INTRINSICS,
];

/// A [`Math`] operation, for backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathOp {
    CountOnes,
    LeadingZeros,
    TrailingZeros,
    /// `x.rotate_left(n)`: the operands are `x` and the `u32` count.
    RotateLeft,
}

/// The [`Math`] operation `id` is and the integer type it operates on.
pub fn decode(id: &IntrinsicId) -> Option<(MathOp, ScalarType)> {
    INTRINSICS
        .iter()
        .flat_map(|ids| ids.iter())
        .find(|d| *d == id)?;
    let op = match id.method {
        "count_ones" => MathOp::CountOnes,
        "leading_zeros" => MathOp::LeadingZeros,
        "trailing_zeros" => MathOp::TrailingZeros,
        "rotate_left" => MathOp::RotateLeft,
        _ => return None,
    };
    match id.self_type()? {
        crate::hir::ValueType::Scalar(ty) => Some((op, ty)),
        _ => None,
    }
}
