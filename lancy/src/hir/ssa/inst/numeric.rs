//! Numeric types and element-wise arithmetic shared by scalar and SIMD instructions.
macro_rules! numeric_types {
    ($($name:ident: $bytes:expr),+) => {
        /// The element type of a vector, or a scalar's numeric type.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Lane { $(#[doc = concat!("`", stringify!($name), "` elements.")] $name),+ }
        /// The type of an SSA value.
        ///
        /// A comparison or classification yields a mask: `I32` (zero or all
        /// ones) for a scalar, and for a vector the integer lanes of the
        /// operand's width.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Type {
            /// A machine address.
            Ptr,
            $(#[doc = concat!("A scalar `", stringify!($name), "`.")] $name,)+
            /// A vector of the target's width with lanes of the type.
            Vector(Lane),
        }
        impl Lane {
            /// Bytes of one element.
            pub const fn bytes(self) -> usize {
                match self { $(Self::$name => $bytes),+ }
            }
            /// The scalar type of one element.
            pub const fn scalar(self) -> Type {
                match self { $(Self::$name => Type::$name),+ }
            }
        }
        impl Type {
            /// The element type of a number or vector; `None` for `Ptr`.
            pub fn lane(self) -> Option<Lane> {
                match self { Self::Ptr => None, Self::Vector(lane) => Some(lane), $(Self::$name => Some(Lane::$name)),+ }
            }
        }
    };
}

numeric_types!(I8:1, U8:1, I16:2, U16:2, I32:4, U32:4, I64:8, U64:8, I128:16, U128:16, Isize:POINTER_BYTES, Usize:POINTER_BYTES, F16:2, BF16:2, F32:4, F64:8);

impl Lane {
    /// Whether the elements are floating point.
    pub fn float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }
    /// Whether the elements are signed integers.
    pub fn signed(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128 | Self::Isize
        )
    }
}

/// Bytes of `Ptr`, `Isize` and `Usize`: the JIT emits only 64-bit code for
/// the host, so the host's pointer width is the target's.
pub(crate) const POINTER_BYTES: usize = size_of::<usize>();
const _: () = assert!(POINTER_BYTES == 8, "lancy generates 64-bit code only");

impl Type {
    pub(crate) fn bytes(self, vector_bytes: usize) -> usize {
        match self {
            Self::Ptr => POINTER_BYTES,
            Self::Vector(_) => vector_bytes,
            _ => self.lane().unwrap().bytes(),
        }
    }
    pub(crate) fn mask_type(self) -> Self {
        match self {
            Self::Vector(Lane::F16 | Lane::BF16) => Self::Vector(Lane::I16),
            Self::Vector(Lane::F32) => Self::Vector(Lane::I32),
            Self::Vector(Lane::F64) => Self::Vector(Lane::I64),
            Self::Vector(_) => self,
            _ => Self::I32,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// An element-wise unary operation; the floating-point ones follow the Rust
/// methods of the same name.
pub enum Unary {
    /// Negation, wrapping for integers.
    Neg,
    /// Absolute value, wrapping for the integer minimum.
    Abs,
    /// Bitwise not.
    Not,
    /// Number of set bits.
    Popcnt,
    /// Number of leading zero bits.
    LeadingZeros,
    /// Number of trailing zero bits.
    TrailingZeros,
    /// Square root.
    Sqrt,
    /// Round toward negative infinity.
    Floor,
    /// Round toward positive infinity.
    Ceil,
    /// Round toward zero.
    Trunc,
    /// Round half away from zero.
    Round,
    /// Round half to even.
    Nearest,
    /// `x - trunc(x)`.
    Fract,
    /// `1 / x`.
    Recip,
    /// `e^x`.
    Exp,
    /// `2^x`.
    Exp2,
    /// Natural logarithm.
    Ln,
    /// Base-2 logarithm.
    Log2,
    /// Base-10 logarithm.
    Log10,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Tangent.
    Tan,
    /// Arcsine.
    Asin,
    /// Arccosine.
    Acos,
    /// Arctangent.
    Atan,
    /// Mask of NaN.
    IsNan,
    /// Mask of infinity.
    IsInfinite,
    /// Mask of neither NaN nor infinity.
    IsFinite,
    /// Mask of normal numbers.
    IsNormal,
    /// Mask of subnormal numbers.
    IsSubnormal,
    /// Mask of a set sign bit.
    SignBit,
    /// `1` with the sign of `x`; NaN stays NaN.
    Signum,
    /// The least number above `x`.
    NextUp,
    /// The greatest number below `x`.
    NextDown,
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Hyperbolic tangent.
    Tanh,
    /// Inverse hyperbolic sine.
    Asinh,
    /// Inverse hyperbolic cosine.
    Acosh,
    /// Inverse hyperbolic tangent.
    Atanh,
    /// `e^x - 1`, accurate near zero.
    ExpM1,
    /// `ln(1 + x)`, accurate near zero.
    Ln1p,
    /// Cube root.
    Cbrt,
}

impl Unary {
    pub(crate) fn result_type(self, ty: Type) -> Type {
        if matches!(
            self,
            Self::IsNan
                | Self::IsInfinite
                | Self::IsFinite
                | Self::IsNormal
                | Self::IsSubnormal
                | Self::SignBit
        ) {
            ty.mask_type()
        } else {
            ty
        }
    }
    pub(crate) fn supports(self, ty: Type) -> bool {
        ty.lane().is_some_and(|lane| match self {
            Self::Not | Self::Neg | Self::Abs => true,
            Self::LeadingZeros | Self::TrailingZeros | Self::Popcnt => !lane.float(),
            _ => lane.float(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// An element-wise three-operand operation.
pub enum Ternary {
    /// One correctly rounded floating-point multiply-add.
    MulAdd,
    /// Take each bit from the second or third operand according to the first.
    BitSelect,
    /// Select by a scalar condition or by the high bit of each vector mask lane.
    Select,
}
