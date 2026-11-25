//! Numeric types and element-wise arithmetic shared by scalar and SIMD instructions.
macro_rules! numeric_types {
    ($($name:ident: $bytes:expr),+) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Lane { $($name),+ }
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Type { Ptr, $($name),+, Vector(Lane) }
        impl Lane {
            pub const fn bytes(self) -> usize {
                match self { $(Self::$name => $bytes),+ }
            }
            pub const fn scalar(self) -> Type {
                match self { $(Self::$name => Type::$name),+ }
            }
        }
        impl Type {
            pub fn lane(self) -> Option<Lane> {
                match self { Self::Ptr => None, Self::Vector(lane) => Some(lane), $(Self::$name => Some(Lane::$name)),+ }
            }
        }
    };
}

numeric_types!(I8:1, U8:1, I16:2, U16:2, I32:4, U32:4, I64:8, U64:8, I128:16, U128:16, Isize:POINTER_BYTES, Usize:POINTER_BYTES, F16:2, BF16:2, F32:4, F64:8);

impl Lane {
    pub fn float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }
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
pub enum Unary {
    Neg,
    Abs,
    Not,
    Popcnt,
    LeadingZeros,
    TrailingZeros,
    Sqrt,
    Floor,
    Ceil,
    Trunc,
    Round,
    Nearest,
    Fract,
    Recip,
    Exp,
    Exp2,
    Ln,
    Log2,
    Log10,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    IsNan,
    IsInfinite,
    IsFinite,
    IsNormal,
    IsSubnormal,
    SignBit,
    Signum,
    NextUp,
    NextDown,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    ExpM1,
    Ln1p,
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
pub enum Ternary {
    /// One correctly rounded floating-point multiply-add.
    MulAdd,
    /// Take each bit from the second or third operand according to the first.
    BitSelect,
    /// Select by a scalar condition or by the high bit of each vector mask lane.
    Select,
}
