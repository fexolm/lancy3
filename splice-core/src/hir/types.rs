//! Scalar types, constants and the reference semantics of scalar operations.
use super::Binary;
use crate::Trap;
use std::fmt;

// `Index` is Rust's `usize`. Every supported target is 64-bit.
const _: () = assert!(usize::BITS == 64);

/// What the bits of a scalar mean.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    Bool,
    /// A Unicode scalar value: an unsigned 32-bit integer outside the
    /// surrogates and at most `0x10FFFF`.
    Char,
    Signed,
    Unsigned,
    Float,
}

/// The properties of a scalar type: one row per [`ScalarType`].
#[derive(Clone, Copy, Debug)]
pub struct ScalarInfo {
    /// The Rust name of the type.
    pub name: &'static str,
    /// Significant bits of a value: 1 for `bool`.
    pub bits: u32,
    pub kind: ScalarKind,
}

macro_rules! scalar_types {
    ($($variant:ident => $name:literal, $bits:literal, $kind:ident;)*) => {
        /// The type of a scalar register. Integers wrap or trap as the
        /// operation says; `Index` is Rust's `usize` (64 bits) and `Char`
        /// Rust's `char`. Every property comes from [`ScalarType::info`].
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum ScalarType {
            $($variant,)*
        }

        impl ScalarType {
            pub const ALL: [Self; [$(Self::$variant),*].len()] = [$(Self::$variant),*];
            const INFO: [ScalarInfo; Self::ALL.len()] = [$(ScalarInfo {
                name: $name,
                bits: $bits,
                kind: ScalarKind::$kind,
            }),*];
        }
    };
}

scalar_types! {
    Bool => "bool", 1, Bool;
    Char => "char", 32, Char;
    I8 => "i8", 8, Signed;
    U8 => "u8", 8, Unsigned;
    I16 => "i16", 16, Signed;
    U16 => "u16", 16, Unsigned;
    I32 => "i32", 32, Signed;
    U32 => "u32", 32, Unsigned;
    I64 => "i64", 64, Signed;
    U64 => "u64", 64, Unsigned;
    Index => "usize", 64, Unsigned;
    F32 => "f32", 32, Float;
    F64 => "f64", 64, Float;
}

impl ScalarType {
    /// The row of the type table that describes this type.
    pub const fn info(self) -> ScalarInfo {
        Self::INFO[self as usize]
    }

    pub const fn kind(self) -> ScalarKind {
        self.info().kind
    }

    /// Significant bits of a value: 1 for `Bool`.
    pub const fn bits(self) -> u32 {
        self.info().bits
    }

    /// Bytes of a value in memory: one for `Bool`.
    pub const fn bytes(self) -> usize {
        self.bits().div_ceil(8) as usize
    }

    pub const fn is_float(self) -> bool {
        matches!(self.kind(), ScalarKind::Float)
    }

    /// A signed or unsigned integer (`Char` and `Bool` are not).
    pub const fn is_integer(self) -> bool {
        matches!(self.kind(), ScalarKind::Signed | ScalarKind::Unsigned)
    }

    pub const fn is_signed(self) -> bool {
        matches!(self.kind(), ScalarKind::Signed)
    }

    /// An integer or float: the types arithmetic applies to.
    pub const fn is_number(self) -> bool {
        self.is_integer() || self.is_float()
    }

    /// Whether the type is the element of a [`Lane`](crate::simd::Lane): a
    /// number other than `u64` and `usize`.
    pub const fn is_lane(self) -> bool {
        self.is_number() && !matches!(self, Self::U64 | Self::Index)
    }

    /// The integer type of the same width and signedness a code generator
    /// stores this type as: `Char` is `U32`; other types are themselves.
    pub const fn representation(self) -> Self {
        match self {
            Self::Char => Self::U32,
            ty => ty,
        }
    }

    /// The integer type of the same width as a lane of this type: the
    /// type of its lane indices and masks. `F32` is `I32`, `F64` is `I64`.
    pub const fn index(self) -> Self {
        match self {
            Self::F32 => Self::I32,
            Self::F64 => Self::I64,
            ty => ty,
        }
    }

    /// Lanes of this type in a vector of `vector_bytes`.
    pub const fn lanes(self, vector_bytes: usize) -> usize {
        vector_bytes / self.bytes()
    }

    /// The Rust name of the type.
    pub const fn name(self) -> &'static str {
        self.info().name
    }

    /// The integer type of `bits` bits: `U64`, not `Index`, for 64
    /// unsigned bits.
    ///
    /// # Panics
    /// If no integer type has `bits` bits.
    pub const fn integer(bits: u32, signed: bool) -> Self {
        match (bits, signed) {
            (8, true) => Self::I8,
            (8, false) => Self::U8,
            (16, true) => Self::I16,
            (16, false) => Self::U16,
            (32, true) => Self::I32,
            (32, false) => Self::U32,
            (64, true) => Self::I64,
            (64, false) => Self::U64,
            _ => panic!("no integer type of this width"),
        }
    }

    /// The type Rust names `name`.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|ty| ty.name() == name)
    }

    /// Whether Rust's `as` converts a value of this type to `to`: numbers
    /// convert to numbers, `bool` and `char` to integers, and `u8` to
    /// `char`.
    pub const fn casts_to(self, to: Self) -> bool {
        match (self.kind(), to.kind()) {
            (ScalarKind::Bool, ScalarKind::Bool) | (ScalarKind::Char, ScalarKind::Char) => true,
            (ScalarKind::Bool | ScalarKind::Char, _) => to.is_integer(),
            (_, ScalarKind::Bool) => false,
            (_, ScalarKind::Char) => matches!(self, Self::U8),
            _ => true,
        }
    }

    /// The smallest and largest value of an integer type.
    const fn range(self) -> (i128, i128) {
        let bits = self.bits();
        if self.is_signed() {
            (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
        } else {
            (0, (1i128 << bits) - 1)
        }
    }

    const fn mask(self) -> u64 {
        u64::MAX >> (64 - self.bits())
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A typed scalar value. The bits of a value occupy the low `ty.bits()`
/// bits; the rest are zero (floats are their IEEE bit pattern).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Constant {
    bits: u64,
    ty: ScalarType,
}

impl Constant {
    /// The value of `ty` held in the low bits of `bits`.
    pub const fn new(ty: ScalarType, bits: u64) -> Self {
        Self {
            bits: bits & ty.mask(),
            ty,
        }
    }

    pub const fn index(value: usize) -> Self {
        Self::new(ScalarType::Index, value as u64)
    }

    pub const fn ty(self) -> ScalarType {
        self.ty
    }

    /// The value's bits, zero-extended to 64.
    pub const fn bits(self) -> u64 {
        self.bits
    }

    /// The value's bits, sign-extended to 64 for signed integers: the
    /// 64-bit word that holds the value in a register.
    pub const fn word(self) -> u64 {
        self.integer() as u64
    }

    /// An integer, `bool` or `char`, exactly.
    const fn integer(self) -> i128 {
        if self.ty.is_signed() {
            let unused = 64 - self.ty.bits();
            ((self.bits << unused) as i64 >> unused) as i128
        } else {
            self.bits as i128
        }
    }

    const fn from_integer(ty: ScalarType, value: i128) -> Self {
        Self::new(ty, value as u64)
    }

    fn float(self) -> f64 {
        match self.ty {
            ScalarType::F32 => f64::from(f32::from_bits(self.bits as u32)),
            _ => f64::from_bits(self.bits),
        }
    }

    /// The bit pattern of `value` rounded to `ty` (`F32` or `F64`).
    fn from_float(ty: ScalarType, value: f64) -> Self {
        match ty {
            ScalarType::F32 => Self::from(value as f32),
            _ => Self::from(value),
        }
    }

    /// Rust's `value as to`, or `None` if Rust has no such cast (see
    /// [`ScalarType::casts_to`]). Floats convert to integers with
    /// saturation and NaN becomes zero.
    pub fn cast(self, to: ScalarType) -> Option<Self> {
        if !self.ty.casts_to(to) {
            return None;
        }
        Some(match (self.ty.is_float(), to.is_float()) {
            (false, false) => Self::from_integer(to, self.integer()),
            (false, true) => match to {
                // Round once, from the exact integer.
                ScalarType::F32 => Self::from(self.integer() as f32),
                _ => Self::from(self.integer() as f64),
            },
            (true, true) if self.ty == to => self,
            (true, true) => Self::from_float(to, self.float()),
            (true, false) => {
                let value = self.float();
                // Rust saturates and maps NaN to zero.
                let (min, max) = to.range();
                let saturated = if value.is_nan() {
                    0
                } else {
                    (value as i128).clamp(min, max)
                };
                Self::from_integer(to, saturated)
            }
        })
    }
}

impl fmt::Debug for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Rust syntax with a type suffix, such as `-1i32`, `true` or `1.5f64`.
impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ty {
            ScalarType::Bool => write!(f, "{}", self.bits != 0),
            ScalarType::F32 => write!(f, "{:?}f32", f32::from_bits(self.bits as u32)),
            ScalarType::F64 => write!(f, "{:?}f64", f64::from_bits(self.bits)),
            ScalarType::Char => match char::from_u32(self.bits as u32) {
                Some(c) => write!(f, "{c:?}"),
                None => write!(f, "'\\u{{{:x}}}'", self.bits),
            },
            ty => write!(f, "{}{ty}", self.integer()),
        }
    }
}

macro_rules! constant_from {
    ($($ty:ty => $scalar:ident),*) => {$(
        impl From<$ty> for Constant {
            fn from(value: $ty) -> Self {
                Self::new(ScalarType::$scalar, value as u64)
            }
        }
    )*};
}
constant_from!(
    bool => Bool, char => Char, i8 => I8, u8 => U8, i16 => I16, u16 => U16, i32 => I32,
    u32 => U32, i64 => I64, u64 => U64, usize => Index
);

impl From<f32> for Constant {
    fn from(value: f32) -> Self {
        Self::new(ScalarType::F32, u64::from(value.to_bits()))
    }
}

impl From<f64> for Constant {
    fn from(value: f64) -> Self {
        Self::new(ScalarType::F64, value.to_bits())
    }
}

impl Binary {
    pub const fn is_comparison(self) -> bool {
        matches!(
            self,
            Self::Eq | Self::Ne | Self::Lt | Self::Le | Self::Gt | Self::Ge
        )
    }

    /// The result type of this operation on two operands of type `ty`, or
    /// `None` if it does not apply to them. Comparisons produce `Bool`.
    /// `Char` supports comparisons, `Min` and `Max`; `Bool` supports `Eq`, `Ne`, `And`, `Or` and `Xor`; floats support
    /// `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max` and comparisons; integers
    /// support every operation, including the `Saturating` forms.
    pub const fn output(self, ty: ScalarType) -> Option<ScalarType> {
        let applies = match ty.kind() {
            ScalarKind::Bool => {
                matches!(self, Self::Eq | Self::Ne | Self::And | Self::Or | Self::Xor)
            }
            ScalarKind::Char => self.is_comparison() || matches!(self, Self::Min | Self::Max),
            ScalarKind::Float => {
                self.is_comparison()
                    || matches!(
                        self,
                        Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Min | Self::Max
                    )
            }
            _ => true,
        };
        if !applies {
            None
        } else if self.is_comparison() {
            Some(ScalarType::Bool)
        } else {
            Some(ty)
        }
    }

    /// The reference semantics of scalar operations, which every code
    /// generator implements. Both operands have one type that the
    /// operation applies to (see [`output`](Self::output)).
    ///
    /// Plain `Add`, `Sub`, `Mul` and shifts wrap; a shift amount is taken
    /// modulo the width. The `Checked` forms trap on overflow, and on a
    /// shift amount outside `0..width`. The `Saturating` forms clamp the
    /// exact result to the type's range. Integer `Div` and `Rem` trap on a
    /// zero divisor and on signed `MIN / -1`. Float `Min` and `Max` are
    /// IEEE 754-2019 `minimumNumber` and `maximumNumber`: a NaN operand
    /// yields the other operand and -0 is below +0.
    ///
    /// # Panics
    /// If the operands have different types, or the operation does not
    /// apply to them.
    pub fn checked_eval(self, a: Constant, b: Constant) -> Result<Constant, Trap> {
        let ty = a.ty();
        let output = self
            .output(ty)
            .filter(|_| b.ty() == ty)
            .unwrap_or_else(|| panic!("{self:?} does not apply to {a:?} and {b:?}"));
        if ty.is_float() {
            return Ok(self.float(a, b, output));
        }
        let (x, y) = (a.integer(), b.integer());
        if self.is_comparison() {
            return Ok(Constant::from(compare(self, x, y)));
        }
        let bits = ty.bits();
        let (min, max) = ty.range();
        let checked = |value: Option<i128>| {
            value
                .filter(|v| (min..=max).contains(v))
                .map(|v| Constant::from_integer(ty, v))
                .ok_or(Trap::Overflow)
        };
        let divide = |op: fn(i128, i128) -> i128| {
            if y == 0 {
                Err(Trap::DivisionByZero)
            } else if ty.is_signed() && x == min && y == -1 {
                Err(Trap::DivisionOverflow)
            } else {
                Ok(Constant::from_integer(ty, op(x, y)))
            }
        };
        // The shift amount as an unsigned value of the operand type.
        let amount = b.bits();
        let wrap = |value: i128| Ok(Constant::from_integer(ty, value));
        match self {
            Self::Add => wrap(x + y),
            Self::Sub => wrap(x - y),
            Self::Mul => wrap(x.wrapping_mul(y)),
            Self::Div => divide(|x, y| x / y),
            Self::Rem => divide(|x, y| x % y),
            Self::And => wrap(x & y),
            Self::Or => wrap(x | y),
            Self::Xor => wrap(x ^ y),
            Self::Shl => wrap(x << (amount % u64::from(bits))),
            Self::Shr => wrap(x >> (amount % u64::from(bits))),
            Self::CheckedAdd => checked(Some(x + y)),
            Self::CheckedSub => checked(Some(x - y)),
            Self::CheckedMul => checked(x.checked_mul(y)),
            Self::CheckedShl | Self::CheckedShr if amount >= u64::from(bits) => Err(Trap::Overflow),
            Self::CheckedShl => wrap(x << amount),
            Self::CheckedShr => wrap(x >> amount),
            Self::SaturatingAdd => wrap((x + y).clamp(min, max)),
            Self::SaturatingSub => wrap((x - y).clamp(min, max)),
            Self::SaturatingMul => wrap(x.checked_mul(y).map_or_else(
                || if (x < 0) == (y < 0) { max } else { min },
                |v| v.clamp(min, max),
            )),
            Self::Min => wrap(x.min(y)),
            Self::Max => wrap(x.max(y)),
            Self::Eq | Self::Ne | Self::Lt | Self::Le | Self::Gt | Self::Ge => unreachable!(),
        }
    }

    fn float(self, a: Constant, b: Constant, output: ScalarType) -> Constant {
        let (x, y) = (a.float(), b.float());
        if output == ScalarType::Bool {
            return Constant::from(match self {
                Self::Eq => x == y,
                Self::Ne => x != y,
                Self::Lt => x < y,
                Self::Le => x <= y,
                Self::Gt => x > y,
                _ => x >= y,
            });
        }
        let value = match self {
            // Rounding an f64 result to f32 rounds the f32 operation
            // correctly: f64 has more than 2 * 24 + 2 bits of precision.
            Self::Add => x + y,
            Self::Sub => x - y,
            Self::Mul => x * y,
            Self::Div => x / y,
            Self::Min | Self::Max => {
                let min = self == Self::Min;
                return if b.float().is_nan() {
                    a
                } else if x.is_nan() {
                    b
                } else if x == y {
                    // Only the signs of equal zeros can differ.
                    if x.is_sign_negative() == min { a } else { b }
                } else if (x < y) == min {
                    a
                } else {
                    b
                };
            }
            _ => unreachable!("verified float operation"),
        };
        Constant::from_float(output, value)
    }
}

fn compare(op: Binary, x: i128, y: i128) -> bool {
    match op {
        Binary::Eq => x == y,
        Binary::Ne => x != y,
        Binary::Lt => x < y,
        Binary::Le => x <= y,
        Binary::Gt => x > y,
        _ => x >= y,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_follow_rust_at_every_width() {
        let i = |v: i32| Constant::from(v);
        assert_eq!(Binary::Add.checked_eval(i(i32::MAX), i(1)), Ok(i(i32::MIN)));
        assert_eq!(
            Binary::CheckedAdd.checked_eval(i(i32::MAX), i(1)),
            Err(Trap::Overflow)
        );
        let u = |v: u64| Constant::new(ScalarType::Index, v);
        assert_eq!(
            Binary::CheckedSub.checked_eval(u(0), u(1)),
            Err(Trap::Overflow)
        );
        assert_eq!(
            Binary::CheckedMul.checked_eval(u(1 << 31), u(2)),
            Ok(u(1 << 32))
        );
        assert_eq!(Binary::Lt.checked_eval(u(0), u(u64::MAX)), Ok(true.into()));
        assert_eq!(
            Binary::Shr.checked_eval(i(-8), i(33)),
            Ok(i(-4)),
            "the amount wraps and the shift is arithmetic"
        );
        assert_eq!(
            Binary::CheckedShl.checked_eval(i(1), i(-1)),
            Err(Trap::Overflow)
        );
        let l = |v: i64| Constant::from(v);
        assert_eq!(
            Binary::Div.checked_eval(l(i64::MIN), l(-1)),
            Err(Trap::DivisionOverflow)
        );
        assert_eq!(
            Binary::Rem.checked_eval(u(5), u(0)),
            Err(Trap::DivisionByZero)
        );
    }

    #[test]
    fn narrow_integers_follow_rust() {
        let b = |v: i8| Constant::from(v);
        let w = |v: u16| Constant::from(v);
        assert_eq!(Binary::Add.checked_eval(b(i8::MAX), b(1)), Ok(b(i8::MIN)));
        assert_eq!(
            Binary::CheckedMul.checked_eval(b(-128), b(-1)),
            Err(Trap::Overflow)
        );
        assert_eq!(
            Binary::Div.checked_eval(b(-128), b(-1)),
            Err(Trap::DivisionOverflow)
        );
        assert_eq!(Binary::Rem.checked_eval(b(-7), b(2)), Ok(b(-1)));
        assert_eq!(Binary::Shr.checked_eval(b(-128), b(9)), Ok(b(-64)));
        assert_eq!(
            Binary::CheckedShl.checked_eval(w(1), w(16)),
            Err(Trap::Overflow)
        );
        assert_eq!(Binary::Lt.checked_eval(b(-1), b(0)), Ok(true.into()));
        assert_eq!(Binary::SaturatingSub.checked_eval(w(1), w(2)), Ok(w(0)));
        assert_eq!(b(-1).word(), u64::MAX, "a signed word is sign-extended");
        assert_eq!(b(-1).bits(), 0xff);
        for ty in ScalarType::ALL {
            assert_eq!(ScalarType::named(ty.name()), Some(ty));
        }
    }

    #[test]
    fn chars_compare_and_cast_like_rust() {
        let c = Constant::from('é');
        assert_eq!(c.cast(ScalarType::U8), Some(0xe9u8.into()));
        assert_eq!(c.cast(ScalarType::I64), Some(0xe9i64.into()));
        assert_eq!(
            Constant::from(200u8).cast(ScalarType::Char),
            Some('È'.into())
        );
        assert_eq!(Constant::from(65u32).cast(ScalarType::Char), None);
        assert_eq!(c.cast(ScalarType::F32), None);
        assert_eq!(
            Binary::Lt.checked_eval('a'.into(), 'b'.into()),
            Ok(true.into())
        );
        assert_eq!(Binary::Add.output(ScalarType::Char), None);
        assert_eq!(c.to_string(), "'é'");
    }

    #[test]
    fn floats_follow_ieee_and_min_prefers_numbers() {
        let f = |v: f32| Constant::from(v);
        assert_eq!(Binary::Min.checked_eval(f(f32::NAN), f(1.0)), Ok(f(1.0)));
        assert_eq!(Binary::Max.checked_eval(f(-0.0), f(0.0)), Ok(f(0.0)));
        assert_eq!(Binary::Min.checked_eval(f(0.0), f(-0.0)), Ok(f(-0.0)));
        assert_eq!(
            Binary::Ne.checked_eval(f(f32::NAN), f(f32::NAN)),
            Ok(true.into())
        );
        assert_eq!(
            Binary::Div.checked_eval(f(1.0), f(0.0)),
            Ok(f(f32::INFINITY))
        );
        assert_eq!(Binary::Rem.output(ScalarType::F32), None);
        assert_eq!(Binary::Lt.output(ScalarType::Bool), None);
    }

    #[test]
    fn casts_follow_rust_as() {
        let c = |v: Constant, to| v.cast(to).unwrap();
        assert_eq!(
            c((-1i32).into(), ScalarType::Index),
            Constant::index(usize::MAX)
        );
        assert_eq!(c(Constant::index(1 << 32), ScalarType::I32), 0i32.into());
        assert_eq!(c(true.into(), ScalarType::I64), 1i64.into());
        assert_eq!(c(f64::NAN.into(), ScalarType::I32), 0i32.into());
        assert_eq!(c((-1.5f32).into(), ScalarType::U32), 0u32.into());
        assert_eq!(c(1e20f64.into(), ScalarType::I64), i64::MAX.into());
        assert_eq!(
            c(u64::MAX.into(), ScalarType::F32),
            (u64::MAX as f32).into()
        );
        assert_eq!(Constant::from(1i32).cast(ScalarType::Bool), None);
        assert_eq!(Constant::from(true).cast(ScalarType::F32), None);
        assert_eq!(c(300f32.into(), ScalarType::U8), 255u8.into());
        assert_eq!(c((-1e9f64).into(), ScalarType::I16), i16::MIN.into());
        assert_eq!(c((-1i8).into(), ScalarType::U16), u16::MAX.into());
        assert_eq!(c(200u8.into(), ScalarType::I8), (-56i8).into());
    }
}
