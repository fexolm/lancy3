//! Scalar types, constants and the reference semantics of scalar operations.
use super::Binary;
use crate::Trap;
use crate::simd::Element;
use std::fmt;

// `Index` is Rust's `usize`. Every supported target is 64-bit.
const _: () = assert!(usize::BITS == 64);

/// The type of a scalar register. Integers wrap or trap as the operation
/// says; `Index` is Rust's `usize` (64 bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Bool,
    I32,
    U32,
    I64,
    U64,
    Index,
    F32,
    F64,
}

impl ScalarType {
    pub const ALL: [Self; 8] = [
        Self::Bool,
        Self::I32,
        Self::U32,
        Self::I64,
        Self::U64,
        Self::Index,
        Self::F32,
        Self::F64,
    ];

    /// Significant bits of a value: 1 for `Bool`.
    pub const fn bits(self) -> u32 {
        match self {
            Self::Bool => 1,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::Index | Self::F64 => 64,
        }
    }

    /// Bytes of a value in memory: one for `Bool`.
    pub const fn bytes(self) -> usize {
        match self {
            Self::Bool => 1,
            _ => self.bits() as usize / 8,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    pub const fn is_integer(self) -> bool {
        !matches!(self, Self::Bool | Self::F32 | Self::F64)
    }

    pub const fn is_signed(self) -> bool {
        matches!(self, Self::I32 | Self::I64)
    }

    /// The element of a slice whose elements have this type.
    pub const fn element(self) -> Option<Element> {
        Some(match self {
            Self::I32 => Element::I32,
            Self::U32 => Element::U32,
            Self::I64 => Element::I64,
            Self::F32 => Element::F32,
            Self::F64 => Element::F64,
            Self::Bool | Self::U64 | Self::Index => return None,
        })
    }

    /// The scalar type of a slice element, if a scalar register can hold it.
    pub const fn of_element(element: Element) -> Option<Self> {
        Some(match element {
            Element::I32 => Self::I32,
            Element::U32 => Self::U32,
            Element::I64 => Self::I64,
            Element::F32 => Self::F32,
            Element::F64 => Self::F64,
            Element::U16 => return None,
        })
    }

    /// The Rust name of the type.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::U64 => "u64",
            Self::Index => "usize",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// Whether Rust's `as` converts a value of this type to `to`: numbers
    /// convert to numbers and `bool` to integers.
    pub const fn casts_to(self, to: Self) -> bool {
        match (self, to) {
            (_, Self::Bool) => matches!(self, Self::Bool),
            (Self::Bool, _) => to.is_integer(),
            _ => true,
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
        match self.ty {
            ScalarType::I32 => self.bits as u32 as i32 as i64 as u64,
            _ => self.bits,
        }
    }

    /// An integer or `bool`, exactly.
    const fn integer(self) -> i128 {
        match self.ty {
            ScalarType::I32 => self.bits as u32 as i32 as i128,
            ScalarType::I64 => self.bits as i64 as i128,
            _ => self.bits as i128,
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
                let saturated = match to {
                    ScalarType::I32 => value as i32 as i128,
                    ScalarType::U32 => value as u32 as i128,
                    ScalarType::I64 => value as i64 as i128,
                    _ => value as u64 as i128,
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
constant_from!(bool => Bool, i32 => I32, u32 => U32, i64 => I64, u64 => U64, usize => Index);

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
    /// `Bool` supports `Eq`, `Ne`, `And`, `Or` and `Xor`; floats support
    /// `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max` and comparisons; integers
    /// support every operation.
    pub const fn output(self, ty: ScalarType) -> Option<ScalarType> {
        let applies = match ty {
            ScalarType::Bool => {
                matches!(self, Self::Eq | Self::Ne | Self::And | Self::Or | Self::Xor)
            }
            ScalarType::F32 | ScalarType::F64 => {
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
    /// shift amount outside `0..width`. Integer `Div` and `Rem` trap on a
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
        let (min, max) = if ty.is_signed() {
            (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
        } else {
            (0, (1i128 << bits) - 1)
        };
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
    }
}
