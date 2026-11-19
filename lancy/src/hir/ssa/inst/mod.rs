mod numeric;
mod simd;
use super::{Call, Edge};
pub use numeric::{Lane, Ternary, Type, Unary};
pub use simd::{Simd, VectorAddress};

crate::slot_key!(pub Value);

/// Byte stride for scaled pointer arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Scale {
    One = 0,
    Two = 1,
    Four = 2,
    Eight = 3,
}

/// Variable-sized operands live in side tables, keeping common operations inline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Op {
    Const(i32),
    /// Low bits of a scalar literal (IEEE bit pattern for floats).
    Literal(Lane, u128),
    Simd(Simd),
    Unary(Unary, Value),
    Ternary(Ternary, Value, Value, Value),
    Cast(Type, Value),
    Bitcast(Type, Value),
    Copy(Value),
    Bin(Binary, Value, Value),
    /// Trap if the unsigned index is at least the unsigned length.
    BoundsCheck(Value, Value),
    /// Base plus a signed i32 index times the byte stride, at pointer width.
    PtrOffset(Value, Value, Scale),
    /// Read a value of the given type from an address, allowing unaligned access.
    Load(Type, Value),
    /// Write the value to an address using the value's width.
    Store(Value, Value),
    Call(Call),
}

/// The operation performed by an `Op::Bin` instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Shl,
    Shr,
    Min,
    Max,
    Ordered,
    Unordered,
    CopySign,
    Pow,
    Atan2,
    Hypot,
    RotateLeft,
    RotateRight,
    SaturatingAdd,
    SaturatingSub,
}

impl Binary {
    pub(crate) fn scalar_alu(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Sub
                | Self::Mul
                | Self::Div
                | Self::Rem
                | Self::And
                | Self::Or
                | Self::Xor
                | Self::Eq
                | Self::Ne
                | Self::Lt
                | Self::Le
                | Self::Gt
                | Self::Ge
                | Self::Shl
                | Self::Shr
                | Self::Min
                | Self::Max
        )
    }
    pub(crate) fn result_type(self, ty: Type) -> Type {
        if matches!(
            self,
            Self::Ordered
                | Self::Unordered
                | Self::Eq
                | Self::Ne
                | Self::Lt
                | Self::Le
                | Self::Gt
                | Self::Ge
        ) {
            ty.mask_type()
        } else {
            ty
        }
    }
    pub(crate) fn supports(self, ty: Type) -> bool {
        let Some(lane) = ty.lane() else { return false };
        match self {
            Self::Ordered
            | Self::Unordered
            | Self::CopySign
            | Self::Pow
            | Self::Atan2
            | Self::Hypot => lane.float(),
            Self::Shl
            | Self::Shr
            | Self::RotateLeft
            | Self::RotateRight
            | Self::SaturatingAdd
            | Self::SaturatingSub => !lane.float(),
            _ => true,
        }
    }

    pub(crate) fn checked_eval(self, a: i32, b: i32) -> Option<i32> {
        Some(match self {
            Self::Add => a.wrapping_add(b),
            Self::Sub => a.wrapping_sub(b),
            Self::Mul => a.wrapping_mul(b),
            Self::Div => a.checked_div(b)?,
            Self::Rem => a.checked_rem(b)?,
            Self::And => a & b,
            Self::Or => a | b,
            Self::Xor => a ^ b,
            Self::Eq => -i32::from(a == b),
            Self::Ne => -i32::from(a != b),
            Self::Lt => -i32::from(a < b),
            Self::Le => -i32::from(a <= b),
            Self::Gt => -i32::from(a > b),
            Self::Ge => -i32::from(a >= b),
            Self::Shl => a.wrapping_shl(b as u32),
            Self::Shr => a.wrapping_shr(b as u32),
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            Self::RotateLeft => a.rotate_left(b as u32),
            Self::RotateRight => a.rotate_right(b as u32),
            Self::SaturatingAdd => a.saturating_add(b),
            Self::SaturatingSub => a.saturating_sub(b),
            _ => return None,
        })
    }
}

impl Op {
    pub(crate) fn pure(self, result_type: Option<Type>) -> bool {
        match self {
            Self::Bin(Binary::Div | Binary::Rem, ..)
            | Self::Simd(Simd::Bin(Binary::Div | Binary::Rem, ..)) => {
                result_type.and_then(Type::lane).is_some_and(Lane::float)
            }
            Self::Simd(
                Simd::Load(..)
                | Simd::Store(..)
                | Simd::MaskedLoad { .. }
                | Simd::MaskedStore { .. }
                | Simd::ExtractLane(..)
                | Simd::InsertLane(..),
            ) => false,
            Self::Load(..) | Self::Store(..) | Self::BoundsCheck(..) | Self::Call(_) => false,
            _ => true,
        }
    }
    pub(in crate::hir::ssa) fn produces_value(self) -> bool {
        !matches!(
            self,
            Self::Store(..)
                | Self::Simd(Simd::MaskedStore { .. } | Simd::Store(..))
                | Self::BoundsCheck(..)
        )
    }

    pub(crate) fn map_operands(&mut self, mut map: impl FnMut(Value) -> Value) {
        match self {
            Self::Simd(op) => op.map_operands(map),
            Self::Copy(v)
            | Self::Load(_, v)
            | Self::Unary(_, v)
            | Self::Cast(_, v)
            | Self::Bitcast(_, v) => *v = map(*v),
            Self::Ternary(_, a, b, c) => {
                *a = map(*a);
                *b = map(*b);
                *c = map(*c);
            }
            Self::Bin(_, a, b)
            | Self::PtrOffset(a, b, _)
            | Self::Store(a, b)
            | Self::BoundsCheck(a, b) => {
                *a = map(*a);
                *b = map(*b);
            }
            Self::Const(_) | Self::Literal(..) | Self::Call(_) => {}
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Terminator {
    Jump(Edge),
    Branch { cond: Value, yes: Edge, no: Edge },
    Return(Value),
}

impl Terminator {
    pub fn edges(self) -> impl Iterator<Item = Edge> {
        let empty = Edge::INVALID;
        let (edges, count) = match self {
            Self::Jump(edge) => ([edge, empty], 1),
            Self::Branch { yes, no, .. } => ([yes, no], 2),
            Self::Return(_) => ([empty, empty], 0),
        };
        edges.into_iter().take(count)
    }
}
