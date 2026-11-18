use super::{Call, Edge};

crate::slot_key!(pub Value);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Type {
    I32,
    Ptr,
}

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
}

impl Binary {
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
        })
    }
}

impl Op {
    pub(crate) fn pure(self) -> bool {
        matches!(
            self,
            Self::Const(_) | Self::Copy(_) | Self::PtrOffset(..) | Self::Bin(..)
        ) && !matches!(self, Self::Bin(Binary::Div | Binary::Rem, ..))
    }
    pub(super) fn produces_value(self) -> bool {
        !matches!(self, Self::Store(..) | Self::BoundsCheck(..))
    }

    pub(crate) fn map_operands(&mut self, mut map: impl FnMut(Value) -> Value) {
        match self {
            Self::Copy(v) | Self::Load(_, v) => *v = map(*v),
            Self::Bin(_, a, b)
            | Self::PtrOffset(a, b, _)
            | Self::Store(a, b)
            | Self::BoundsCheck(a, b) => {
                *a = map(*a);
                *b = map(*b);
            }
            Self::Const(_) | Self::Call(_) => {}
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
