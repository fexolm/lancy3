mod info;
mod numeric;
mod simd;
use super::{Call, Edge};
pub(crate) use info::{Effects, TrapKind};
pub use numeric::{Lane, Ternary, Type, Unary};
pub use simd::{Simd, VectorAddress};

crate::support::slotmap::slot_key!(
    /// An SSA value: a parameter or an instruction result.
    pub Value
);

/// Byte stride for scaled pointer arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Scale {
    /// 1 byte.
    One = 0,
    /// 2 bytes.
    Two = 1,
    /// 4 bytes.
    Four = 2,
    /// 8 bytes.
    Eight = 3,
}

pub use splice_target::abi::{RustEntry, RustFunction};

/// Variable-sized operands live in side tables, keeping common operations inline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Op {
    /// An `i32` literal.
    Const(i32),
    /// Low bits of a scalar literal (IEEE bit pattern for floats).
    Literal(Lane, u128),
    /// A vector operation without a scalar meaning.
    Simd(Simd),
    /// An element-wise unary operation.
    Unary(Unary, Value),
    /// An element-wise three-operand operation.
    Ternary(Ternary, Value, Value, Value),
    /// Convert numbers to the type: float to integer truncates and saturates,
    /// NaN gives zero. A vector cast keeps the lane width.
    Cast(Type, Value),
    /// Reinterpret the bits as the type of the same size.
    Bitcast(Type, Value),
    /// The operand itself.
    Copy(Value),
    /// An element-wise binary operation.
    Bin(Binary, Value, Value),
    /// Trap if the unsigned index is at least the unsigned length.
    BoundsCheck(Value, Value),
    /// Stop with the trap if the i32 condition is nonzero.
    TrapIf(Value, crate::Trap),
    /// Stop with the trap. Code after it in the block does not run.
    Trap(crate::Trap),
    /// Base plus a signed i32 index times the byte stride, at pointer width.
    PtrOffset(Value, Value, Scale),
    /// Address of `bytes` of this frame's scratch memory at an 8-byte-aligned
    /// `offset`. The frame's scratch covers its largest `offset + bytes`,
    /// starts uninitialized and holds the packets of the frame's Rust
    /// calls. Inlining moves a callee's scratch above the caller's.
    CallScratch {
        /// Byte offset in the frame's scratch, a multiple of 8.
        offset: u32,
        /// Bytes addressed from `offset`.
        bytes: u32,
    },
    /// Read a value of the given type from an address, allowing unaligned access.
    Load(Type, Value),
    /// Write the value to an address using the value's width.
    Store(Value, Value),
    /// Save a value in this invocation's debugger workspace at a byte offset.
    /// Inlining moves a callee's offsets above the caller's workspace.
    DebugStore(u32, Value),
    /// Associate the current machine-code position with an embedder's source point.
    DebugPoint(u32),
    /// Call a HIR function with the arguments recorded for the call.
    Call(Call),
    /// Call the Rust function with a `ptr` receiver, an `i32` operation, a
    /// `ptr` packet and an `i32` word. A trapping status stops the kernel.
    RustCall(&'static RustFunction, [Value; 4]),
}

/// The operation performed by an `Op::Bin` instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
    /// Wrapping addition for integers.
    Add,
    /// Wrapping subtraction for integers.
    Sub,
    /// Wrapping multiplication for integers.
    Mul,
    /// Division; for integers by the type's signedness, trapping on a zero
    /// divisor and on signed overflow.
    Div,
    /// Remainder with the sign of the dividend; traps as `Div`.
    Rem,
    /// Bitwise and.
    And,
    /// Bitwise or.
    Or,
    /// Bitwise exclusive or.
    Xor,
    /// Equal: zero or all ones, see [`Type`] for the mask type.
    Eq,
    /// Not equal.
    Ne,
    /// Less than, by the type's signedness.
    Lt,
    /// Less or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater or equal.
    Ge,
    /// Left shift by the count modulo the lane width.
    Shl,
    /// Right shift by the count modulo the lane width; arithmetic for signed types.
    Shr,
    /// For floats, IEEE 754-2019 minimumNumber: a NaN operand (quiet or
    /// signaling) yields the other operand, and -0 is below +0.
    Min,
    /// For floats, IEEE 754-2019 maximumNumber, with the rules of `Min`.
    Max,
    /// Neither float operand is NaN.
    Ordered,
    /// Either float operand is NaN.
    Unordered,
    /// The first operand with the sign of the second.
    CopySign,
    /// The first operand raised to the second.
    Pow,
    /// The four-quadrant arctangent of `a / b`.
    Atan2,
    /// `sqrt(a * a + b * b)` without intermediate overflow.
    Hypot,
    /// Rotate left by the count modulo the lane width.
    RotateLeft,
    /// Rotate right by the count modulo the lane width.
    RotateRight,
    /// Integer addition clamped to the type's range.
    SaturatingAdd,
    /// Integer subtraction clamped to the type's range.
    SaturatingSub,
}

impl Binary {
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
}

impl Op {
    pub(crate) fn map_operands(&mut self, mut map: impl FnMut(Value) -> Value) {
        match self {
            Self::Simd(op) => op.map_operands(map),
            Self::Copy(v)
            | Self::TrapIf(v, _)
            | Self::DebugStore(_, v)
            | Self::Load(_, v)
            | Self::Unary(_, v)
            | Self::Cast(_, v)
            | Self::Bitcast(_, v) => *v = map(*v),
            Self::Ternary(_, a, b, c) => {
                *a = map(*a);
                *b = map(*b);
                *c = map(*c);
            }
            Self::RustCall(_, args) => {
                for arg in args {
                    *arg = map(*arg);
                }
            }
            Self::Bin(_, a, b)
            | Self::PtrOffset(a, b, _)
            | Self::Store(a, b)
            | Self::BoundsCheck(a, b) => {
                *a = map(*a);
                *b = map(*b);
            }
            Self::Const(_)
            | Self::Literal(..)
            | Self::Call(_)
            | Self::DebugPoint(_)
            | Self::Trap(_)
            | Self::CallScratch { .. } => {}
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// How a block ends.
pub enum Terminator {
    /// Continue along the edge.
    Jump(Edge),
    /// Take `yes` if the `i32` condition is nonzero, else `no`.
    Branch {
        /// The `i32` condition.
        cond: Value,
        /// Taken if `cond` is nonzero.
        yes: Edge,
        /// Taken if `cond` is zero.
        no: Edge,
    },
    /// Return the value from the function.
    Return(Value),
}

impl Terminator {
    pub(crate) fn operand(self) -> Option<Value> {
        match self {
            Self::Branch { cond, .. } | Self::Return(cond) => Some(cond),
            Self::Jump(_) => None,
        }
    }

    /// The outgoing edges, `yes` before `no`.
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
