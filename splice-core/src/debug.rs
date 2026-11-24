//! Source-level metadata supplied by the frontend to native code generators.

use crate::hir::{Buffer, BufferKind, Function, Reg, ValueType};
use crate::simd::Element;

#[derive(Clone, Copy, Debug)]
pub struct Location {
    pub file: &'static str,
    pub directory: &'static str,
    pub line: u32,
    pub column: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Register(Reg, Scalar),
    Slice(Buffer),
    Constant(Option<Constant>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    I32,
    Bool,
    Index,
}

#[derive(Clone, Copy, Debug)]
pub struct Constant {
    bits: u64,
    ty: ConstantType,
}

impl Constant {
    pub fn parts(self) -> (ConstantType, u64) {
        (self.ty, self.bits)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstantType {
    Bool,
    Signed(u8),
    Unsigned(u8),
    Usize,
    Float(u8),
}

/// Recognize primitive specialization values without adding bounds to the
/// kernel or borrowing its host-only data for the lifetime of the JIT.
#[doc(hidden)]
pub struct ConstantArgument<'a, T: ?Sized>(pub &'a T);

#[doc(hidden)]
pub trait SnapshotConstant {
    fn snapshot(self) -> Option<Constant>;
}

impl<T: ?Sized> SnapshotConstant for &&ConstantArgument<'_, T> {
    fn snapshot(self) -> Option<Constant> {
        None
    }
}

macro_rules! primitive_constants {
    ($($ty:ty => $kind:expr),* $(,)?) => {$(
        impl SnapshotConstant for &ConstantArgument<'_, $ty> {
            fn snapshot(self) -> Option<Constant> {
                Some(Constant { bits: *self.0 as u64, ty: $kind })
            }
        }
    )*};
}

primitive_constants!(
    bool => ConstantType::Bool,
    i8 => ConstantType::Signed(1),
    i16 => ConstantType::Signed(2),
    i32 => ConstantType::Signed(4),
    i64 => ConstantType::Signed(8),
    u8 => ConstantType::Unsigned(1),
    u16 => ConstantType::Unsigned(2),
    u32 => ConstantType::Unsigned(4),
    u64 => ConstantType::Unsigned(8),
    usize => ConstantType::Usize,
);

impl SnapshotConstant for &ConstantArgument<'_, f32> {
    fn snapshot(self) -> Option<Constant> {
        Some(Constant {
            bits: u64::from(self.0.to_bits()),
            ty: ConstantType::Float(4),
        })
    }
}

impl SnapshotConstant for &ConstantArgument<'_, f64> {
    fn snapshot(self) -> Option<Constant> {
        Some(Constant {
            bits: self.0.to_bits(),
            ty: ConstantType::Float(8),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Variable {
    pub name: &'static str,
    pub value: Value,
}

#[derive(Clone, Debug)]
pub struct Point {
    pub location: Location,
    pub variables: Box<[Variable]>,
}

/// Shared layout of the values saved at a source statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(Scalar),
    Vector { element: Element, lanes: usize },
    Slice(Element),
    Constant(ConstantType),
}

impl Type {
    pub fn storage_bytes(self) -> usize {
        match self {
            Self::Scalar(Scalar::Index) => 8,
            Self::Scalar(_) => 4,
            Self::Vector { element, lanes } => element.bytes() * lanes,
            Self::Slice(_) => 16,
            Self::Constant(_) => 0,
        }
    }
}

/// Where a source point saves one of its runtime variables.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub variable: Variable,
    pub ty: Type,
    /// Byte offset in the point's debug workspace.
    pub offset: usize,
}

impl Point {
    /// The runtime variables this point saves, for registers of the given
    /// types and vectors of `vector_bytes`.
    pub fn slots<'a>(
        &'a self,
        reg_type: impl Fn(Reg) -> ValueType + 'a,
        buffers: &'a [BufferKind],
        vector_bytes: usize,
    ) -> impl Iterator<Item = Slot> + 'a {
        let vector = move |element: Element| Type::Vector {
            element,
            lanes: element.lanes(vector_bytes),
        };
        self.variables
            .iter()
            .filter_map(move |&variable| {
                let ty = match variable.value {
                    Value::Register(reg, scalar) => match reg_type(reg) {
                        ValueType::Scalar => Type::Scalar(scalar),
                        ValueType::Vector(element) => vector(element),
                        ValueType::Mask(element) => vector(element.index()),
                    },
                    Value::Slice(buffer) => Type::Slice(buffers[buffer.index()].element()),
                    Value::Constant(_) => return None,
                };
                Some((variable, ty))
            })
            .scan(0usize, |offset, (variable, ty)| {
                let bytes = ty.storage_bytes();
                let start = offset.next_multiple_of(bytes.min(8));
                *offset = start + bytes;
                Some(Slot {
                    variable,
                    ty,
                    offset: start,
                })
            })
    }

    /// The slots of this point in a portable function.
    pub fn function_slots<'a>(
        &'a self,
        function: &'a Function,
        vector_bytes: usize,
    ) -> impl Iterator<Item = Slot> + 'a {
        self.slots(
            |r| function.reg_type(r),
            function.buffer_kinds(),
            vector_bytes,
        )
    }

    /// Bytes of the debug workspace for vectors of `vector_bytes`.
    pub fn storage_bytes(&self, function: &Function, vector_bytes: usize) -> usize {
        self.function_slots(function, vector_bytes)
            .last()
            .map_or(0, |slot| slot.offset + slot.ty.storage_bytes())
    }
}
