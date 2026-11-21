//! Source-level metadata supplied by the frontend to native code generators.

use crate::hir::{Buffer, Function, Reg, ValueType};
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

impl Value {
    pub(crate) fn register(self) -> Option<Reg> {
        match self {
            Self::Register(reg, _) => Some(reg),
            Self::Slice(_) | Self::Constant(_) => None,
        }
    }
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
    Vector(Element),
    Slice(Element),
    Constant(ConstantType),
}

impl Type {
    pub fn storage_bytes(self) -> usize {
        match self {
            Self::Scalar(Scalar::Index) => 8,
            Self::Scalar(_) => 4,
            Self::Vector(_) => crate::simd::vector_bytes(),
            Self::Slice(_) => 16,
            Self::Constant(_) => 0,
        }
    }
}

impl Point {
    pub fn slots<'a>(
        &'a self,
        function: &'a Function,
    ) -> impl Iterator<Item = (Variable, Type, usize)> + 'a {
        self.variables
            .iter()
            .filter_map(move |&var| {
                let ty = match var.value {
                    Value::Register(reg, scalar) => match function.reg_type(reg) {
                        ValueType::Scalar => Type::Scalar(scalar),
                        ValueType::Vector(element) => Type::Vector(element),
                        ValueType::Mask(element) => Type::Vector(element.index()),
                    },
                    Value::Slice(buffer) => {
                        Type::Slice(function.buffer_kinds()[buffer.index()].element())
                    }
                    Value::Constant(_) => return None,
                };
                Some((var, ty))
            })
            .scan(0usize, |offset, (var, ty)| {
                let bytes = ty.storage_bytes();
                let start = offset.next_multiple_of(bytes.min(8));
                *offset = start + bytes;
                Some((var, ty, start))
            })
    }

    pub fn storage_bytes(&self, function: &Function) -> usize {
        self.slots(function)
            .last()
            .map_or(0, |(_, ty, offset)| offset + ty.storage_bytes())
    }
}
