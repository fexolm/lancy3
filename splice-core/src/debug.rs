//! Source-level metadata supplied by the frontend to native code generators.

use crate::hir::{Function, Reg, ScalarType, Slice, ValueType};

/// A point in a kernel's source. `#[kernel]` gives an absolute `file`; a
/// relative one is relative to `directory`, the compilation directory.
#[derive(Clone, Copy, Debug)]
pub struct Location {
    pub file: &'static str,
    pub directory: &'static str,
    pub line: u32,
    pub column: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Register(Reg),
    Slice(Slice),
    Constant(Option<Constant>),
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
    /// A scalar in its memory layout: a `bool` is one byte holding 0 or 1.
    Scalar(ScalarType),
    Vector {
        element: ScalarType,
        lanes: usize,
    },
    Slice(ScalarType),
    Constant(ConstantType),
}

impl Type {
    pub fn storage_bytes(self) -> usize {
        match self {
            Self::Scalar(ty) => ty.bytes(),
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

/// One store that saves (part of) a slot. Every backend writes the same
/// bytes for it, in its own representation of the value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Store {
    /// A scalar register in its memory layout: a `bool` is one byte, 0 or 1.
    Scalar(Reg, ScalarType),
    /// A vector register.
    Vector(Reg),
    /// A mask register as the vector of its index lanes: all bits set for
    /// an active lane, zero otherwise.
    Mask(Reg, ScalarType),
    /// The data pointer of a slice, this register.
    Data(Reg),
    /// The length of a slice, this `Index` register, as 64 bits.
    Length(Reg),
}

impl Slot {
    /// The stores that save this slot, at their byte offsets in the point's
    /// workspace, for registers of the given types.
    pub fn stores(
        &self,
        reg_type: impl Fn(Reg) -> ValueType,
    ) -> impl Iterator<Item = (usize, Store)> {
        let (first, second) = match self.variable.value {
            Value::Register(reg) => (
                match reg_type(reg) {
                    ValueType::Scalar(ty) => Store::Scalar(reg, ty),
                    ValueType::Vector(_) => Store::Vector(reg),
                    ValueType::Mask(element) => Store::Mask(reg, element),
                    ValueType::Ptr(_) => unreachable!("a pointer is saved with its slice"),
                },
                None,
            ),
            Value::Slice(slice) => (
                Store::Data(slice.ptr),
                Some((self.offset + 8, Store::Length(slice.len))),
            ),
            Value::Constant(_) => unreachable!("only runtime values have slots"),
        };
        std::iter::once((self.offset, first)).chain(second)
    }
}

impl Point {
    /// The runtime variables this point saves, for registers of the given
    /// types and vectors of `vector_bytes`.
    pub fn slots<'a>(
        &'a self,
        reg_type: impl Fn(Reg) -> ValueType + 'a,
        vector_bytes: usize,
    ) -> impl Iterator<Item = Slot> + 'a {
        let vector = move |element: ScalarType| Type::Vector {
            element,
            lanes: element.lanes(vector_bytes),
        };
        self.variables
            .iter()
            .filter_map(move |&variable| {
                let ty = match variable.value {
                    Value::Register(reg) => match reg_type(reg) {
                        ValueType::Scalar(ty) => Type::Scalar(ty),
                        ValueType::Vector(element) => vector(element),
                        ValueType::Mask(element) => vector(element.index()),
                        ValueType::Ptr(_) => unreachable!("a pointer is saved with its slice"),
                    },
                    Value::Slice(slice) => match reg_type(slice.ptr) {
                        ValueType::Ptr(kind) => Type::Slice(kind.element()),
                        _ => unreachable!("a verified slice"),
                    },
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
        self.slots(|r| function.reg_type(r), vector_bytes)
    }

    /// Bytes of the debug workspace for vectors of `vector_bytes`.
    pub fn storage_bytes(&self, function: &Function, vector_bytes: usize) -> usize {
        self.function_slots(function, vector_bytes)
            .last()
            .map_or(0, |slot| slot.offset + slot.ty.storage_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_backend_saves_the_same_stores_for_each_slot() {
        let location = Location {
            file: "f.rs",
            directory: "",
            line: 1,
            column: 1,
        };
        let variable = |name, value| Variable { name, value };
        let point = Point {
            location,
            variables: [
                variable("flag", Value::Register(Reg(2))),
                variable("xs", Value::Slice(Slice::param(0, 0))),
                variable("known", Value::Constant(None)),
                variable("lanes", Value::Register(Reg(3))),
                variable("active", Value::Register(Reg(4))),
            ]
            .into(),
        };
        let types = [
            ValueType::Ptr(crate::BufferKind::new(ScalarType::U16, false)),
            ValueType::Scalar(ScalarType::Index),
            ValueType::Scalar(ScalarType::Bool),
            ValueType::Vector(ScalarType::F32),
            ValueType::Mask(ScalarType::F32),
        ];
        let reg_type = |r: Reg| types[r.0 as usize];
        let stores: Vec<_> = point
            .slots(reg_type, 16)
            .flat_map(|slot| slot.stores(reg_type).collect::<Vec<_>>())
            .collect();
        assert_eq!(
            stores,
            [
                (0, Store::Scalar(Reg(2), ScalarType::Bool)),
                (8, Store::Data(Reg(0))),
                (16, Store::Length(Reg(1))),
                (24, Store::Vector(Reg(3))),
                (40, Store::Mask(Reg(4), ScalarType::F32)),
            ]
        );
    }
}
