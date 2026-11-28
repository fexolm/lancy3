//! The memory layout of Rust values a kernel reads from and writes to
//! memory: scalars and the structs `#[derive(KernelType)]` describes.
//!
//! A layout comes from rustc (`size_of`, `align_of`, `offset_of!`) for the
//! process that stages the kernel, so a struct needs no `repr(C)`: code is
//! only generated for the architecture of the running process (see
//! [`Target::check_execution`](crate::Target::check_execution)), and portable
//! IR is not serialized. A backend never sees a struct: the frontend reads
//! and writes one scalar field ([`StructType::leaf`]) at a time.
use crate::hir::ScalarType;
use std::fmt;

/// The type of a value in memory: a scalar, or a struct of such values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    Scalar(ScalarType),
    Struct(&'static StructType),
}

impl DataType {
    /// The size of a value of this type, padding included.
    pub const fn bytes(self) -> usize {
        match self {
            Self::Scalar(ty) => ty.bytes(),
            Self::Struct(ty) => ty.size,
        }
    }

    /// The number of scalars in a value of this type.
    pub fn leaf_count(self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Struct(ty) => ty.fields.iter().map(|f| f.ty.leaf_count()).sum(),
        }
    }

    /// Scalar `index` of a value, in field declaration order with nested
    /// structs expanded in place: its byte offset in the value and its type.
    pub fn leaf(self, index: usize) -> Option<(usize, ScalarType)> {
        match self {
            Self::Scalar(ty) => (index == 0).then_some((0, ty)),
            Self::Struct(ty) => {
                let mut index = index;
                for field in ty.fields {
                    let count = field.ty.leaf_count();
                    if index < count {
                        let (offset, ty) = field.ty.leaf(index)?;
                        return Some((field.offset + offset, ty));
                    }
                    index -= count;
                }
                None
            }
        }
    }

    /// The field that `path` names, from this value: its type and the
    /// index of its first scalar. A step into a scalar names the scalar
    /// itself: Rust only accepts it for a `transparent` newtype, whose
    /// type is its field's.
    pub fn project(self, path: &[&str]) -> Option<(Self, usize)> {
        let Some((first, rest)) = path.split_first() else {
            return Some((self, 0));
        };
        match self {
            Self::Scalar(_) => self.project(rest),
            Self::Struct(ty) => {
                let mut start = 0;
                for field in ty.fields {
                    if field.name == *first {
                        let (ty, leaf) = field.ty.project(rest)?;
                        return Some((ty, start + leaf));
                    }
                    start += field.ty.leaf_count();
                }
                None
            }
        }
    }

    /// Check that every scalar lies within the value, at an offset its
    /// type is aligned to: a layout that claims otherwise could address
    /// memory outside an element.
    pub fn check(self) -> Result<(), String> {
        let Self::Struct(ty) = self else {
            return Ok(());
        };
        if ty.align == 0 || !ty.align.is_power_of_two() || ty.size % ty.align != 0 {
            return Err(format!("struct {} has an invalid alignment", ty.name));
        }
        for field in ty.fields {
            field.ty.check()?;
            let inside = field
                .offset
                .checked_add(field.ty.bytes())
                .is_some_and(|end| end <= ty.size);
            if !inside || field.offset % field.ty.align() != 0 {
                return Err(format!(
                    "field {} of struct {} lies outside it or is misaligned",
                    field.name, ty.name
                ));
            }
        }
        Ok(())
    }

    /// The alignment a valid layout gives values of this type.
    pub const fn align(self) -> usize {
        match self {
            Self::Scalar(ty) => ty.bytes(),
            Self::Struct(ty) => ty.align,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(ty) => f.write_str(ty.name()),
            Self::Struct(ty) => f.write_str(ty.name),
        }
    }
}

/// The layout of a Rust struct, as `#[derive(KernelType)]` reads it from
/// rustc: fields in declaration order with their offsets, which need not
/// increase (`repr(Rust)` may reorder fields).
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct StructType {
    /// The Rust type, for messages.
    pub name: &'static str,
    pub size: usize,
    pub align: usize,
    pub fields: &'static [Field],
}

/// A named field of a [`StructType`] (tuple fields by position).
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Field {
    pub name: &'static str,
    pub offset: usize,
    pub ty: DataType,
}

#[cfg(test)]
mod tests {
    use super::*;

    const INNER: StructType = StructType {
        name: "Inner",
        size: 8,
        align: 4,
        fields: &[
            Field {
                name: "a",
                offset: 4,
                ty: DataType::Scalar(ScalarType::U16),
            },
            Field {
                name: "b",
                offset: 0,
                ty: DataType::Scalar(ScalarType::F32),
            },
        ],
    };
    const OUTER: StructType = StructType {
        name: "Outer",
        size: 16,
        align: 4,
        fields: &[
            Field {
                name: "flag",
                offset: 12,
                ty: DataType::Scalar(ScalarType::Bool),
            },
            Field {
                name: "inner",
                offset: 4,
                ty: DataType::Struct(&INNER),
            },
        ],
    };

    #[test]
    fn nested_fields_are_scalars_in_declaration_order() {
        let outer = DataType::Struct(&OUTER);
        assert_eq!(outer.leaf_count(), 3);
        assert_eq!(outer.leaf(0), Some((12, ScalarType::Bool)));
        assert_eq!(outer.leaf(1), Some((8, ScalarType::U16)));
        assert_eq!(outer.leaf(2), Some((4, ScalarType::F32)));
        assert_eq!(outer.leaf(3), None);
        assert_eq!(
            outer.project(&["inner", "b"]),
            Some((DataType::Scalar(ScalarType::F32), 2))
        );
        assert_eq!(
            outer.project(&["inner"]),
            Some((DataType::Struct(&INNER), 1))
        );
        assert_eq!(outer.project(&["missing"]), None);
        assert_eq!(outer.check(), Ok(()));
    }

    #[test]
    fn a_field_outside_its_struct_is_rejected() {
        static FIELDS: [Field; 1] = [Field {
            name: "x",
            offset: 6,
            ty: DataType::Scalar(ScalarType::U32),
        }];
        static PAST_END: StructType = StructType {
            name: "PastEnd",
            size: 8,
            align: 4,
            fields: &FIELDS,
        };
        assert!(DataType::Struct(&PAST_END).check().is_err());
        static MISALIGNED_FIELDS: [Field; 1] = [Field {
            name: "x",
            offset: 2,
            ty: DataType::Scalar(ScalarType::U32),
        }];
        static MISALIGNED: StructType = StructType {
            name: "Misaligned",
            size: 8,
            align: 4,
            fields: &MISALIGNED_FIELDS,
        };
        assert!(DataType::Struct(&MISALIGNED).check().is_err());
    }
}
