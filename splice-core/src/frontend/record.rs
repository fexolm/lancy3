//! Struct values while a kernel is staged: the registers of their scalar
//! fields. A struct a kernel reads from memory is read field by field
//! ([`KernelBuilder::load_field`]); inside the kernel it is these
//! registers, like a struct literal's aggregate.
use crate::frontend::KernelBuilder;
use crate::hir::{Constant, DataType, Element, Reg, Slice, ValueType};
use crate::{BuildError, Result};

/// A value of a [`DataType`]: one register per scalar field, in the order
/// of [`DataType::leaf`].
#[derive(Clone, Debug)]
pub struct Record {
    ty: DataType,
    leaves: Vec<Reg>,
}

/// The type of the elements of `slice`.
fn element(builder: &dyn KernelBuilder, slice: Slice) -> Result<Element> {
    match builder.reg_type(slice.ptr) {
        Some(ValueType::Ptr(kind)) => Ok(kind.element()),
        _ => Err(BuildError::verify("a struct is read from a slice")),
    }
}

fn element_type(builder: &dyn KernelBuilder, slice: Slice) -> Result<DataType> {
    match element(builder, slice)? {
        Element::Scalar(ty) => Ok(DataType::Scalar(ty)),
        Element::Struct(ty) => Ok(DataType::Struct(ty)),
        Element::Slice(_) => Err(BuildError::staging(
            "a slice of slices holds no struct fields",
        )),
    }
}

fn project(ty: DataType, path: &[&str]) -> Result<(DataType, usize)> {
    ty.project(path)
        .ok_or_else(|| BuildError::staging(format!("{ty} has no field `{}`", path.join("."))))
}

fn mismatch(expected: DataType, actual: DataType) -> BuildError {
    BuildError::staging(format!("expected a value of type {expected}, got {actual}"))
}

impl Record {
    /// The type of the field `path` names in an element of `slice`.
    pub fn field_type(
        builder: &dyn KernelBuilder,
        slice: Slice,
        path: &[&str],
    ) -> Result<DataType> {
        Ok(project(element_type(builder, slice)?, path)?.0)
    }

    /// Field `path` of element `index` of `slice` (the element when `path`
    /// is empty): only the scalars it holds are read.
    pub fn load(
        builder: &mut dyn KernelBuilder,
        slice: Slice,
        index: Reg,
        path: &[&str],
    ) -> Result<Self> {
        let element = element_type(builder, slice)?;
        let (ty, first) = project(element, path)?;
        let leaves = match element {
            DataType::Scalar(_) => vec![builder.load(slice, index)?],
            DataType::Struct(_) => (first..first + ty.leaf_count())
                .map(|field| builder.load_field(slice, index, field))
                .collect::<Result<_>>()?,
        };
        Ok(Self { ty, leaves })
    }

    /// The value of a struct parameter, which the call passes as a slice
    /// of one element.
    pub fn param(builder: &mut dyn KernelBuilder, slice: Slice) -> Result<Self> {
        let zero = builder.constant(Constant::from(0usize))?;
        Self::load(builder, slice, zero, &[])
    }

    /// Write this value to field `path` of element `index` of `slice`.
    pub fn store(
        &self,
        builder: &mut dyn KernelBuilder,
        slice: Slice,
        index: Reg,
        path: &[&str],
    ) -> Result<()> {
        let element = element_type(builder, slice)?;
        let (ty, first) = project(element, path)?;
        if ty != self.ty {
            return Err(mismatch(ty, self.ty));
        }
        match element {
            DataType::Scalar(_) => builder.store(slice, index, self.leaves[0]),
            DataType::Struct(_) => {
                for (field, &value) in (first..).zip(&self.leaves) {
                    builder.store_field(slice, index, field, value)?;
                }
                Ok(())
            }
        }
    }

    /// A scalar register as a value.
    pub fn of(builder: &dyn KernelBuilder, reg: Reg) -> Result<Self> {
        match builder.reg_type(reg) {
            Some(ValueType::Scalar(ty)) => Ok(Self {
                ty: DataType::Scalar(ty),
                leaves: vec![reg],
            }),
            _ => Err(BuildError::staging("a struct field holds a scalar")),
        }
    }

    /// A known scalar as a value of type `ty`. Rustc typed the source by
    /// its use, but the macro reads an unsuffixed literal as an `i32` or an
    /// `f64`: a number converts to another type of its kind.
    pub fn constant(
        builder: &mut dyn KernelBuilder,
        ty: DataType,
        value: Constant,
    ) -> Result<Self> {
        let source = DataType::Scalar(value.ty());
        let DataType::Scalar(target) = ty else {
            return Err(mismatch(ty, source));
        };
        let same_kind = (value.ty().is_float() && target.is_float())
            || (value.ty().is_integer() && target.is_integer())
            || value.ty() == target;
        let value = value
            .cast(target)
            .filter(|_| same_kind)
            .ok_or_else(|| mismatch(ty, source))?;
        let reg = builder.constant(value)?;
        Self::of(builder, reg)
    }

    /// A struct literal of type `ty` from its fields by name, in any order.
    pub fn literal(ty: DataType, fields: Vec<(&str, Self)>) -> Result<Self> {
        let DataType::Struct(layout) = ty else {
            return Err(BuildError::staging(format!("{ty} is not a struct")));
        };
        let mut leaves = Vec::with_capacity(ty.leaf_count());
        for field in layout.fields {
            let value = fields
                .iter()
                .find(|(name, _)| *name == field.name)
                .map(|(_, value)| value)
                .ok_or_else(|| {
                    BuildError::staging(format!("{ty} literal has no field `{}`", field.name))
                })?;
            if value.ty != field.ty {
                return Err(mismatch(field.ty, value.ty));
            }
            leaves.extend_from_slice(&value.leaves);
        }
        Ok(Self { ty, leaves })
    }

    /// A tuple struct of type `ty` from its fields `"0"`, `"1"`, ...; a
    /// `transparent` newtype is its one field.
    pub fn tuple(ty: DataType, fields: Vec<(&str, Self)>) -> Result<Self> {
        match (ty, &fields[..]) {
            (DataType::Scalar(_), [(_, value)]) if value.ty == ty => Ok(value.clone()),
            (DataType::Scalar(_), [(_, value)]) => Err(mismatch(ty, value.ty)),
            _ => Self::literal(ty, fields),
        }
    }

    pub fn ty(&self) -> DataType {
        self.ty
    }

    /// The type of field `path` of a value of type `ty`.
    pub fn type_of(ty: DataType, path: &[&str]) -> Result<DataType> {
        Ok(project(ty, path)?.0)
    }

    /// The type of field `path` of this value.
    pub fn path_type(&self, path: &[&str]) -> Result<DataType> {
        Ok(project(self.ty, path)?.0)
    }

    /// Field `path` of this value.
    pub fn field(&self, path: &[&str]) -> Result<Self> {
        let (ty, first) = project(self.ty, path)?;
        Ok(Self {
            ty,
            leaves: self.leaves[first..first + ty.leaf_count()].to_vec(),
        })
    }

    /// The register of a scalar value.
    pub fn scalar(&self) -> Result<Reg> {
        match self.ty {
            DataType::Scalar(_) => Ok(self.leaves[0]),
            DataType::Struct(ty) => Err(BuildError::unsupported(format!(
                "a {} value is used as a scalar: kernels do not compare, return or pass \
                 struct values to calls yet",
                ty.name
            ))),
        }
    }

    /// The same value in fresh registers, which later assignments to
    /// either value leave apart.
    pub fn copy(&self, builder: &mut dyn KernelBuilder) -> Result<Self> {
        let leaves = self
            .leaves
            .iter()
            .map(|&leaf| builder.copy(leaf))
            .collect::<Result<_>>()?;
        Ok(Self {
            ty: self.ty,
            leaves,
        })
    }

    /// Assign `value` to field `path` of this value, which a mutable
    /// binding holds in registers of its own.
    pub fn assign(
        &self,
        builder: &mut dyn KernelBuilder,
        path: &[&str],
        value: &Self,
    ) -> Result<()> {
        let target = self.field(path)?;
        if target.ty != value.ty {
            return Err(mismatch(target.ty, value.ty));
        }
        for (&dst, &src) in target.leaves.iter().zip(&value.leaves) {
            builder.assign(dst, src)?;
        }
        Ok(())
    }
}
