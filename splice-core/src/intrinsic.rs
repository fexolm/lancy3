//! Kernel intrinsics: methods declared without a body by `#[intrinsic]`,
//! whose calls reach the backend as one operation.
//!
//! An intrinsic is a method of an `#[intrinsic] trait`, implemented for a
//! kernel type by an `#[intrinsic] impl` (`<i32 as splice::Math>::count_ones`).
//! Its identity is structural, an [`IntrinsicId`]: the declaring trait, the
//! generic arguments (the implementing type first), the method and the
//! signature. There is no fallback: every backend implements every
//! intrinsic Splice declares, and a call of an intrinsic that the build's
//! backend does not implement fails the build with an error that names
//! both ([`unsupported`]).
use crate::hir::{ScalarType, ValueType};
use crate::simd::Element;

/// The item that declares an intrinsic: the path of an `#[intrinsic]`
/// trait, as `module_path!()` spells the module that declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Owner {
    Trait(&'static str),
}

impl Owner {
    pub const fn path(self) -> &'static str {
        match self {
            Self::Trait(path) => path,
        }
    }
}

/// A generic argument of an intrinsic's owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GenericArg {
    Type(ValueType),
}

/// The structural identity of an intrinsic: `<i32 as splice::Math>::count_ones`
/// with signature `(i32) -> u32`. Backends match on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntrinsicId {
    pub owner: Owner,
    /// The owner's generic arguments; for a trait, the implementing type
    /// comes first.
    pub generics: &'static [GenericArg],
    pub method: &'static str,
    /// The operand types, the receiver first.
    pub params: &'static [ValueType],
    pub result: ValueType,
}

impl IntrinsicId {
    /// The implementing type of a trait intrinsic.
    pub const fn self_type(&self) -> Option<ValueType> {
        match self.generics.first() {
            Some(GenericArg::Type(ty)) => Some(*ty),
            None => None,
        }
    }

    /// Whether this is `method` of `owner` for the implementing type `ty`.
    pub fn is(&self, owner: Owner, method: &str, ty: ScalarType) -> bool {
        self.owner == owner
            && self.method == method
            && self.self_type() == Some(ValueType::Scalar(ty))
    }
}

/// A kernel type an intrinsic takes or returns.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a kernel type an intrinsic can take or return"
)]
pub trait IntrinsicValue {
    const TYPE: ValueType;
}

macro_rules! scalar_value {
    ($($ty:ty => $scalar:ident),*) => {$(
        impl IntrinsicValue for $ty {
            const TYPE: ValueType = ValueType::Scalar(ScalarType::$scalar);
        }
    )*};
}
scalar_value!(bool => Bool, i32 => I32, u32 => U32, i64 => I64, u64 => U64, usize => Index, f32 => F32, f64 => F64);

/// Every intrinsic Splice declares. Every backend implements all of them.
pub fn declared() -> Vec<IntrinsicId> {
    crate::math::INTRINSICS.concat()
}

/// The build error for a call of `id`, which `backend` does not implement.
pub fn unsupported(backend: &str, id: &IntrinsicId) -> crate::BuildError {
    crate::BuildError::unsupported(format!(
        "intrinsic {id} is not implemented by backend `{backend}`"
    ))
}

fn value_type(f: &mut std::fmt::Formatter<'_>, ty: ValueType) -> std::fmt::Result {
    let element = |e: Element| format!("{e:?}").to_lowercase();
    match ty {
        ValueType::Scalar(ty) => write!(f, "{ty}"),
        ValueType::Vector(e) => write!(f, "Lane<{}>", element(e)),
        ValueType::Mask(e) => write!(f, "Mask<{}>", element(e)),
    }
}

impl std::fmt::Display for IntrinsicId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Owner::Trait(path) = self.owner;
        f.write_str("`<")?;
        match self.generics.first() {
            Some(GenericArg::Type(ty)) => value_type(f, *ty)?,
            None => f.write_str("_")?,
        }
        write!(f, " as {path}>::{}(", self.method)?;
        for (i, ty) in self.params.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            value_type(f, *ty)?;
        }
        f.write_str(") -> ")?;
        value_type(f, self.result)?;
        f.write_str("`")
    }
}

/// Stage the intrinsic `id` on the operands a kernel passes: a known
/// operand becomes a constant of its parameter's type. Generated code
/// calls it.
#[doc(hidden)]
pub fn stage(
    builder: &mut dyn crate::frontend::KernelBuilder,
    id: &IntrinsicId,
    operands: &[crate::frontend::methods::Operand],
) -> crate::Result<crate::hir::Reg> {
    use crate::frontend::methods::Operand;
    if operands.len() != id.params.len() {
        return Err(crate::BuildError::staging(format!(
            "intrinsic {id} takes {} operand(s), got {}",
            id.params.len(),
            operands.len()
        )));
    }
    let mut args = Vec::with_capacity(operands.len());
    for (&operand, &ty) in operands.iter().zip(id.params) {
        args.push(match (operand, ty) {
            (Operand::Reg(reg), _) => reg,
            (Operand::Known(value), ValueType::Scalar(ty)) => {
                let value = value.cast(ty).ok_or_else(|| {
                    crate::BuildError::staging(format!(
                        "intrinsic {id}: a known operand does not convert to {ty}"
                    ))
                })?;
                builder.constant(value)?
            }
            _ => {
                return Err(crate::BuildError::staging(format!(
                    "intrinsic {id} takes runtime values of its parameter types"
                )));
            }
        });
    }
    builder.intrinsic(id, &args)
}
