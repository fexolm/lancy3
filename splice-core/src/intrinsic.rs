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

/// The item that declares an intrinsic, by its path as `module_path!()`
/// spells the declaring module: an `#[intrinsic]` trait, or an
/// `#[intrinsic]` struct whose `#[intrinsic] impl` declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Owner {
    Trait(&'static str),
    Type(&'static str),
}

impl Owner {
    pub const fn path(self) -> &'static str {
        match self {
            Self::Trait(path) | Self::Type(path) => path,
        }
    }
}

/// The type of a value of an `#[intrinsic] struct`: the struct's path and
/// generic arguments. Only a backend that implements the struct's
/// intrinsics represents it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OpaqueType {
    pub path: &'static str,
    pub generics: &'static [GenericArg],
}

/// An `#[intrinsic] struct`. A field of an exported kernel's `BuildArgs`
/// or `CallArgs` cannot hold one (nor a reference to one): only kernels
/// pass intrinsic values to each other.
#[diagnostic::on_unimplemented(message = "`{Self}` is not an #[intrinsic] struct")]
pub trait IntrinsicType {
    const TYPE: OpaqueType;
}

impl<T: IntrinsicType> IntrinsicValue for T {
    const TYPE: ValueType = ValueType::Opaque(T::TYPE);
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

/// The build error for a value of the intrinsic type `ty`, which
/// `backend` does not represent.
pub fn unsupported_type(backend: &str, ty: &OpaqueType) -> crate::BuildError {
    crate::BuildError::unsupported(format!(
        "intrinsic type `{ty}` is not implemented by backend `{backend}`"
    ))
}

/// Check, before `B` compiles `function`, that `B` implements every
/// intrinsic that `function` and its callees call on `target`, and that
/// they hold no value of an `#[intrinsic] struct` type, which no backend
/// represents. Backends lower only functions that pass.
pub(crate) fn check<B: crate::Backend>(
    function: &crate::hir::Function,
    target: &crate::Target,
) -> crate::Result<()> {
    let backend = std::any::type_name::<B>()
        .rsplit("::")
        .next()
        .unwrap_or("?");
    for f in function.reachable_functions() {
        for inst in f.instructions() {
            if let crate::hir::Instr::Intrinsic { id, .. } = inst
                && !B::supports_intrinsic(id, target)
            {
                return Err(unsupported(backend, id));
            }
        }
        for index in 0..f.register_count() {
            if let ValueType::Opaque(ty) = f.reg_type(crate::hir::Reg(index as u32)) {
                return Err(unsupported_type(backend, &ty));
            }
        }
    }
    Ok(())
}

fn value_type(f: &mut std::fmt::Formatter<'_>, ty: ValueType) -> std::fmt::Result {
    let element = |e: Element| format!("{e:?}").to_lowercase();
    match ty {
        ValueType::Scalar(ty) => write!(f, "{ty}"),
        ValueType::Vector(e) => write!(f, "Lane<{}>", element(e)),
        ValueType::Mask(e) => write!(f, "Mask<{}>", element(e)),
        ValueType::Opaque(ty) => write!(f, "{ty}"),
    }
}

fn generics(f: &mut std::fmt::Formatter<'_>, args: &[GenericArg]) -> std::fmt::Result {
    if args.is_empty() {
        return Ok(());
    }
    f.write_str("<")?;
    for (i, GenericArg::Type(ty)) in args.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        value_type(f, *ty)?;
    }
    f.write_str(">")
}

impl std::fmt::Display for OpaqueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.path)?;
        generics(f, self.generics)
    }
}

impl std::fmt::Display for IntrinsicId {
    /// `<i32 as splice_core::math::Math>::count_ones(i32) -> u32` for a
    /// trait's intrinsic, `crate::Atomic<i32>::swap(crate::Atomic<i32>, i32) -> i32`
    /// for a struct's.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("`")?;
        match self.owner {
            Owner::Trait(path) => {
                f.write_str("<")?;
                match self.generics.first() {
                    Some(GenericArg::Type(ty)) => value_type(f, *ty)?,
                    None => f.write_str("_")?,
                }
                write!(f, " as {path}")?;
                generics(f, self.generics.get(1..).unwrap_or_default())?;
                f.write_str(">")?;
            }
            Owner::Type(path) => {
                f.write_str(path)?;
                generics(f, self.generics)?;
            }
        }
        write!(f, "::{}(", self.method)?;
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

/// The check that keeps `#[intrinsic] struct` values out of an exported
/// kernel's `BuildArgs` and `CallArgs`: generated code calls
/// `crossing((&Probe::<T>::new()).__splice_boundary())` for each field
/// type `T` (references peeled). Method resolution picks [`Opaque`] when
/// `T` is an intrinsic struct and [`Plain`] otherwise; only [`Plain`]'s
/// result is [`Crossing`].
#[doc(hidden)]
pub mod boundary {
    use super::IntrinsicType;
    use std::marker::PhantomData;

    pub struct Probe<T: ?Sized>(PhantomData<fn() -> *const T>);

    #[allow(clippy::new_without_default)]
    impl<T: ?Sized> Probe<T> {
        pub const fn new() -> Self {
            Self(PhantomData)
        }
    }

    pub struct Accepted;
    pub struct Rejected;

    #[diagnostic::on_unimplemented(
        message = "an #[intrinsic] struct cannot be a field of an exported kernel's `BuildArgs` or `CallArgs`",
        label = "a value of an intrinsic type",
        note = "only kernels pass intrinsic values to each other"
    )]
    pub trait Crossing {}
    impl Crossing for Accepted {}

    pub fn crossing<C: Crossing>(_: C) {}

    pub trait Opaque {
        fn __splice_boundary(&self) -> Rejected {
            Rejected
        }
    }
    impl<T: IntrinsicType> Opaque for Probe<T> {}

    pub trait Plain {
        fn __splice_boundary(&self) -> Accepted {
            Accepted
        }
    }
    impl<T: ?Sized> Plain for &Probe<T> {}
}
