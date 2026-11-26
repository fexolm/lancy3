//! The registry of kernel intrinsics: named operations with the behavior of a
//! Rust method, staged as portable IR. `x.saturating_add(y)` on a runtime
//! integer and a `#[intrinsic("splice.i32.saturating_add")]` method both
//! resolve here, so every backend receives the same operation.
//!
//! An intrinsic is named `splice.<type>.<method>`, where `<type>` is the
//! Rust name of the receiver's scalar type. Every operand has that type.
use super::KernelBuilder;
use crate::hir::{Binary, Constant, Reg, ScalarType, ValueType};

/// How an intrinsic stages its operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lowering {
    /// One binary operation of the receiver and the argument.
    Binary(Binary),
    /// `0 - x` without an overflow check (`wrapping_neg`).
    WrappingNeg,
}

impl Lowering {
    fn arity(self) -> usize {
        match self {
            Self::Binary(_) => 2,
            Self::WrappingNeg => 1,
        }
    }
}

/// The method of an intrinsic and its lowering for the integer types.
const INTEGER_METHODS: [(&str, Lowering); 9] = [
    ("wrapping_add", Lowering::Binary(Binary::Add)),
    ("wrapping_sub", Lowering::Binary(Binary::Sub)),
    ("wrapping_mul", Lowering::Binary(Binary::Mul)),
    ("wrapping_neg", Lowering::WrappingNeg),
    ("saturating_add", Lowering::Binary(Binary::SaturatingAdd)),
    ("saturating_sub", Lowering::Binary(Binary::SaturatingSub)),
    ("saturating_mul", Lowering::Binary(Binary::SaturatingMul)),
    ("min", Lowering::Binary(Binary::Min)),
    ("max", Lowering::Binary(Binary::Max)),
];

/// The lowering of `method` on a receiver of type `ty`, if registered.
fn lookup(ty: ScalarType, method: &str) -> Option<Lowering> {
    if !ty.is_integer() {
        return None;
    }
    INTEGER_METHODS
        .iter()
        .find(|(name, _)| *name == method)
        .map(|&(_, lowering)| lowering)
}

/// Whether some scalar type has the intrinsic method `method`.
#[doc(hidden)]
pub const fn is_method(method: &str) -> bool {
    let mut i = 0;
    while i < INTEGER_METHODS.len() {
        if same(INTEGER_METHODS[i].0, method) {
            return true;
        }
        i += 1;
    }
    false
}

const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Stage the intrinsic `name` on `args` if it is registered and the
/// operands have its types; `Ok(None)` otherwise, and the caller stages
/// the method's own body.
pub fn call(
    builder: &mut dyn KernelBuilder,
    name: &str,
    args: &[Reg],
) -> crate::Result<Option<Reg>> {
    let Some((ty, method)) = name
        .strip_prefix("splice.")
        .and_then(|rest| rest.split_once('.'))
    else {
        return Ok(None);
    };
    let Some(ty) = ScalarType::ALL.into_iter().find(|t| t.name() == ty) else {
        return Ok(None);
    };
    match lookup(ty, method) {
        Some(lowering) if operands_have_type(builder, args, ty, lowering) => {
            stage(builder, lowering, ty, args).map(Some)
        }
        _ => Ok(None),
    }
}

/// Stage the Rust method `method` on runtime scalars: `args[0]` is the
/// receiver. A method without an intrinsic for the receiver's type, or
/// with operands of other types, is a staging error.
#[doc(hidden)]
pub fn method(builder: &mut dyn KernelBuilder, method: &str, args: &[Reg]) -> crate::Result<Reg> {
    let receiver = args.first().and_then(|&r| builder.reg_type(r));
    let Some(ValueType::Scalar(ty)) = receiver else {
        return Err(crate::BuildError::staging(format!(
            "`{method}` requires a runtime scalar receiver"
        )));
    };
    match lookup(ty, method) {
        Some(lowering) if operands_have_type(builder, args, ty, lowering) => {
            stage(builder, lowering, ty, args)
        }
        Some(lowering) => Err(crate::BuildError::staging(format!(
            "`{ty}::{method}` takes {} operand(s) of type {ty}",
            lowering.arity()
        ))),
        None => Err(crate::BuildError::staging(format!(
            "`{ty}::{method}` is not a kernel operation"
        ))),
    }
}

fn operands_have_type(
    builder: &dyn KernelBuilder,
    args: &[Reg],
    ty: ScalarType,
    lowering: Lowering,
) -> bool {
    args.len() == lowering.arity()
        && args
            .iter()
            .all(|&r| builder.reg_type(r) == Some(ValueType::Scalar(ty)))
}

fn stage(
    builder: &mut dyn KernelBuilder,
    lowering: Lowering,
    ty: ScalarType,
    args: &[Reg],
) -> crate::Result<Reg> {
    match lowering {
        Lowering::Binary(op) => builder.binary(op, args[0], args[1]),
        Lowering::WrappingNeg => {
            let zero = builder.constant(Constant::new(ty, 0))?;
            builder.binary(Binary::Sub, zero, args[0])
        }
    }
}
