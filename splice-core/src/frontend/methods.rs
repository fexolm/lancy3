//! Portable operations for Rust integer methods on runtime scalars.
use super::KernelBuilder;
use crate::hir::{Binary, Constant, Reg, ScalarType, ValueType};

/// How a core method stages its operands.
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

/// A core method of the scalar types and its lowering. A method applies to
/// the types its operation does: `min` and `max` also to `char`.
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

/// The lowering of `method` on a receiver of type `ty`, if it is a core method.
fn lookup(ty: ScalarType, method: &str) -> Option<Lowering> {
    let (_, lowering) = INTEGER_METHODS.iter().find(|(name, _)| *name == method)?;
    let applies = match *lowering {
        Lowering::Binary(op) => op.output(ty) == Some(ty),
        Lowering::WrappingNeg => ty.is_integer(),
    };
    applies.then_some(*lowering)
}

/// Stage the Rust method `method` on runtime scalars: `args[0]` is the
/// receiver. A method that is not a core operation of the receiver's type, or
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
