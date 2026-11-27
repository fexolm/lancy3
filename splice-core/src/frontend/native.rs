//! Type-directed linking of scalar Rust functions. Rust resolves the function
//! and its argument types before the specialization factory runs.
use super::{KernelBuilder, scalar::RunArg};
use crate::hir::Reg;

/// A Rust function a kernel calls, typed by its signature.
pub struct Candidate<F>(pub F);

pub struct Operand<T>(pub T);
pub trait Materialize {
    fn known(self) -> bool;
    fn materialize(self, builder: &mut dyn KernelBuilder) -> crate::Result<Reg>;
}
impl<T: RunArg + Copy> Materialize for &Operand<T> {
    fn known(self) -> bool {
        self.0.known().is_some()
    }

    fn materialize(self, builder: &mut dyn KernelBuilder) -> crate::Result<Reg> {
        self.0.reg(builder)
    }
}
impl<T> Materialize for &&Operand<T> {
    fn known(self) -> bool {
        false
    }

    fn materialize(self, _: &mut dyn KernelBuilder) -> crate::Result<Reg> {
        Err(crate::BuildError::staging(
            "argument cannot be materialized as a scalar Rust operand",
        ))
    }
}
