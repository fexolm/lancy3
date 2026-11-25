//! Type-directed linking of scalar Rust functions. Rust resolves the function
//! and its argument types before the specialization factory runs.
use super::{
    KernelBuilder,
    scalar::{ImportWord, RunArg},
};
use crate::hir::Reg;

pub struct Candidate<F, P>(pub F, pub P);

pub trait ProjectResult<R> {
    fn project(&self, result: R) -> i32;
}

pub struct EncodeResult;

impl<R: ImportWord> ProjectResult<R> for EncodeResult {
    fn project(&self, result: R) -> i32 {
        result.encode()
    }
}

pub struct MapResult<F>(pub F);

impl<R, F: Fn(R) -> i32> ProjectResult<R> for MapResult<F> {
    fn project(&self, result: R) -> i32 {
        (self.0)(result)
    }
}

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
        self.0.word(builder)
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
