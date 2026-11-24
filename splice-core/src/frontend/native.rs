//! Type-directed linking of scalar Rust functions. Rust resolves the function
//! and its argument types before the specialization factory runs.
use super::{
    KernelBuilder,
    scalar::{ImportWord, RunArg, invoke_import},
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

pub trait ScalarAdapter {
    fn supported(self) -> bool;
    fn invoke(self, a: i32, b: i32) -> i64;
}

macro_rules! adapter {
    (($a:ident, $b:ident), $($arg:ident = $word:ident),*) => {
        impl<R, P: ProjectResult<R>, $($arg: ImportWord),*> ScalarAdapter for &Candidate<fn($($arg),*) -> R, P> {
            fn supported(self) -> bool { true }
            #[allow(unused_variables)]
            fn invoke(self, $a: i32, $b: i32) -> i64 {
                invoke_import(|| self.1.project((self.0)($($arg::decode($word)),*)))
            }
        }
    }
}
adapter!((a, b),);
adapter!((a, b), A = a);
adapter!((a, b), A = a, B = b);

impl<F, P> ScalarAdapter for &&Candidate<F, P> {
    fn supported(self) -> bool {
        false
    }
    fn invoke(self, _: i32, _: i32) -> i64 {
        splice_target::abi::Trap::Abi.status()
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
