//! Type-directed linking of scalar Rust functions. Rust resolves the function
//! and its argument types before the specialization factory runs.
use super::{
    KernelBuilder,
    scalar::{ImportWord, RunArg, invoke_import},
};
use crate::hir::Reg;

pub struct Candidate<F>(pub F);

pub trait ScalarAdapter {
    fn supported(self) -> bool;
    fn invoke(self, a: i32, b: i32) -> i64;
}

macro_rules! adapter {
    (($a:ident, $b:ident), $($arg:ident = $word:ident),*) => {
        impl<R: ImportWord, $($arg: ImportWord),*> ScalarAdapter for &Candidate<fn($($arg),*) -> R> {
            fn supported(self) -> bool { true }
            #[allow(unused_variables)]
            fn invoke(self, $a: i32, $b: i32) -> i64 {
                invoke_import(|| (self.0)($($arg::decode($word)),*))
            }
        }
    }
}
adapter!((a, b),);
adapter!((a, b), A = a);
adapter!((a, b), A = a, B = b);

impl<F> ScalarAdapter for &&Candidate<F> {
    fn supported(self) -> bool {
        false
    }
    fn invoke(self, _: i32, _: i32) -> i64 {
        5_i64 << 32
    }
}

pub struct Operand<T>(pub T);
pub trait Materialize {
    fn materialize(self, builder: &mut dyn KernelBuilder) -> crate::Result<Reg>;
}
impl<T: RunArg + Copy> Materialize for &Operand<T> {
    fn materialize(self, builder: &mut dyn KernelBuilder) -> crate::Result<Reg> {
        Ok(self.0.reg(builder))
    }
}
impl<T> Materialize for &&Operand<T> {
    fn materialize(self, _: &mut dyn KernelBuilder) -> crate::Result<Reg> {
        Err("argument cannot be materialized as a scalar Rust operand".into())
    }
}
