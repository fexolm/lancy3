//! Execute a scalar function across independent inputs using the available native backend.
use super::{NativeKernel, ScalarKernel};
use crate::hir::{Function, vectorize::vector_program};

pub struct MappedKernel {
    arity: usize,
    code: MapCode,
}

enum MapCode {
    Vector(NativeKernel),
    Scalar(ScalarKernel),
}

impl MappedKernel {
    pub fn compile(f: &Function) -> Result<Self, String> {
        if !f.buffer_kinds().is_empty() {
            return Err(
                "map expects independent scalar inputs; use ScalarKernel for slices".into(),
            );
        }
        let f = f.simplified()?;
        let code = match vector_program(&f) {
            Some(program) => MapCode::Vector(NativeKernel::specialize(&program, &[])?),
            None => MapCode::Scalar(ScalarKernel::baseline(&f)?),
        };
        Ok(Self {
            arity: f.arity(),
            code,
        })
    }

    pub fn is_vectorized(&self) -> bool {
        matches!(&self.code, MapCode::Vector(_))
    }

    pub fn run(&self, columns: &[&[i32]], out: &mut [i32]) -> Result<(), String> {
        if columns.len() != self.arity || columns.iter().any(|c| c.len() != out.len()) {
            return Err("map expects one equally sized column per scalar argument".into());
        }
        match &self.code {
            MapCode::Vector(kernel) => {
                if kernel.run(columns, out)? != out.len() {
                    return Err("unguarded map exited early".into());
                }
            }
            MapCode::Scalar(kernel) => {
                let mut caller = kernel.caller();
                let mut args = vec![0; self.arity];
                for (row, dst) in out.iter_mut().enumerate() {
                    for (arg, col) in args.iter_mut().zip(columns) {
                        *arg = col[row];
                    }
                    *dst = caller.call(&args)?;
                }
            }
        }
        Ok(())
    }
}
