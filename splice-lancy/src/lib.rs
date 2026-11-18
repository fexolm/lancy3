//! Compile portable Splice kernels with the Lancy x64 JIT.
mod lower;

use lancy::ScalarKernel;
pub use lower::lower;
use splice::{BufferArg, Codegen, Executable, Function};

/// Compile Splice kernels through Lancy's SSA HIR and x64 backend.
pub struct Lancy;

impl Codegen for Lancy {
    type Output = LancyKernel;

    fn build(
        arity: usize,
        kinds: &[splice::BufferKind],
        body: splice::frontend::Body<'_>,
    ) -> splice::Result<Self::Output> {
        ScalarKernel::compile(&lower::build(arity, kinds, body)?).map(LancyKernel)
    }

    fn compile(function: &Function) -> splice::Result<Self::Output> {
        ScalarKernel::compile(&lower(function)?).map(LancyKernel)
    }
}

pub struct LancyKernel(ScalarKernel);

impl Executable for LancyKernel {
    fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32, String> {
        let mut views: Vec<_> = buffers
            .iter_mut()
            .map(|buffer| match buffer {
                BufferArg::Read(data) => lancy::BufferArg::Read(data),
                BufferArg::Write(data) => lancy::BufferArg::Write(data),
            })
            .collect();
        self.0.call_with_buffers(args, &mut views)
    }
}
