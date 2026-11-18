//! Compile portable Splice kernels with the Lancy x64 JIT.
mod direct;
mod lower;

use lancy::ScalarKernel;
pub use lower::lower;
use splice::{BufferArg, Codegen, Executable, Function};

/// The optimizing Lancy backend.
pub struct Lancy;

impl Codegen for Lancy {
    type Output = Result<LancyKernel, String>;

    fn build(
        arity: usize,
        kinds: &[splice::BufferKind],
        body: splice::frontend::Body<'_>,
    ) -> Self::Output {
        direct::compile(arity, kinds, body, true).map(LancyKernel)
    }

    fn compile(function: &Function) -> Self::Output {
        ScalarKernel::compile_hir(&lower(function)?).map(LancyKernel)
    }
}

/// The low-latency Lancy tier, available through the same kernel factory.
pub struct LancyBaseline;

impl Codegen for LancyBaseline {
    type Output = Result<LancyKernel, String>;

    fn build(
        arity: usize,
        kinds: &[splice::BufferKind],
        body: splice::frontend::Body<'_>,
    ) -> Self::Output {
        direct::compile(arity, kinds, body, false).map(LancyKernel)
    }

    fn compile(function: &Function) -> Self::Output {
        ScalarKernel::baseline_hir(&lower(function)?).map(LancyKernel)
    }
}

pub struct LancyKernel(ScalarKernel);

impl Executable for LancyKernel {
    fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32, String> {
        call_native(&self.0, args, buffers)
    }
}

/// Invoke an existing Lancy version (including an adaptive kernel's current
/// version) using borrowed Splice arguments. Only slice descriptors are adapted.
pub fn call_native(
    code: &ScalarKernel,
    args: &[i32],
    buffers: &mut [BufferArg<'_>],
) -> Result<i32, String> {
    let count = buffers.len();
    let views = buffers.iter_mut().map(|buffer| match buffer {
        BufferArg::Read(data) => lancy::BufferArg::Read(data),
        BufferArg::Write(data) => lancy::BufferArg::Write(data),
    });
    if count == 0 {
        return code.call(args);
    }
    if count <= 8 {
        let mut local: [_; 8] = std::array::from_fn(|_| lancy::BufferArg::Read(&[]));
        for (slot, view) in local.iter_mut().zip(views) {
            *slot = view;
        }
        code.call_with_buffers(args, &mut local[..count])
    } else {
        code.call_with_buffers(args, &mut views.collect::<Vec<_>>())
    }
}
