//! Compile portable Splice kernels with the Lancy x64 JIT.
mod batch;
mod lower;

use lancy::ScalarKernel;
pub use lower::lower;
use splice::{BufferArg, Codegen, Executable, Function};

/// The optimizing Lancy backend.
pub struct Lancy;

impl Codegen for Lancy {
    type Kernel = LancyKernel;

    fn compile(function: &Function) -> Result<LancyKernel, String> {
        if let Some(row) = function.batch_row()
            && let Some(projection) = lower(row)?.vectorize()?
        {
            let kernel = batch::Batch::compile(&projection, function.buffer_kinds())?;
            return Ok(LancyKernel(Code::Batch(kernel)));
        }
        ScalarKernel::compile(&lower(function)?).map(LancyKernel::from)
    }
}

/// The low-latency Lancy tier, available through the same kernel factory.
pub struct LancyBaseline;

impl Codegen for LancyBaseline {
    type Kernel = LancyKernel;

    fn compile(function: &Function) -> Result<LancyKernel, String> {
        ScalarKernel::baseline(&lower(function)?).map(LancyKernel::from)
    }
}

pub struct LancyKernel(Code);

enum Code {
    Scalar(ScalarKernel),
    Batch(batch::Batch),
}

impl LancyKernel {
    /// Whether the independent batch rows were compiled into a SIMD loop.
    pub fn is_vectorized(&self) -> bool {
        matches!(self.0, Code::Batch(_))
    }
}

impl From<ScalarKernel> for LancyKernel {
    fn from(code: ScalarKernel) -> Self {
        Self(Code::Scalar(code))
    }
}

impl Executable for LancyKernel {
    fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32, String> {
        match &self.0 {
            Code::Scalar(code) => call_native(code, args, buffers),
            Code::Batch(batch) => batch.call(args, buffers),
        }
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
