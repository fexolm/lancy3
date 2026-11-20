//! Compile portable Splice kernels with Lancy's native JIT.
mod lower;

use lancy::{CompileError, ScalarKernel};
pub use lower::lower;
use splice::{BufferArg, Codegen, Executable, Function};

/// Compile Splice kernels through Lancy's SSA HIR and host backend.
pub struct Lancy;

impl Codegen for Lancy {
    type Output = LancyKernel;

    fn build(
        arity: usize,
        kinds: &[splice::BufferKind],
        body: splice::frontend::Body<'_>,
    ) -> splice::Result<Self::Output> {
        LancyKernel::new(&lower::build(arity, kinds, body)?, arity, kinds)
            .map_err(|e| e.to_string())
    }

    fn compile(function: &Function) -> splice::Result<Self::Output> {
        LancyKernel::new(&lower(function)?, function.arity(), function.buffer_kinds())
            .map_err(|e| e.to_string())
    }
}

pub struct LancyKernel {
    code: ScalarKernel,
    arity: usize,
    slices: Box<[splice::BufferKind]>,
}

impl LancyKernel {
    fn new(
        function: &lancy::hir::Function,
        arity: usize,
        kinds: &[splice::BufferKind],
    ) -> Result<Self, CompileError> {
        Ok(Self {
            code: ScalarKernel::compile(function)?,
            arity,
            slices: kinds.into(),
        })
    }
}

impl Executable for LancyKernel {
    fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32, String> {
        if args.len() != self.arity {
            return Err(format!(
                "expected {} scalar arguments, got {}",
                self.arity,
                args.len()
            ));
        }
        splice::hir::validate_buffers(&self.slices, buffers)?;
        if self.slices.is_empty() {
            return self.code.call(args).map_err(|e| e.to_string());
        }
        let mut words = Vec::with_capacity(self.arity + 2 * buffers.len());
        words.extend(args.iter().map(|&v| v as u32 as u64));
        for slice in buffers {
            words.extend([slice.data() as u64, slice.len() as u64]);
        }
        // SAFETY: both Splice lowering paths guard every access using the length
        // paired with its original slice, and validate permissions on every store and call.
        // BufferArg's Rust borrows keep writable slices disjoint from every other slice.
        unsafe { self.code.call_raw(&words) }
            .map(|v| v as i32)
            .map_err(|e| e.to_string())
    }
}
