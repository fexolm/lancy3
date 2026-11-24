//! Compile portable Splice kernels with Lancy's native JIT.
mod debug_info;
mod lower;

use lancy::ScalarKernel;
pub use lower::lower;
use splice_core::frontend::{Body, Signature};
use splice_core::{
    Backend, BufferArg, BuildOptions, CallError, Executable, Function, SignatureError, Trap,
};

/// Compile Splice kernels through Lancy's SSA HIR and host backend.
pub struct Lancy;

impl Backend for Lancy {
    type Output = LancyKernel;

    fn stage(
        signature: Signature<'_>,
        body: Body<'_>,
        options: BuildOptions,
    ) -> splice_core::Result<Self::Output> {
        let _span = tracing::debug_span!(target: "splice_core::compile", "compile", backend = "lancy", input = "staged").entered();
        LancyKernel::compile(signature, options, |vector_bytes, debug| {
            lower::stage(signature, body, vector_bytes, debug)
        })
    }

    fn compile_function(
        function: &Function,
        options: BuildOptions,
    ) -> splice_core::Result<Self::Output> {
        let _span = tracing::debug_span!(target: "splice_core::compile", "compile", backend = "lancy", input = "portable").entered();
        LancyKernel::compile(function.signature(), options, |vector_bytes, debug| {
            let graph = lower::lower_with(function, vector_bytes, debug)?;
            Ok((graph, function.source()))
        })
    }
}

pub struct LancyKernel {
    _debug: Option<splice_jit::Registration>,
    code: ScalarKernel,
    arity: usize,
    slices: Box<[splice_core::BufferKind]>,
}

impl LancyKernel {
    /// Lower with `lower`, which records the source points of a debug build.
    fn compile(
        signature: Signature<'_>,
        options: BuildOptions,
        lower: impl FnOnce(
            usize,
            Option<&mut Vec<lower::DebugSource>>,
        ) -> splice_core::Result<(
            lancy::hir::Function,
            Option<&'static splice_core::frontend::KernelSource>,
        )>,
    ) -> splice_core::Result<Self> {
        let target = options.target;
        let mut sources = Vec::new();
        let debug = options.debug();
        let (hir, source) = tracing::debug_span!(target: "splice_core::compile", "lower_lancy_hir")
            .in_scope(|| lower(target.vector_bytes(), debug.then_some(&mut sources)))?;
        let mut compile_options = lancy::CompileOptions::new(target);
        if debug {
            compile_options = compile_options.with_debug();
        }
        let code = ScalarKernel::compile(&hir, &compile_options)
            .map_err(|e| splice_core::BuildError::backend("lancy", e))?;
        let registration = if debug {
            Some(debug_info::publish(&code, &sources)?)
        } else {
            if splice_jit::perf_enabled() {
                splice_jit::profile(&[splice_jit::Function {
                    name: debug_info::name(source),
                    address: code.code_address() as u64,
                    size: code.code_size() as u64,
                    statements: Vec::new(),
                }]);
            }
            None
        };
        Ok(Self {
            _debug: registration,
            code,
            arity: signature.arity,
            slices: signature.buffers.into(),
        })
    }
}

impl Executable for LancyKernel {
    fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32, CallError> {
        if args.len() != self.arity {
            return Err(SignatureError::Scalars {
                expected: self.arity,
                actual: args.len(),
            }
            .into());
        }
        splice_core::hir::validate_buffers(&self.slices, buffers)?;
        if self.slices.is_empty() && !self.code.requires_raw_call() {
            return self.code.call(args).map_err(call_error);
        }
        let mut words = Vec::with_capacity(self.arity + 2 * buffers.len());
        words.extend(args.iter().map(|&v| v as u32 as u64));
        for slice in buffers {
            words.extend([slice.data() as u64, slice.len() as u64]);
        }
        // SAFETY: The validated descriptors supply live slices and exclusive
        // Rust contexts. Both Splice lowering paths check permissions and use
        // native call scratch only for initialized, shape-checked packets.
        unsafe { self.code.call_raw(&words) }
            .map(|v| v as i32)
            .map_err(call_error)
    }
}

fn call_error(error: lancy::CallError) -> CallError {
    match error {
        lancy::CallError::Trap(trap) => trap.into(),
        lancy::CallError::Target(error) => error.into(),
        // The arguments were checked against the Splice signature above, so
        // a Lancy signature mismatch is a defect at the native boundary.
        _ => Trap::Abi.into(),
    }
}
