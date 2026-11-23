//! Compile portable Splice kernels with Lancy's native JIT.
mod debug_info;
mod lower;

use lancy::{CompileError, ScalarKernel};
pub use lower::lower;
use splice_core::{
    BufferArg, BuildOptions, CallError, Codegen, Executable, Function, SignatureError, Trap,
};

/// Compile Splice kernels through Lancy's SSA HIR and host backend.
pub struct Lancy;

impl Codegen for Lancy {
    type Output = LancyKernel;

    fn build_with_options(
        arity: usize,
        kinds: &[splice_core::BufferKind],
        body: splice_core::frontend::Body<'_>,
        options: BuildOptions,
    ) -> splice_core::Result<Self::Output> {
        if options.debug() {
            return Self::compile(&splice_core::frontend::materialize(
                arity, kinds, body, options,
            )?);
        }
        let _span = tracing::debug_span!(target: "splice_core::compile", "compile", backend = "lancy", input = "staged").entered();
        let (hir, source) = tracing::debug_span!(target: "splice_core::compile", "stage_lancy_hir")
            .in_scope(|| {
                lower::build_with_source(arity, kinds, body, options.target.vector_bytes())
            })?;
        LancyKernel::new(hir, &options.target, arity, kinds, source)
            .map_err(|e| splice_core::BuildError::backend("lancy", e))
    }

    fn compile_raw(function: &Function) -> splice_core::Result<Self::Output> {
        Self::compile_raw_with_options(function, function.options())
    }

    fn compile_raw_with_options(
        function: &Function,
        options: BuildOptions,
    ) -> splice_core::Result<Self::Output> {
        let _span = tracing::debug_span!(target: "splice_core::compile", "compile", backend = "lancy", input = "portable").entered();
        let target = options.target;
        if options.debug() {
            let mut sources = Vec::new();
            let hir = lower::lower_debug(function, &target, &mut sources)?;
            let code =
                ScalarKernel::compile(&hir, &lancy::CompileOptions::new(target).with_debug())
                    .map_err(|e| splice_core::BuildError::backend("lancy", e))?;
            let registration = debug_info::publish(&code, &sources)?;
            return Ok(LancyKernel {
                _debug: Some(registration),
                code,
                arity: function.arity(),
                slices: function.buffer_kinds().into(),
            });
        }
        let hir = tracing::debug_span!(target: "splice_core::compile", "lower_lancy_hir")
            .in_scope(|| lower(function, &target))?;
        LancyKernel::new(
            hir,
            &target,
            function.arity(),
            function.buffer_kinds(),
            function.source(),
        )
        .map_err(|e| splice_core::BuildError::backend("lancy", e))
    }
}

pub struct LancyKernel {
    _debug: Option<splice_jit::Registration>,
    code: ScalarKernel,
    arity: usize,
    slices: Box<[splice_core::BufferKind]>,
}

impl LancyKernel {
    fn new(
        function: lancy::hir::Function,
        target: &splice_core::Target,
        arity: usize,
        kinds: &[splice_core::BufferKind],
        source: Option<&splice_core::frontend::KernelSource>,
    ) -> Result<Self, CompileError> {
        let code = ScalarKernel::compile(&function, &lancy::CompileOptions::new(*target))?;
        if splice_jit::perf_enabled() {
            splice_jit::profile(&[splice_jit::Function {
                name: debug_info::name(source),
                address: code.code_address() as u64,
                size: code.code_size() as u64,
                statements: Vec::new(),
            }]);
        }
        Ok(Self {
            _debug: None,
            code,
            arity,
            slices: kinds.into(),
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
