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

    fn build(
        arity: usize,
        kinds: &[splice_core::BufferKind],
        body: splice_core::frontend::Body<'_>,
    ) -> splice_core::Result<Self::Output> {
        Self::build_with_options(arity, kinds, body, BuildOptions::default())
    }

    fn build_with_options(
        arity: usize,
        kinds: &[splice_core::BufferKind],
        body: splice_core::frontend::Body<'_>,
        options: BuildOptions,
    ) -> splice_core::Result<Self::Output> {
        if options == BuildOptions::Debug {
            return Self::compile(&splice_core::frontend::materialize_with_options(
                arity, kinds, body, options,
            )?);
        }
        let _span = tracing::debug_span!(target: "splice_core::compile", "compile", backend = "lancy", input = "staged").entered();
        let (hir, source) = tracing::debug_span!(target: "splice_core::compile", "stage_lancy_hir")
            .in_scope(|| lower::build_with_source(arity, kinds, body))?;
        LancyKernel::new(hir, arity, kinds, source).map_err(|e| e.to_string())
    }

    fn compile_raw(function: &Function) -> splice_core::Result<Self::Output> {
        Self::compile_with_options(function, function.options())
    }

    fn compile_raw_with_options(
        function: &Function,
        options: BuildOptions,
    ) -> splice_core::Result<Self::Output> {
        let _span = tracing::debug_span!(target: "splice_core::compile", "compile", backend = "lancy", input = "portable").entered();
        if options == BuildOptions::Debug {
            let mut sources = Vec::new();
            let hir = lower::lower_debug(function, &mut sources)?;
            let code = ScalarKernel::compile_debug(&hir).map_err(|e| e.to_string())?;
            let registration = debug_info::publish(&code, &sources)?;
            return Ok(LancyKernel {
                _debug: Some(registration),
                code,
                arity: function.arity(),
                slices: function.buffer_kinds().into(),
            });
        }
        let hir = tracing::debug_span!(target: "splice_core::compile", "lower_lancy_hir")
            .in_scope(|| lower(function))?;
        LancyKernel::new(
            hir,
            function.arity(),
            function.buffer_kinds(),
            function.source(),
        )
        .map_err(|e| e.to_string())
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
        arity: usize,
        kinds: &[splice_core::BufferKind],
        source: Option<&splice_core::frontend::KernelSource>,
    ) -> Result<Self, CompileError> {
        let code = ScalarKernel::compile(&function)?;
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
