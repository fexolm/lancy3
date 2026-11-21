//! Compile portable Splice kernels with Lancy's native JIT.
mod debug_info;
mod lower;

use lancy::{CompileError, ScalarKernel};
pub use lower::lower;
use splice::{BufferArg, BuildOptions, Codegen, Executable, Function};

/// Compile Splice kernels through Lancy's SSA HIR and host backend.
pub struct Lancy;

impl Codegen for Lancy {
    type Output = LancyKernel;

    fn build(
        arity: usize,
        kinds: &[splice::BufferKind],
        body: splice::frontend::Body<'_>,
    ) -> splice::Result<Self::Output> {
        Self::build_with_options(arity, kinds, body, BuildOptions::default())
    }

    fn build_with_options(
        arity: usize,
        kinds: &[splice::BufferKind],
        body: splice::frontend::Body<'_>,
        options: BuildOptions,
    ) -> splice::Result<Self::Output> {
        if options == BuildOptions::Debug {
            return Self::compile(&splice::frontend::materialize_with_options(
                arity, kinds, body, options,
            )?);
        }
        let _span = tracing::debug_span!(target: "splice::compile", "compile", backend = "lancy", input = "staged").entered();
        let (hir, source) = tracing::debug_span!(target: "splice::compile", "stage_lancy_hir")
            .in_scope(|| lower::build_with_source(arity, kinds, body))?;
        LancyKernel::new(hir, arity, kinds, source).map_err(|e| e.to_string())
    }

    fn compile(function: &Function) -> splice::Result<Self::Output> {
        Self::compile_with_options(function, function.options())
    }

    fn compile_with_options(
        function: &Function,
        options: BuildOptions,
    ) -> splice::Result<Self::Output> {
        let _span = tracing::debug_span!(target: "splice::compile", "compile", backend = "lancy", input = "portable").entered();
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
        let hir = tracing::debug_span!(target: "splice::compile", "lower_lancy_hir")
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
    slices: Box<[splice::BufferKind]>,
}

impl LancyKernel {
    fn new(
        function: lancy::hir::Function,
        arity: usize,
        kinds: &[splice::BufferKind],
        source: Option<&splice::frontend::KernelSource>,
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
        // SAFETY: BufferArg supplies initialized slices of the validated lengths
        // for this call. Its Rust borrows ensure exclusive writable bytes and
        // immutable shared bytes. Both lowering paths check store permissions.
        unsafe { self.code.call_raw(&words) }
            .map(|v| v as i32)
            .map_err(|e| e.to_string())
    }
}
