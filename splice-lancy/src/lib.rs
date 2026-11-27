//! Compile portable Splice kernels with Lancy's native JIT.
mod debug_info;
mod lower;

use lancy::Kernel;
pub use lower::lower;
use splice_core::call::{Interface, SliceDescriptor};
use splice_core::frontend::{Body, Signature};
use splice_core::{Backend, BuildOptions, CallError, Executable, Function, Trap};

/// Compile Splice kernels through Lancy's SSA HIR and host backend.
pub struct Lancy;

impl Backend for Lancy {
    type Output = LancyKernel;

    fn supports_intrinsic(
        intrinsic: &splice_core::intrinsic::Intrinsic<'_>,
        _target: &splice_core::Target,
    ) -> bool {
        lower::native_intrinsic(intrinsic).is_some()
    }

    fn stage(
        signature: Signature<'_>,
        body: Body<'_>,
        options: BuildOptions,
    ) -> splice_core::Result<Self::Output> {
        let _span = tracing::debug_span!(target: "splice_core::compile", "compile", backend = "lancy", input = "staged").entered();
        LancyKernel::compile(signature, options, |vector_bytes, debug| {
            tracing::debug_span!(target: "splice_core::compile", "stage_lancy_hir").in_scope(|| {
                lower::stage(
                    signature,
                    body,
                    vector_bytes,
                    options.unsupported_intrinsic,
                    debug,
                )
            })
        })
    }

    fn compile_function(
        function: &Function,
        options: BuildOptions,
    ) -> splice_core::Result<Self::Output> {
        let _span = tracing::debug_span!(target: "splice_core::compile", "compile", backend = "lancy", input = "portable").entered();
        LancyKernel::compile(function.signature(), options, |vector_bytes, debug| {
            let graph = tracing::debug_span!(target: "splice_core::compile", "lower_lancy_hir")
                .in_scope(|| {
                    lower::lower_with(function, vector_bytes, options.unsupported_intrinsic, debug)
                })?;
            Ok((graph, function.source()))
        })
    }
}

pub struct LancyKernel {
    code: splice_jit::Published<Kernel>,
    interface: Interface,
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
        let (hir, source) = lower(target.vector_bytes(), debug.then_some(&mut sources))?;
        let mut compile_options = lancy::CompileOptions::new(target);
        if debug {
            compile_options = compile_options.with_debug();
        }
        let code = Kernel::compile(&hir, &compile_options)
            .map_err(|e| splice_core::BuildError::backend("lancy", e))?;
        let mut publish = splice_jit::Publish::new(debug);
        if debug {
            debug_info::describe(&mut publish, &code, &sources);
        } else {
            publish.function(
                source.map_or("kernel", |source| source.name),
                code.code_address() as u64,
                code.code_size() as u64,
            );
        }
        Ok(Self {
            code: publish.finish(true, code)?,
            interface: Interface::new(signature),
        })
    }
}

/// Words of a call whose arguments Lancy receives on the stack.
const INLINE_WORDS: usize = 32;

// SAFETY: `enter` runs the code compiled for `interface`, and Lancy checks
// the calling thread's target itself.
unsafe impl Executable for LancyKernel {
    fn interface(&self) -> &Interface {
        &self.interface
    }

    unsafe fn enter(&self, words: &[u64], slices: &[SliceDescriptor]) -> Result<u64, CallError> {
        let mut result = [0];
        if slices.is_empty() {
            // SAFETY: as below, without slices.
            unsafe { self.code.memory().call_packed_unchecked(words, &mut result) }
                .map_err(call_error)?;
            return Ok(result[0]);
        }
        // Lancy takes the scalars followed by each slice's address and length.
        let count = words.len() + 2 * slices.len();
        let mut inline = [0; INLINE_WORDS];
        let mut heap = Vec::new();
        let packed = if count <= INLINE_WORDS {
            &mut inline[..count]
        } else {
            heap.resize(count, 0);
            &mut heap[..]
        };
        let (scalars, pairs) = packed.split_at_mut(words.len());
        scalars.copy_from_slice(words);
        for (pair, slice) in pairs.as_chunks_mut::<2>().0.iter_mut().zip(slices) {
            *pair = [slice.data as u64, slice.len];
        }
        // SAFETY: the caller's contract: the descriptors supply live slices
        // and exclusive Rust contexts. Both Splice lowering paths check
        // permissions and use native call scratch only for initialized,
        // shape-checked packets.
        unsafe {
            self.code
                .memory()
                .call_packed_unchecked(packed, &mut result)
        }
        .map_err(call_error)?;
        Ok(result[0])
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
