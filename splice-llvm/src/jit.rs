use crate::{Entry, build_module};
use inkwell::context::Context;
use inkwell::llvm_sys::error::{
    LLVMConsumeError, LLVMDisposeErrorMessage, LLVMErrorRef, LLVMGetErrorMessage,
};
use inkwell::llvm_sys::orc2::{
    LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess, LLVMOrcJITDylibAddGenerator,
    lljit::{
        LLVMOrcCreateLLJIT, LLVMOrcDisposeLLJIT, LLVMOrcLLJITAddObjectFile,
        LLVMOrcLLJITGetGlobalPrefix, LLVMOrcLLJITGetMainJITDylib, LLVMOrcLLJITLookup,
        LLVMOrcLLJITRef,
    },
};
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{FileType, TargetMachine};
use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::ptr::null_mut;

pub(crate) struct Jit(LLVMOrcLLJITRef);

// SAFETY: Each compilation gets its own ORC session. After lookup, calls
// only touch immutable executable code, and dropping its owner requires
// exclusive access.
unsafe impl Send for Jit {}
// SAFETY: a shared `Jit` is only used to run its immutable code.
unsafe impl Sync for Jit {}

impl Drop for Jit {
    fn drop(&mut self) {
        // SAFETY: `self.0` is the live LLJIT this value owns; no code of it
        // runs once its owner is dropped.
        unsafe {
            let error = LLVMOrcDisposeLLJIT(self.0);
            if !error.is_null() {
                LLVMConsumeError(error);
            }
        }
    }
}

impl Jit {
    pub fn compile(
        auto_vectorize: bool,
        options: splice_core::BuildOptions,
        populate: impl for<'ctx> FnOnce(
            &'ctx Context,
            &Module<'ctx>,
            crate::target::Machine,
        ) -> splice_core::Result<crate::lower::Staged>,
    ) -> splice_core::Result<(splice_jit::Published<Self>, Entry)> {
        options.target.check_execution()?;
        options.target.check_native_convention()?;
        let debug = options.debug();
        let machine = crate::target::Machine {
            backend: if auto_vectorize {
                "Llvm"
            } else {
                "LlvmManualSimd"
            },
            ..crate::target::Machine::new(&options.target)
        };
        let target = tracing::debug_span!(target: "splice_core::compile", "llvm_target")
            .in_scope(|| crate::target::machine(&options.target, debug))?;
        let context = Context::create();
        let mut staged = crate::lower::Staged::default();
        let module = build_module(&context, &target, machine, |context, module| {
            staged = populate(context, module, machine)?;
            Ok(())
        })?;
        if !debug {
            optimize(&module, &target, auto_vectorize)?;
        }
        let object = tracing::debug_span!(target: "splice_core::compile", "emit_llvm_object")
            .in_scope(|| target.write_to_memory_buffer(&module, FileType::Object))
            .map_err(crate::error)?;
        let info = if debug || splice_jit::perf_enabled() {
            Some(crate::debug_info::ObjectSymbols::read(&object, debug)?)
        } else {
            None
        };
        let jit = tracing::debug_span!(target: "splice_core::compile", "link_llvm_object")
            .in_scope(|| Self::link(object))?;
        let address = jit.lookup(c"splice_entry")?;
        let mut publish = splice_jit::Publish::new(debug);
        if let Some(info) = info {
            info.describe(&mut publish, &jit, &staged, debug)?;
        }
        // SAFETY: The lowerer declares the entry with the host C ABI. It
        // receives the checked argument arrays and ORC retains the code.
        let entry = unsafe { std::mem::transmute::<usize, Entry>(address as usize) };
        Ok((publish.finish(true, jit)?, entry))
    }

    /// Inkwell owns the module, optimizer and object generation. Its llvm-sys
    /// bindings expose the ORC object linker and executable mapping lifetime.
    ///
    /// Never enable ORC's GDB debug support or registration listener here:
    /// `splice-jit` owns the process-wide `__jit_debug_descriptor`, which is
    /// LLVM's own definition when LLVM is linked statically, and registers
    /// symbols under a lock that LLVM does not take.
    fn link(object: MemoryBuffer<'static>) -> splice_core::Result<Self> {
        // SAFETY: every ORC handle is created here and passed to LLVM by the
        // ownership rules of its C API: the object buffer and the generator
        // are consumed by the calls that take them.
        unsafe {
            let mut raw = null_mut();
            check(LLVMOrcCreateLLJIT(&mut raw, null_mut()))?;
            let jit = Self(raw);
            let dylib = LLVMOrcLLJITGetMainJITDylib(raw);
            let mut generator = null_mut();
            // LLVM may lower optimized loops to C runtime functions, e.g. memset.
            check(LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
                &mut generator,
                LLVMOrcLLJITGetGlobalPrefix(raw),
                None,
                null_mut(),
            ))?;
            LLVMOrcJITDylibAddGenerator(dylib, generator);
            // ORC consumes the memory buffer on success and failure.
            let object = ManuallyDrop::new(object);
            check(LLVMOrcLLJITAddObjectFile(raw, dylib, object.as_mut_ptr()))?;
            Ok(jit)
        }
    }

    pub(super) fn lookup(&self, symbol: &CStr) -> splice_core::Result<u64> {
        let mut address = 0;
        // SAFETY: self owns the live ORC session and address points to a u64
        // that LLVM fills synchronously.
        unsafe {
            check(LLVMOrcLLJITLookup(self.0, &mut address, symbol.as_ptr()))?;
        }
        if address == 0 {
            return Err(crate::error(format!(
                "LLVM returned a null address for {}",
                symbol.to_string_lossy()
            )));
        }
        Ok(address)
    }
}

/// LLVM 19's `default<O3>` for this architecture, as
/// `opt -mtriple=<arch>-unknown-linux-gnu -passes='default<O3>' -print-pipeline-passes`
/// prints it. `LLVMRunPasses` ignores the vectorization fields of
/// `PassBuilderOptions` for `default<O3>`, so manual SIMD spells it out.
#[cfg(target_arch = "x86_64")]
const O3: &str = include_str!("pipeline/o3-x86_64.txt");
#[cfg(target_arch = "aarch64")]
const O3: &str = include_str!("pipeline/o3-aarch64.txt");

/// O3 with the loop vectorizer limited to loops that request it (Splice
/// emits none) and without the SLP vectorizer. Every other pass and the
/// target's code generation, including explicit vector and floating-point
/// operations, are those of `default<O3>`.
fn manual_simd_pipeline() -> &'static str {
    static PIPELINE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        const LOOP: &str = "loop-vectorize<no-interleave-forced-only;no-vectorize-forced-only;>";
        let o3 = O3.trim();
        assert!(o3.contains(LOOP) && o3.contains(",slp-vectorizer,"));
        o3.replace(
            LOOP,
            "loop-vectorize<interleave-forced-only;vectorize-forced-only;>",
        )
        .replace(",slp-vectorizer,", ",")
    });
    &PIPELINE
}

/// Both lowering paths finish defining all private callees before the module inliner runs.
pub(crate) fn optimize(
    module: &Module<'_>,
    target: &TargetMachine,
    auto_vectorize: bool,
) -> splice_core::Result<()> {
    let _span =
        tracing::debug_span!(target: "splice_core::compile", "optimize_llvm_ir", auto_vectorize)
            .entered();
    let pipeline = if auto_vectorize {
        "default<O3>"
    } else {
        manual_simd_pipeline()
    };
    module
        .run_passes(pipeline, target, PassBuilderOptions::create())
        .map_err(crate::error)
}

unsafe fn check(error: LLVMErrorRef) -> splice_core::Result<()> {
    if error.is_null() {
        return Ok(());
    }
    // SAFETY: `error` is a live error the caller owns; reading its message
    // consumes it, and the message is disposed after it is copied.
    unsafe {
        let message = LLVMGetErrorMessage(error); // consumes error
        let reason = CStr::from_ptr(message).to_string_lossy().into_owned();
        LLVMDisposeErrorMessage(message);
        Err(crate::error(reason))
    }
}
