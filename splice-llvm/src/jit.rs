use crate::{Entry, build_module};
use inkwell::attributes::{Attribute, AttributeLoc};
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

// Each compilation gets its own ORC session. After lookup, calls only touch
// immutable executable code, and dropping its owner requires exclusive access.
unsafe impl Send for Jit {}
unsafe impl Sync for Jit {}

impl Drop for Jit {
    fn drop(&mut self) {
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
        source: Option<&splice_core::Function>,
        options: splice_core::BuildOptions,
        populate: impl for<'ctx> FnOnce(
            &'ctx Context,
            &Module<'ctx>,
            crate::target::Machine,
        ) -> splice_core::Result<
            Option<&'static splice_core::frontend::KernelSource>,
        >,
    ) -> splice_core::Result<(Self, Entry, Option<splice_jit::Registration>)> {
        options.target.check_execution()?;
        options.target.check_native_convention()?;
        let debug = options.debug();
        let machine = crate::target::Machine::new(&options.target);
        let target = tracing::debug_span!(target: "splice_core::compile", "llvm_target")
            .in_scope(|| crate::target::machine(&options.target, debug))?;
        let context = Context::create();
        let mut kernel_source = None;
        let module = build_module(&context, &target, machine, |context, module| {
            kernel_source = populate(context, module, machine)?;
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
        let registration = info
            .map(|info| info.publish(&jit, source, kernel_source, machine.vector_bytes, debug))
            .transpose()?
            .flatten();
        // SAFETY: The lowerer declares the entry with the host C ABI. It
        // receives the checked argument arrays and ORC retains the code.
        let entry = unsafe { std::mem::transmute::<usize, Entry>(address as usize) };
        Ok((jit, entry, registration))
    }

    /// Inkwell owns the module, optimizer and object generation. Its llvm-sys
    /// bindings expose the ORC object linker and executable mapping lifetime.
    ///
    /// Never enable ORC's GDB debug support or registration listener here:
    /// `splice-jit` owns the process-wide `__jit_debug_descriptor`, which is
    /// LLVM's own definition when LLVM is linked statically, and registers
    /// symbols under a lock that LLVM does not take.
    fn link(object: MemoryBuffer<'static>) -> splice_core::Result<Self> {
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

/// Both lowering paths finish defining all private callees before the module inliner runs.
pub(crate) fn optimize(
    module: &Module<'_>,
    target: &TargetMachine,
    auto_vectorize: bool,
) -> splice_core::Result<()> {
    let _span =
        tracing::debug_span!(target: "splice_core::compile", "optimize_llvm_ir", auto_vectorize)
            .entered();
    let run = || {
        module
            .run_passes("default<O3>", target, PassBuilderOptions::create())
            .map_err(crate::error)
    };
    if auto_vectorize {
        return run();
    }

    // The default<O3> alias resets both vectorization options. Both passes
    // honor noimplicitfloat. Remove it before codegen so the target can
    // legalize and select explicit SIMD and floating-point operations normally.
    let kind = Attribute::get_named_enum_kind_id("noimplicitfloat");
    let attr = module.get_context().create_enum_attribute(kind, 0);
    for function in module.get_functions() {
        if !function.as_global_value().is_declaration() {
            function.add_attribute(AttributeLoc::Function, attr);
        }
    }
    let result = run();
    // O3 may inline and delete functions, so iterate over the surviving ones.
    for function in module.get_functions() {
        function.remove_enum_attribute(AttributeLoc::Function, kind);
    }
    result
}

unsafe fn check(error: LLVMErrorRef) -> splice_core::Result<()> {
    if error.is_null() {
        return Ok(());
    }
    unsafe {
        let message = LLVMGetErrorMessage(error); // consumes error
        let reason = CStr::from_ptr(message).to_string_lossy().into_owned();
        LLVMDisposeErrorMessage(message);
        Err(crate::error(reason))
    }
}
