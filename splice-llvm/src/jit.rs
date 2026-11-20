use crate::target::host_machine;
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
        populate: impl for<'ctx> FnOnce(&'ctx Context, &Module<'ctx>) -> Result<(), String>,
    ) -> Result<(Self, Entry), String> {
        let target = tracing::debug_span!(target: "splice::compile", "llvm_target")
            .in_scope(host_machine)?;
        let context = Context::create();
        let module = build_module(&context, &target, populate)?;
        optimize(&module, &target, auto_vectorize)?;
        let object = tracing::debug_span!(target: "splice::compile", "emit_llvm_object")
            .in_scope(|| target.write_to_memory_buffer(&module, FileType::Object))
            .map_err(|error| error.to_string())?;
        tracing::debug_span!(target: "splice::compile", "link_llvm_object")
            .in_scope(|| Self::link(object))
    }

    /// Inkwell owns the module, optimizer and object generation. Its llvm-sys
    /// bindings expose the ORC object linker and executable mapping lifetime.
    fn link(object: MemoryBuffer<'static>) -> Result<(Self, Entry), String> {
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
            let mut address = 0;
            check(LLVMOrcLLJITLookup(
                raw,
                &mut address,
                c"splice_entry".as_ptr(),
            ))?;
            if address == 0 {
                return Err("LLVM returned a null kernel entry".into());
            }
            // The lowerer declares the entry with this C ABI. ORC owns its
            // machine code until all outstanding borrows of Jit end.
            let entry = std::mem::transmute::<usize, Entry>(address as usize);
            Ok((jit, entry))
        }
    }
}

/// Both lowering paths finish defining all private callees before the module inliner runs.
pub(crate) fn optimize(
    module: &Module<'_>,
    target: &TargetMachine,
    auto_vectorize: bool,
) -> Result<(), String> {
    let _span = tracing::debug_span!(target: "splice::compile", "optimize_llvm_ir", auto_vectorize)
        .entered();
    let run = || {
        module
            .run_passes("default<O3>", target, PassBuilderOptions::create())
            .map_err(|error| error.to_string())
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

unsafe fn check(error: LLVMErrorRef) -> Result<(), String> {
    if error.is_null() {
        return Ok(());
    }
    unsafe {
        let message = LLVMGetErrorMessage(error); // consumes error
        let reason = CStr::from_ptr(message).to_string_lossy().into_owned();
        LLVMDisposeErrorMessage(message);
        Err(reason)
    }
}
