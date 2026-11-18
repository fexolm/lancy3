use crate::{Entry, build_module};
use inkwell::OptimizationLevel;
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
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::ptr::null_mut;
use std::sync::Once;

pub(crate) fn host_machine() -> Result<TargetMachine, String> {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        Target::initialize_x86(&InitializationConfig {
            asm_parser: false,
            disassembler: false,
            ..Default::default()
        })
    });
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let cpu = TargetMachine::get_host_cpu_name();
    let features = TargetMachine::get_host_cpu_features();
    target
        .create_target_machine(
            &triple,
            cpu.to_str().map_err(|e| e.to_string())?,
            features.to_str().map_err(|e| e.to_string())?,
            OptimizationLevel::Aggressive,
            RelocMode::PIC,
            CodeModel::JITDefault,
        )
        .ok_or_else(|| "LLVM failed to create the host target machine".into())
}

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
        populate: impl for<'ctx> FnOnce(&'ctx Context, &Module<'ctx>) -> Result<(), String>,
    ) -> Result<(Self, Entry), String> {
        let target = host_machine()?;
        let context = Context::create();
        let module = build_module(&context, &target, populate)?;
        module
            .run_passes("default<O3>", &target, PassBuilderOptions::create())
            .map_err(|error| error.to_string())?;
        let object = target
            .write_to_memory_buffer(&module, FileType::Object)
            .map_err(|error| error.to_string())?;
        Self::link(object)
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
