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
use splice_target::Profile;
use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::ptr::null_mut;
use std::sync::Once;

pub(crate) fn host_machine() -> Result<TargetMachine, String> {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let config = InitializationConfig {
            asm_parser: false,
            disassembler: false,
            ..Default::default()
        };
        #[cfg(target_arch = "x86_64")]
        Target::initialize_x86(&config);
        #[cfg(target_arch = "aarch64")]
        Target::initialize_aarch64(&config);
    });
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let simd = splice_target::simd()?;
    // Explicit profiles start from a baseline CPU: an inferred CPU model
    // could re-enable extensions that the selected profile excludes.
    let (cpu, features) = match simd.profile {
        Profile::Auto => (
            TargetMachine::get_host_cpu_name()
                .to_string_lossy()
                .into_owned(),
            TargetMachine::get_host_cpu_features()
                .to_string_lossy()
                .into_owned(),
        ),
        profile => {
            let features = match profile {
                Profile::Sse => "+sse2",
                Profile::Avx2 => "+avx2",
                Profile::Avx512 => "+avx512f",
                Profile::Avx10 => match simd.bytes {
                    32 => "+avx10.1-256",
                    64 => "+avx10.1-512",
                    _ => return Err("LLVM requires at least 256-bit AVX10 support".into()),
                },
                Profile::Neon => "+neon",
                Profile::Sve => "+neon,+sve",
                Profile::Sve2 => "+neon,+sve,+sve2",
                Profile::Auto => unreachable!(),
            };
            let cpu = if cfg!(target_arch = "x86_64") {
                "x86-64"
            } else {
                "generic"
            };
            (cpu.to_owned(), features.to_owned())
        }
    };
    target
        .create_target_machine(
            &triple,
            &cpu,
            &features,
            OptimizationLevel::Aggressive,
            RelocMode::PIC,
            CodeModel::JITDefault,
        )
        .ok_or_else(|| "LLVM failed to create the host target machine".into())
}

/// Keep target-specific intrinsics in sync with the actual codegen profile.
pub(crate) fn x86_compress(lanes: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let simd = splice_target::simd().expect("JIT validates the SIMD profile before lowering");
        match simd.profile {
            Profile::Auto => {
                is_x86_feature_detected!("avx512f")
                    && (lanes == 16 || is_x86_feature_detected!("avx512vl"))
            }
            Profile::Avx512 | Profile::Avx10 => true,
            _ => false,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = lanes;
        false
    }
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
        optimize(&module, &target)?;
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

/// Both lowering paths finish defining all private callees before the module inliner runs.
pub(crate) fn optimize(module: &Module<'_>, target: &TargetMachine) -> Result<(), String> {
    module
        .run_passes("default<O3>", target, PassBuilderOptions::create())
        .map_err(|error| error.to_string())
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
