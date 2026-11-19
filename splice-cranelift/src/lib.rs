//! Compile verified Splice kernels to native code with Cranelift.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(any(
    all(target_arch = "x86_64", any(windows, target_os = "linux")),
    all(target_arch = "aarch64", target_os = "linux")
)))]
compile_error!("splice-cranelift supports x86-64 Windows and Linux, and AArch64 Linux");

mod lower;
mod module;

use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Module, default_libcall_names};
use splice::{BufferArg, BufferKind, Codegen, Executable, Function, Result};
use std::sync::Mutex;

/// Compile a specialized kernel or a portable Splice function with Cranelift.
pub struct Cranelift;

fn builder() -> Result<JITBuilder> {
    use cranelift_codegen::settings::{self, Configurable};
    use splice_target::Profile;

    let simd = splice_target::simd()?;
    let mut flags = settings::builder();
    for (name, value) in [
        ("opt_level", "speed"),
        ("enable_probestack", "true"),
        ("probestack_strategy", "inline"),
        // Definitions may be farther apart than a direct AArch64 call.
        ("use_colocated_libcalls", "false"),
        ("is_pic", "false"),
    ] {
        flags.set(name, value).map_err(|e| e.to_string())?;
    }
    let auto = simd.profile == Profile::Auto;
    let isa = cranelift_native::builder_with_options(auto)?;
    #[cfg(target_arch = "x86_64")]
    let isa = {
        let mut isa = isa;
        let profile = simd.profile;
        // Cranelift has no AVX10 setting. Enable only instructions that it
        // models and the OS reports under their legacy feature names.
        for (enabled, flag) in [
            (is_x86_feature_detected!("ssse3"), "has_ssse3"),
            (is_x86_feature_detected!("sse4.1"), "has_sse41"),
            (is_x86_feature_detected!("sse4.2"), "has_sse42"),
            (
                profile != Profile::Sse && is_x86_feature_detected!("avx"),
                "has_avx",
            ),
            (
                profile != Profile::Sse && is_x86_feature_detected!("avx2"),
                "has_avx2",
            ),
            (profile == Profile::Avx512, "has_avx512f"),
        ] {
            if !auto && enabled {
                isa.enable(flag).map_err(|e| e.to_string())?;
            }
        }
        isa
    };
    let isa = isa
        .finish(settings::Flags::new(flags))
        .map_err(|e| e.to_string())?;
    Ok(JITBuilder::with_isa(isa, default_libcall_names()))
}

#[repr(C)]
struct Descriptor {
    data: *mut u8,
    len: i32,
}

// A successful i32 is zero-extended. Negative i64s can therefore carry
// faults through any number of calls without losing a valid i32 result.
const BOUNDS: i64 = -1;
const DIV_ZERO: i64 = -2;
const OVERFLOW: i64 = -3;
type Entry = unsafe extern "C" fn(*const i32, *const Descriptor) -> i64;

struct CodeMemory(Option<JITModule>);

impl Drop for CodeMemory {
    fn drop(&mut self) {
        if let Some(module) = self.0.take() {
            // SAFETY: No entry pointer escapes its kernel. Drop has exclusive
            // access after the last call, or compilation has failed.
            unsafe { module.free_memory() };
        }
    }
}

/// Owns the executable code of the entry point and all its callees.
pub struct CraneliftKernel {
    // The module is Send. Only its owner needs exclusive access on drop.
    // Concurrent calls use the immutable machine code without taking a lock.
    _memory: Mutex<CodeMemory>,
    entry: Entry,
    arity: usize,
    buffers: Box<[BufferKind]>,
}

impl Codegen for Cranelift {
    type Output = CraneliftKernel;

    fn compile(function: &Function) -> Result<CraneliftKernel> {
        let mut memory = CodeMemory(Some(JITModule::new(builder()?)));
        let module = memory.0.as_mut().unwrap();
        let kernel = module::KernelModule::lower(module, function)?;
        let id = kernel.entry;
        kernel.emit(&mut module.make_context(), |id, ctx| {
            module.define_function(id, ctx).map_err(|e| e.to_string())
        })?;
        module.finalize_definitions().map_err(|e| e.to_string())?;
        let address = module.get_finalized_function(id);
        // SAFETY: KernelModule declares the entry with the host C calling convention,
        // two pointer parameters and an i64 return. The kernel owns its module.
        let entry = unsafe { std::mem::transmute::<*const u8, Entry>(address) };
        Ok(CraneliftKernel {
            _memory: Mutex::new(memory),
            entry,
            arity: function.arity(),
            buffers: function.buffer_kinds().into(),
        })
    }
}

impl Executable for CraneliftKernel {
    fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32> {
        if args.len() != self.arity {
            return Err(format!(
                "expected {} scalar arguments, got {}",
                self.arity,
                args.len()
            ));
        }
        splice::validate_buffers(&self.buffers, buffers)?;
        let descriptors: Vec<_> = buffers
            .iter_mut()
            .map(|buffer| Descriptor {
                len: buffer.len() as i32,
                data: buffer.data(),
            })
            .collect();
        // SAFETY: The signature check covers both input arrays, and their
        // backing slices stay borrowed throughout the call. Verified HIR
        // enforces write permissions, and lowering guards each element access.
        let result = unsafe { (self.entry)(args.as_ptr(), descriptors.as_ptr()) };
        match result {
            BOUNDS => Err("kernel slice index out of bounds".into()),
            DIV_ZERO => Err("kernel integer division by zero".into()),
            OVERFLOW => Err("kernel signed division overflow".into()),
            0..=0xffff_ffff => Ok(result as i32),
            _ => Err(format!("unexpected Cranelift kernel result {result}")),
        }
    }
}
