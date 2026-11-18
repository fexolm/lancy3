//! Compile verified Splice kernels to native code with Cranelift.
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(not(all(
    target_arch = "x86_64",
    target_pointer_width = "64",
    any(windows, target_os = "linux")
)))]
compile_error!("splice-cranelift currently supports x86-64 Windows and Linux");

mod lower;

use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::default_libcall_names;
use splice::{BufferArg, BufferKind, Codegen, Executable, Function, Result};
use std::sync::Mutex;

/// Compile a specialized kernel or a portable Splice function with Cranelift.
pub struct Cranelift;

#[repr(C)]
struct Descriptor {
    data: *mut i32,
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
        let builder = JITBuilder::with_flags(
            &[
                ("opt_level", "speed"),
                ("enable_probestack", "true"),
                ("probestack_strategy", "inline"),
            ],
            default_libcall_names(),
        )
        .map_err(|e| e.to_string())?;
        let mut memory = CodeMemory(Some(JITModule::new(builder)));
        let module = memory.0.as_mut().unwrap();
        let id = lower::compile(module, function)?;
        module.finalize_definitions().map_err(|e| e.to_string())?;
        let address = module.get_finalized_function(id);
        // SAFETY: lower declares the entry with the host C calling convention,
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
                data: match buffer {
                    BufferArg::Read(s) => s.as_ptr().cast_mut(),
                    BufferArg::Write(s) => s.as_mut_ptr(),
                },
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
