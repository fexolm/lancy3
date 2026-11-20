//! Install verified HIR as executable machine code.
use super::memory::Executable;
use crate::hir::{Function, Type};
use crate::{CallError, CompileError, Inlining};
use std::mem::MaybeUninit;

#[cfg(all(test, target_arch = "x86_64"))]
mod tests;

type Entry = unsafe extern "C" fn(*const u64, *mut u64) -> u64;

/// Owns the executable memory backing a compiled function and its callees.
pub struct ScalarKernel {
    memory: Executable,
    entry: Entry,
    arity: usize,
    scalar_signature: bool,
    words: usize,
    arg_words: usize,
    result_words: usize,
    pointer_signature: bool,
    #[cfg(target_arch = "aarch64")]
    sites: Box<[crate::aarch64::Site]>,
}

impl ScalarKernel {
    /// Verify the reachable functions, generate machine code and install it for execution.
    pub fn compile(function: &Function) -> Result<Self, CompileError> {
        Self::compile_with_inlining(function, Inlining::Always)
    }

    /// Select the HIR inlining policy for this compilation.
    pub fn compile_with_inlining(
        function: &Function,
        inlining: Inlining,
    ) -> Result<Self, CompileError> {
        let _span =
            tracing::debug_span!(target: "splice::compile", "lancy_jit", inlining = ?inlining)
                .entered();
        splice_target::simd().map_err(CompileError::SimdProfile)?;
        let (source, ids) = function.verified_module()?;
        let code = super::optimize_and_lower(&source, &ids, inlining, crate::lir::emit)?;
        let memory = tracing::debug_span!(target: "splice::compile", "install_code")
            .in_scope(|| Executable::new(&code.bytes))?;
        // SAFETY: the host emitter implements Entry and memory owns the sealed code.
        let entry =
            unsafe { std::mem::transmute::<*mut std::ffi::c_void, Entry>(memory.address.as_ptr()) };
        Ok(Self {
            memory,
            entry,
            arity: function.arity(),
            scalar_signature: function.return_type() == Type::I32
                && function.param_types().iter().all(|&ty| ty == Type::I32),
            words: code.words,
            arg_words: function
                .param_types()
                .iter()
                .map(|ty| ty.bytes(Self::vector_bytes()).div_ceil(8))
                .sum(),
            result_words: function
                .return_type()
                .bytes(Self::vector_bytes())
                .div_ceil(8),
            pointer_signature: function.param_types().contains(&Type::Ptr)
                || function.return_type() == Type::Ptr,
            #[cfg(target_arch = "aarch64")]
            sites: code.sites,
        })
    }
    /// Size of one scalable vector for kernels compiled in this process.
    pub fn vector_bytes() -> usize {
        splice_target::vector_bytes()
    }

    pub fn code_size(&self) -> usize {
        self.memory.len
    }
    /// Workspace size in 32-bit words.
    pub fn workspace_words(&self) -> usize {
        self.words * 2
    }

    pub fn call(&self, args: &[i32]) -> Result<i32, CallError> {
        if !self.scalar_signature {
            return Err(CallError::PointerSignature);
        }
        self.check_arity(args.len())?;
        let mut scratch = vec![MaybeUninit::<u64>::uninit(); self.words + args.len()];
        let (workspace, packed) = scratch.split_at_mut(self.words);
        for (slot, &arg) in packed.iter_mut().zip(args) {
            slot.write(arg as u32 as u64);
        }
        // SAFETY: all ABI arguments are initialized integers. HIR has no integer-to-pointer casts.
        unsafe { self.invoke(packed.as_ptr().cast(), workspace) }.map(|value| value as i32)
    }

    /// Invoke the typed kernel with one native word per argument. I32 uses the low 32 bits.
    /// Earlier stores remain visible if a later check traps.
    ///
    /// # Safety
    /// Each pointer must be valid for every executed access, including addresses
    /// derived from it or loaded transitively. The caller must uphold lifetimes,
    /// initialization and Rust aliasing rules for the duration of the call.
    /// For declared slice parameters the length must be nonnegative and the
    /// entire slice must be valid and initialized, even for a zero-trip loop.
    /// Writable slices must be exclusive and shared slices remain immutable.
    pub unsafe fn call_raw(&self, args: &[u64]) -> Result<u64, CallError> {
        if self.result_words != 1 || self.arity != self.arg_words {
            return Err(CallError::WideSignature);
        }
        self.check_arity(args.len())?;
        let mut workspace = vec![MaybeUninit::<u64>::uninit(); self.words];
        // SAFETY: the caller supplies valid memory for every external access.
        unsafe { self.invoke(args.as_ptr(), &mut workspace) }
    }

    /// Invoke a numeric signature. Values occupy consecutive little-endian words,
    /// each starting at a word boundary. Narrow scalars use the low bits.
    pub fn call_words(&self, args: &[u64], result: &mut [u64]) -> Result<(), CallError> {
        if self.pointer_signature {
            return Err(CallError::PointerSignature);
        }
        // SAFETY: verification prevents constructing pointers from numeric values.
        unsafe { self.call_words_raw(args, result) }
    }

    /// Invoke any signature with the same packed representation as `call_words`.
    /// # Safety
    /// Pointer values must satisfy the memory and lifetime contract of `call_raw`.
    pub unsafe fn call_words_raw(&self, args: &[u64], result: &mut [u64]) -> Result<(), CallError> {
        if args.len() != self.arg_words {
            return Err(CallError::Arity {
                expected: self.arg_words,
                actual: args.len(),
            });
        }
        if result.len() != self.result_words {
            return Err(CallError::ResultWords {
                expected: self.result_words,
                actual: result.len(),
            });
        }
        let mut workspace = vec![MaybeUninit::<u64>::uninit(); self.words];
        // SAFETY: the sizes match the compiled ABI and the caller covers pointer validity.
        let low = unsafe { self.invoke(args.as_ptr(), &mut workspace) }?;
        if self.result_words == 1 {
            result[0] = low;
        } else {
            for (dst, slot) in result.iter_mut().zip(&workspace[1..]) {
                // SAFETY: a successful wide root return writes every result byte at root[1].
                *dst = unsafe { slot.assume_init() };
            }
        }
        Ok(())
    }

    fn check_arity(&self, count: usize) -> Result<(), CallError> {
        if count != self.arity {
            return Err(CallError::Arity {
                expected: self.arity,
                actual: count,
            });
        }
        Ok(())
    }

    unsafe fn invoke(
        &self,
        args: *const u64,
        workspace: &mut [MaybeUninit<u64>],
    ) -> Result<u64, CallError> {
        let mut status = 0i32;
        workspace[0].write(std::ptr::from_mut(&mut status) as u64);
        #[cfg(target_arch = "aarch64")]
        {
            splice_target::simd()
                .expect("compilation validated the selected profile")
                .check_execution()
                .map_err(|_| CallError::VectorLength)?;
            workspace[crate::aarch64::SITE_POINTER].write(self.sites.as_ptr() as u64);
            // SAFETY: the executable mapping was synchronized before publication.
            // ISB synchronizes the instruction stream on this calling core.
            unsafe { std::arch::asm!("isb", options(nostack, preserves_flags)) };
        }
        // SAFETY: compile verified the internal frame layout and the caller covers external memory.
        let value = unsafe { (self.entry)(args, workspace.as_mut_ptr().cast()) };
        match status {
            0 => Ok(value),
            1 => Err(CallError::OutOfBounds),
            2 => Err(CallError::DivisionByZero),
            3 => Err(CallError::DivisionOverflow),
            _ => unreachable!("unknown native kernel status {status}"),
        }
    }
}
