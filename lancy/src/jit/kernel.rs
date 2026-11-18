//! Install verified HIR as executable machine code.
use super::memory::Executable;
use crate::hir::{Function, Type};
use crate::{CallError, CompileError};
use std::mem::MaybeUninit;

#[cfg(test)]
mod tests;

type Entry = unsafe extern "C" fn(*const u64, *mut u64) -> u64;

/// Controls expansion of runtime calls in HIR.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Inlining {
    /// Expand the entire acyclic graph when the resulting function is representable.
    #[default]
    Always,
    /// Keep the internal call ABI while optimizing each body independently.
    Never,
}

/// Owns the executable memory backing a compiled function and its callees.
pub struct ScalarKernel {
    memory: Executable,
    entry: Entry,
    arity: usize,
    scalar_signature: bool,
    words: usize,
}

impl ScalarKernel {
    /// Verify the reachable functions, generate x64 code and install it for execution.
    pub fn compile(function: &Function) -> Result<Self, CompileError> {
        Self::compile_with_inlining(function, Inlining::Always)
    }

    /// Select the HIR inlining policy for this compilation.
    pub fn compile_with_inlining(
        function: &Function,
        inlining: Inlining,
    ) -> Result<Self, CompileError> {
        let (source, ids) = function.verified_module()?;
        let emit = |optimized: Function| {
            let (module, ids) = optimized.verified_module()?;
            crate::lir::emit(module, &ids)
        };
        let outlined = || emit(source.simplify_bodies(&ids)?);
        let (bytes, words) = if inlining == Inlining::Always && source.functions.len() > 1 {
            let expanded = source.inline_all(&ids).and_then(|mut body| {
                body.simplify()?;
                emit(body)
            });
            match expanded {
                // Selection can introduce extra virtual registers and branches
                // after the HIR expansion has passed its size check.
                Err(CompileError::FunctionTooLarge | CompileError::BranchOutOfRange { .. }) => {
                    outlined()
                }
                other => other,
            }
        } else {
            outlined()
        }?;
        let memory = Executable::new(&bytes)?;
        // SAFETY: the emitter implements Entry for both supported x64 ABIs.
        let entry =
            unsafe { std::mem::transmute::<*mut std::ffi::c_void, Entry>(memory.address.as_ptr()) };
        Ok(Self {
            memory,
            entry,
            arity: function.arity(),
            scalar_signature: function.return_type() == Type::I32
                && function.param_types().iter().all(|&ty| ty == Type::I32),
            words,
        })
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
    pub unsafe fn call_raw(&self, args: &[u64]) -> Result<u64, CallError> {
        self.check_arity(args.len())?;
        let mut workspace = vec![MaybeUninit::<u64>::uninit(); self.words];
        // SAFETY: the caller supplies valid memory for every external access.
        unsafe { self.invoke(args.as_ptr(), &mut workspace) }
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
