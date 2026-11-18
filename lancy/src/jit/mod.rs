//! Install verified HIR as executable machine code.
mod buffers;
mod memory;

use self::buffers::{descriptors, prepare_header};
use self::memory::Executable;
use crate::hir::{BufferArg, BufferKind, Function};
use std::mem::MaybeUninit;

type Entry = unsafe extern "C" fn(*const i32, *mut i32) -> i32;

/// Owns the executable memory backing a compiled function and its callees.
pub struct ScalarKernel {
    memory: Executable,
    entry: Entry,
    arity: usize,
    buffers: Box<[BufferKind]>,
    words: usize,
}

impl ScalarKernel {
    pub fn compile(function: &Function) -> Result<Self, String> {
        let (bytes, words) = crate::lir::emit(function)?;
        let memory = Executable::new(&bytes)?;
        // SAFETY: the emitter implements Entry for both supported x64 ABIs.
        let entry =
            unsafe { std::mem::transmute::<*mut std::ffi::c_void, Entry>(memory.address.as_ptr()) };
        Ok(Self {
            memory,
            entry,
            arity: function.arity(),
            buffers: function.buffer_kinds().into(),
            words,
        })
    }
    pub fn code_size(&self) -> usize {
        self.memory.len
    }
    pub fn workspace_words(&self) -> usize {
        self.words
    }

    pub fn call(&self, args: &[i32]) -> Result<i32, String> {
        self.call_with_buffers(args, &mut [])
    }

    /// Earlier stores remain visible if a bounds or arithmetic check fails.
    pub fn call_with_buffers(
        &self,
        args: &[i32],
        buffers: &mut [BufferArg<'_>],
    ) -> Result<i32, String> {
        if args.len() != self.arity {
            return Err(format!(
                "expected {} scalar arguments, got {}",
                self.arity,
                args.len()
            ));
        }
        let views = descriptors(&self.buffers, buffers)?;
        let mut workspace = vec![MaybeUninit::uninit(); self.words];
        let mut status = 0;
        prepare_header(&mut workspace, &mut status, &views);
        // SAFETY: the verifier covers all reachable reads. Workspace sizing
        // includes all call frames. Arguments and slice descriptors live through the call.
        let value = unsafe { (self.entry)(args.as_ptr(), workspace.as_mut_ptr().cast()) };
        match status {
            0 => Ok(value),
            1 => Err("kernel slice index out of bounds".into()),
            2 => Err("kernel integer division by zero".into()),
            3 => Err("kernel signed division overflow".into()),
            _ => unreachable!("unknown native kernel status {status}"),
        }
    }
}
