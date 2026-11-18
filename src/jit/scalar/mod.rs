//! Installation and invocation of scalar functions.
mod adaptive;
mod adaptive_buffer;
mod buffers;
mod profile;

use super::Executable;
use crate::hir::{BufferArg, BufferKind, Function};
use crate::lir::{emit_scalar, emit_scalar3};
pub use adaptive::AdaptiveScalar;
pub use adaptive_buffer::AdaptiveBufferKernel;
use buffers::{Descriptor, descriptors, prepare_header};
use std::mem::MaybeUninit;

type Entry = unsafe extern "C" fn(*const i32, *mut i32) -> i32;
type Entry3 = unsafe extern "C" fn(i32, i32, i32, *mut i32) -> i32;

#[inline]
fn with_workspace<T>(words: usize, f: impl FnOnce(&mut [MaybeUninit<i32>]) -> T) -> T {
    const INLINE_WORDS: usize = 128;
    if words <= INLINE_WORDS {
        f(&mut [MaybeUninit::uninit(); INLINE_WORDS])
    } else {
        f(&mut vec![MaybeUninit::uninit(); words])
    }
}

/// A scalar function which owns its immutable machine code.
pub struct ScalarKernel {
    memory: Executable,
    entry: Entry,
    arity: usize,
    buffers: Box<[BufferKind]>,
    needs_status: bool,
    words: usize,
}

impl ScalarKernel {
    pub fn compile(function: &Function) -> Result<Self, String> {
        Self::baseline(&function.simplified()?)
    }

    /// Emit the verified instruction stream directly for low startup latency.
    pub fn baseline(function: &Function) -> Result<Self, String> {
        let (bytes, words) = emit_scalar(function)?;
        let memory = Executable::new(&bytes)?;
        // SAFETY: the verified scalar backend emits only functions with this ABI.
        // The owner remains live for every invocation.
        let entry =
            unsafe { std::mem::transmute::<*mut std::ffi::c_void, Entry>(memory.address.as_ptr()) };
        Ok(Self {
            memory,
            entry,
            arity: function.arity(),
            buffers: function.buffer_kinds().into(),
            needs_status: function.needs_status(),
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
        self.check_scalars_only(args.len())?;
        self.invoke(args)
    }

    /// Run one whole function. Slices follow their order in the kernel signature.
    /// An invalid index stops the invocation. Earlier stores remain visible.
    pub fn call_with_buffers(
        &self,
        args: &[i32],
        buffers: &mut [BufferArg<'_>],
    ) -> Result<i32, String> {
        if self.buffers.is_empty() && buffers.is_empty() {
            return self.call(args);
        }
        self.check(args.len())?;
        let mut views = Vec::with_capacity(self.buffers.len());
        descriptors(&self.buffers, buffers, &mut views)?;
        with_workspace(self.words, |scratch| {
            self.execute_with_views(args, &views, scratch)
        })
    }

    /// Prepare reusable storage for calls with arbitrary register pressure.
    pub fn caller(&self) -> ScalarCaller<'_> {
        ScalarCaller {
            kernel: self,
            workspace: vec![MaybeUninit::uninit(); self.words],
            views: Vec::with_capacity(self.buffers.len()),
        }
    }

    fn check(&self, n: usize) -> Result<(), String> {
        if n != self.arity {
            return Err(format!("expected {} scalar arguments, got {n}", self.arity));
        }
        Ok(())
    }

    fn check_scalars_only(&self, n: usize) -> Result<(), String> {
        self.check(n)?;
        if !self.buffers.is_empty() {
            return Err("this kernel expects slices; use call_with_buffers".into());
        }
        Ok(())
    }

    fn execute_with_views(
        &self,
        args: &[i32],
        views: &[Descriptor],
        scratch: &mut [MaybeUninit<i32>],
    ) -> Result<i32, String> {
        if !self.needs_status {
            return Ok(self.execute(args, scratch));
        }
        let mut status = 0;
        prepare_header(scratch, &mut status, views);
        let value = self.execute(args, scratch);
        match status {
            0 => Ok(value),
            1 => Err("kernel slice index out of bounds".into()),
            2 => Err("kernel integer division by zero".into()),
            3 => Err("kernel signed division overflow".into()),
            _ => unreachable!("unknown native kernel status {status}"),
        }
    }

    #[inline]
    fn invoke(&self, args: &[i32]) -> Result<i32, String> {
        with_workspace(self.words, |scratch| {
            self.execute_with_views(args, &[], scratch)
        })
    }

    #[inline]
    fn execute(&self, args: &[i32], scratch: &mut [MaybeUninit<i32>]) -> i32 {
        debug_assert_eq!(args.len(), self.arity);
        debug_assert!(scratch.len() >= self.words);
        // SAFETY: arity is checked dynamically or by the typed constructor.
        // Every reachable operand is initialized before it is read, and the
        // emitter accounts for the workspace of all nested calls.
        unsafe { (self.entry)(args.as_ptr(), scratch.as_mut_ptr().cast()) }
    }
}

pub struct ScalarCaller<'a> {
    kernel: &'a ScalarKernel,
    workspace: Vec<MaybeUninit<i32>>,
    views: Vec<Descriptor>,
}

impl ScalarCaller<'_> {
    pub fn call(&mut self, args: &[i32]) -> Result<i32, String> {
        self.kernel.check_scalars_only(args.len())?;
        self.kernel
            .execute_with_views(args, &[], &mut self.workspace)
    }

    /// Reuse both the native workspace and the slice descriptor allocation.
    pub fn call_with_buffers(
        &mut self,
        args: &[i32],
        buffers: &mut [BufferArg<'_>],
    ) -> Result<i32, String> {
        if self.kernel.buffers.is_empty() && buffers.is_empty() {
            return self.call(args);
        }
        self.kernel.check(args.len())?;
        descriptors(&self.kernel.buffers, buffers, &mut self.views)?;
        let result = self
            .kernel
            .execute_with_views(args, &self.views, &mut self.workspace);
        self.views.clear();
        result
    }
}

/// Validate the function's arity once during installation.
pub struct ScalarFn<const N: usize>(TypedCode);

enum TypedCode {
    Generic(ScalarKernel),
    Three {
        _owner: Executable,
        entry: Entry3,
        words: usize,
    },
}

impl<const N: usize> ScalarFn<N> {
    pub fn compile(f: &Function) -> Result<Self, String> {
        Self::baseline(&f.simplified()?)
    }

    pub fn baseline(f: &Function) -> Result<Self, String> {
        if !f.buffer_kinds().is_empty() {
            return Err(
                "ScalarFn expects only scalar arguments; use ScalarKernel for slices".into(),
            );
        }
        if f.arity() != N {
            return Err(format!(
                "expected arity {N}, function has {} parameters",
                f.arity()
            ));
        }
        let code = if N == 3 && !f.needs_status() {
            let (bytes, words) = emit_scalar3(f)?;
            let memory = Executable::new(&bytes)?;
            // SAFETY: emit_scalar3 uses the native ABI with three i32 parameters and scratch.
            let entry = unsafe {
                std::mem::transmute::<*mut std::ffi::c_void, Entry3>(memory.address.as_ptr())
            };
            TypedCode::Three {
                _owner: memory,
                entry,
                words,
            }
        } else {
            TypedCode::Generic(ScalarKernel::baseline(f)?)
        };
        Ok(Self(code))
    }
    #[inline]
    pub fn invoke(&self, args: [i32; N]) -> i32 {
        self.try_invoke(args)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Returns runtime arithmetic errors to the caller.
    #[inline]
    pub fn try_invoke(&self, args: [i32; N]) -> Result<i32, String> {
        match &self.0 {
            TypedCode::Generic(k) => k.invoke(&args),
            TypedCode::Three { entry, words, .. } => Ok(with_workspace(*words, |scratch| {
                // SAFETY: only the checked N=3 constructor creates this variant.
                unsafe { entry(args[0], args[1], args[2], scratch.as_mut_ptr().cast()) }
            })),
        }
    }
}

pub type ScalarFn3 = ScalarFn<3>;
impl ScalarFn<3> {
    #[inline]
    pub fn call(&self, a: i32, b: i32, c: i32) -> i32 {
        self.invoke([a, b, c])
    }
}
