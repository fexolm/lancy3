//! Install verified HIR as executable machine code.
use super::{CompileOptions, memory::Executable};
use crate::hir::{Function, Type};
use crate::{CallError, CompileError, SignatureError, Trap};
use splice_target::Target;
use std::mem::MaybeUninit;

#[cfg(test)]
mod tests;

/// The native entry `(argv, workspace) -> value` in the target's calling
/// convention. Both x64 conventions are callable on either OS.
#[derive(Clone, Copy)]
enum Entry {
    #[cfg(target_arch = "x86_64")]
    SysV(SysV),
    #[cfg(target_arch = "x86_64")]
    Win64(Win64),
    #[cfg(target_arch = "aarch64")]
    Aapcs64(Aapcs64),
}

#[cfg(target_arch = "x86_64")]
type SysV = unsafe extern "sysv64" fn(*const u64, *mut u64) -> u64;
#[cfg(target_arch = "x86_64")]
type Win64 = unsafe extern "win64" fn(*const u64, *mut u64) -> u64;
#[cfg(target_arch = "aarch64")]
type Aapcs64 = unsafe extern "C" fn(*const u64, *mut u64) -> u64;

impl Entry {
    /// # Safety
    /// `address` is the entry of live code generated for `target`, which
    /// runs on this machine.
    unsafe fn new(target: &Target, address: *const ()) -> Self {
        use splice_target::CallingConvention as C;
        // SAFETY: the caller guarantees code with this exact signature and
        // convention at `address`.
        unsafe {
            match target.calling_convention() {
                #[cfg(target_arch = "x86_64")]
                C::SysV => Self::SysV(std::mem::transmute::<*const (), SysV>(address)),
                #[cfg(target_arch = "x86_64")]
                C::Win64 => Self::Win64(std::mem::transmute::<*const (), Win64>(address)),
                #[cfg(target_arch = "aarch64")]
                C::Aapcs64 => Self::Aapcs64(std::mem::transmute::<*const (), Aapcs64>(address)),
                convention => unreachable!("{convention:?} code does not run here"),
            }
        }
    }

    /// # Safety
    /// The pointers satisfy the contract of the generated code.
    unsafe fn call(self, args: *const u64, workspace: *mut u64) -> u64 {
        // SAFETY: forwarded from the caller.
        unsafe {
            match self {
                #[cfg(target_arch = "x86_64")]
                Self::SysV(entry) => entry(args, workspace),
                #[cfg(target_arch = "x86_64")]
                Self::Win64(entry) => entry(args, workspace),
                #[cfg(target_arch = "aarch64")]
                Self::Aapcs64(entry) => entry(args, workspace),
            }
        }
    }

    /// Other code at `address` with the same signature and convention.
    /// # Safety
    /// As for [`Entry::new`].
    #[cfg(all(test, target_arch = "x86_64"))]
    unsafe fn at(self, address: *const ()) -> Self {
        // SAFETY: forwarded from the caller.
        unsafe {
            match self {
                #[cfg(target_arch = "x86_64")]
                Self::SysV(_) => Self::SysV(std::mem::transmute::<*const (), SysV>(address)),
                #[cfg(target_arch = "x86_64")]
                Self::Win64(_) => Self::Win64(std::mem::transmute::<*const (), Win64>(address)),
                #[cfg(target_arch = "aarch64")]
                Self::Aapcs64(_) => {
                    Self::Aapcs64(std::mem::transmute::<*const (), Aapcs64>(address))
                }
            }
        }
    }

    #[cfg(test)]
    fn address(self) -> *const () {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::SysV(entry) => entry as *const (),
            #[cfg(target_arch = "x86_64")]
            Self::Win64(entry) => entry as *const (),
            #[cfg(target_arch = "aarch64")]
            Self::Aapcs64(entry) => entry as *const (),
        }
    }
}

/// Owns the executable memory backing a compiled function and its callees.
pub struct ScalarKernel {
    memory: Executable,
    entry: Entry,
    target: Target,
    arity: usize,
    scalar_signature: bool,
    words: usize,
    arg_words: usize,
    result_words: usize,
    pointer_signature: bool,
    debug: Vec<super::DebugFunction>,
    /// The fallback sites whose addresses the code embeds.
    _sites: crate::lir::Sites,
}

impl ScalarKernel {
    /// Verify the reachable functions, generate machine code for
    /// `options.target` and install it for execution on this machine.
    pub fn compile(function: &Function, options: &CompileOptions) -> Result<Self, CompileError> {
        let _span = tracing::debug_span!(
            target: "lancy::compile",
            "lancy_jit",
            inlining = ?options.inlining,
            debug = options.debug
        )
        .entered();
        let target = options.target;
        target.check_execution()?;
        let source = function.verified_module()?;
        // Scratch can materialize a pointer even for a numeric signature.
        // Its raw caller must ensure initialization and bound every access.
        let has_scratch = source.functions.values().any(|body| body.has_call_scratch);
        let code = if options.debug {
            crate::lir::emit(&target, source, true, &mut crate::lir::trace)?
        } else {
            super::optimize_and_lower(&source, options.inlining, |module| {
                crate::lir::emit(&target, module, false, &mut crate::lir::trace)
            })?
        };
        let memory = tracing::debug_span!(target: "lancy::compile", "install_code")
            .in_scope(|| Executable::new(&code.bytes))?;
        // SAFETY: the emitter implements the entry for `target`, which
        // check_execution accepted, and memory owns the sealed code.
        let entry = unsafe { Entry::new(&target, memory.address.as_ptr().cast_const().cast()) };
        let width = target.vector_bytes();
        Ok(Self {
            memory,
            entry,
            target,
            arity: function.arity(),
            scalar_signature: !has_scratch
                && function.return_type() == Type::I32
                && function.param_types().iter().all(|&ty| ty == Type::I32),
            words: code.words,
            arg_words: function
                .param_types()
                .iter()
                .map(|ty| ty.bytes(width).div_ceil(8))
                .sum(),
            result_words: function.return_type().bytes(width).div_ceil(8),
            pointer_signature: has_scratch
                || function.param_types().contains(&Type::Ptr)
                || function.return_type() == Type::Ptr,
            debug: code.debug,
            _sites: code.sites,
        })
    }

    /// The target the kernel was compiled for. Its vector width sizes
    /// vector arguments and results.
    pub fn target(&self) -> &Target {
        &self.target
    }

    pub fn code_size(&self) -> usize {
        self.memory.len
    }

    /// Address of the installed code, valid while this kernel is alive.
    pub fn code_address(&self) -> usize {
        self.memory.address.as_ptr() as usize
    }

    pub fn debug_functions(&self) -> &[super::DebugFunction] {
        &self.debug
    }
    /// Workspace size in 32-bit words.
    pub fn workspace_words(&self) -> usize {
        self.words * 2
    }

    /// Whether external pointers or native call scratch require the raw ABI.
    pub fn requires_raw_call(&self) -> bool {
        self.pointer_signature
    }

    pub fn call(&self, args: &[i32]) -> Result<i32, CallError> {
        if !self.scalar_signature {
            return Err(CallError::Signature(SignatureError::PointerSignature));
        }
        self.check_arity(args.len())?;
        super::workspace::with(self.words + args.len(), |scratch| {
            let (workspace, packed) = scratch.split_at_mut(self.words);
            for (slot, &arg) in packed.iter_mut().zip(args) {
                slot.write(arg as u32 as u64);
            }
            // SAFETY: all ABI arguments are initialized integers. HIR has no
            // integer-to-pointer casts.
            unsafe { self.invoke(packed.as_ptr().cast(), workspace) }
        })
        .map(|value| value as i32)
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
    /// A Rust context dispatcher must receive a live, exclusive context token
    /// of the exact type expected by that dispatcher.
    pub unsafe fn call_raw(&self, args: &[u64]) -> Result<u64, CallError> {
        if self.result_words != 1 || self.arity != self.arg_words {
            return Err(CallError::Signature(SignatureError::WideSignature));
        }
        self.check_arity(args.len())?;
        // SAFETY: the caller supplies valid memory for every external access.
        super::workspace::with(self.words, |workspace| unsafe {
            self.invoke(args.as_ptr(), workspace)
        })
    }

    /// Invoke a numeric signature. Values occupy consecutive little-endian words,
    /// each starting at a word boundary. Narrow scalars use the low bits.
    pub fn call_words(&self, args: &[u64], result: &mut [u64]) -> Result<(), CallError> {
        if self.requires_raw_call() {
            return Err(CallError::Signature(SignatureError::PointerSignature));
        }
        // SAFETY: verification prevents constructing pointers from numeric values.
        unsafe { self.call_words_raw(args, result) }
    }

    /// Invoke any signature with the same packed representation as `call_words`.
    /// # Safety
    /// Pointer values must satisfy the memory and lifetime contract of `call_raw`.
    pub unsafe fn call_words_raw(&self, args: &[u64], result: &mut [u64]) -> Result<(), CallError> {
        if args.len() != self.arg_words {
            return Err(CallError::Signature(SignatureError::Arity {
                expected: self.arg_words,
                actual: args.len(),
            }));
        }
        if result.len() != self.result_words {
            return Err(CallError::Signature(SignatureError::ResultWords {
                expected: self.result_words,
                actual: result.len(),
            }));
        }
        super::workspace::with(self.words, |workspace| {
            // SAFETY: the sizes match the compiled ABI and the caller covers
            // pointer validity.
            let low = unsafe { self.invoke(args.as_ptr(), workspace) }?;
            if self.result_words == 1 {
                result[0] = low;
            } else {
                for (dst, slot) in result.iter_mut().zip(&workspace[1..]) {
                    // SAFETY: a successful wide root return writes every
                    // result byte at root[1].
                    *dst = unsafe { slot.assume_init() };
                }
            }
            Ok(())
        })
    }

    fn check_arity(&self, count: usize) -> Result<(), CallError> {
        if count != self.arity {
            return Err(CallError::Signature(SignatureError::Arity {
                expected: self.arity,
                actual: count,
            }));
        }
        Ok(())
    }

    unsafe fn invoke(
        &self,
        args: *const u64,
        workspace: &mut [MaybeUninit<u64>],
    ) -> Result<u64, CallError> {
        // SAFETY: the caller's contract is forwarded unchanged, and the
        // closure enters the entry exactly once.
        unsafe {
            self.invoke_with(args, workspace, |entry, args, workspace| {
                entry.call(args, workspace)
            })
        }
    }

    /// Prepare the workspace, enter through `enter` and decode the status.
    /// `enter` must call the entry once with the given arguments.
    unsafe fn invoke_with(
        &self,
        args: *const u64,
        workspace: &mut [MaybeUninit<u64>],
        enter: impl FnOnce(Entry, *const u64, *mut u64) -> u64,
    ) -> Result<u64, CallError> {
        let mut status = 0u32;
        workspace[0].write(std::ptr::from_mut(&mut status) as u64);
        #[cfg(target_arch = "aarch64")]
        {
            // Each thread has its own SVE vector length.
            self.target.check_execution()?;
            // SAFETY: the executable mapping was synchronized before publication.
            // ISB synchronizes the instruction stream on this calling core.
            unsafe { std::arch::asm!("isb", options(nostack, preserves_flags)) };
        }
        // SAFETY: compile verified the internal frame layout and the caller covers external memory.
        let value = enter(self.entry, args, workspace.as_mut_ptr().cast());
        match status {
            0 => Ok(value),
            // Native code stores only trap codes; any other value would be a
            // code generation defect at the native boundary.
            code => Err(Trap::from_code(code).unwrap_or(Trap::Abi).into()),
        }
    }
}
