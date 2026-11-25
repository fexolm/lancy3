//! The one call path of every compiled kernel: the checks of a call's
//! arguments, the slice descriptors native code reads, the native entry
//! and the decoding of its status. A backend provides only
//! [`Executable::enter`].
use crate::hir::{BufferArg, BufferKind, Constant, ScalarType};
use crate::{CallError, Target, TargetError, frontend::Signature};

use splice_target::abi::decode_status;

/// A compiled kernel. Scalars and slices have separate argument lists, each
/// following declaration order. Each scalar argument and the result is one
/// word holding the value's [`Constant::bits`]: see [`KernelScalar`](crate::KernelScalar) for
/// the encoding. An error stops execution; completed stores remain visible
/// to the caller.
///
/// # Safety
/// [`enter`](Self::enter) must run code compiled for
/// [`interface`](Self::interface), so that arguments that match the
/// interface are all the code needs to be sound.
pub unsafe trait Executable {
    /// What a call must pass.
    fn interface(&self) -> &Interface;

    /// Run the code on arguments [`call`](Self::call) has checked; the
    /// result's bits above its type may be undefined.
    ///
    /// # Safety
    /// `words` holds one word per parameter of the interface with exactly
    /// its value, and `slices` one descriptor per buffer of the buffer's
    /// kind: a live slice, exclusively borrowed when writable, or an
    /// exclusively borrowed Rust context, for the whole call.
    #[doc(hidden)]
    unsafe fn enter(
        &self,
        words: &[u64],
        slices: &[SliceDescriptor],
    ) -> std::result::Result<u64, CallError>;

    /// Call the kernel. Every backend shares this one path: it checks the
    /// arguments against the interface and the calling thread, describes
    /// the slices, enters the code and decodes the trap.
    fn call(
        &self,
        args: &[u64],
        buffers: &mut [BufferArg<'_>],
    ) -> std::result::Result<u64, CallError> {
        call(self, args, buffers)
    }
}

// SAFETY: both forward to the kernel they refer to.
unsafe impl<E: Executable + ?Sized> Executable for &E {
    fn interface(&self) -> &Interface {
        (**self).interface()
    }
    unsafe fn enter(
        &self,
        words: &[u64],
        slices: &[SliceDescriptor],
    ) -> std::result::Result<u64, CallError> {
        // SAFETY: the caller's contract, for the same interface.
        unsafe { (**self).enter(words, slices) }
    }
}

// SAFETY: as for `&E`.
unsafe impl<E: Executable + ?Sized> Executable for Box<E> {
    fn interface(&self) -> &Interface {
        (**self).interface()
    }
    unsafe fn enter(
        &self,
        words: &[u64],
        slices: &[SliceDescriptor],
    ) -> std::result::Result<u64, CallError> {
        // SAFETY: the caller's contract, for the same interface.
        unsafe { (**self).enter(words, slices) }
    }
}

/// A slice as native code receives it. A Rust context is a descriptor
/// whose `data` is the context and whose `len` is zero.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SliceDescriptor {
    pub data: *mut u8,
    /// At most `i32::MAX`: [`crate::validate_buffers`] rejects longer slices.
    pub len: u64,
}

const _: () =
    assert!(size_of::<SliceDescriptor>() == 16 && std::mem::offset_of!(SliceDescriptor, len) == 8);

/// The entry LLVM and Cranelift emit: one scalar word per parameter, one
/// descriptor per buffer and the result word, which the caller zeroes. It
/// returns a status ([`crate::Trap::status`]); the result travels in memory.
pub type Entry = unsafe extern "C" fn(
    words: *const u64,
    slices: *const SliceDescriptor,
    result: *mut u64,
) -> i64;

/// Run a native [`Entry`] and decode its status.
///
/// # Safety
/// `entry` must be live code compiled for the interface `words` and
/// `slices` were checked against (see [`Executable::enter`]).
#[inline]
pub unsafe fn enter_native(
    entry: Entry,
    words: &[u64],
    slices: &[SliceDescriptor],
) -> Result<u64, CallError> {
    let mut result = 0;
    // SAFETY: the caller's contract: the arrays match the compiled code,
    // which writes at most the result word.
    let status = unsafe { entry(words.as_ptr(), slices.as_ptr(), &mut result) };
    decode_status(status)
        .map(|_| result)
        .map_err(CallError::from)
}

/// What a compiled kernel expects of a call: the scalar and buffer
/// parameters, the result, and the vector length of the calling thread.
#[derive(Clone, Debug)]
pub struct Interface {
    params: Box<[ScalarType]>,
    result: ScalarType,
    buffers: Box<[BufferKind]>,
    sve: Option<Sve>,
}

/// SVE code depends on the vector length of the thread that runs it, which
/// each thread can set: it is checked on every call, not once at the build.
#[derive(Clone, Copy, Debug)]
struct Sve {
    target: Target,
    /// The code assumes a power-of-two vector length (LLVM's scalable
    /// vectors do).
    power_of_two: bool,
}

impl Interface {
    pub fn new(signature: Signature<'_>) -> Self {
        Self {
            params: signature.params.into(),
            result: signature.result,
            buffers: signature.buffers.into(),
            sve: None,
        }
    }

    /// The code was compiled for `target` and reads the calling thread's
    /// SVE vector length; with `power_of_two`, it also assumes that the
    /// length is a power of two.
    pub fn with_thread_vector_length(self, target: Target, power_of_two: bool) -> Self {
        Self {
            sve: target.sve().then_some(Sve {
                target,
                power_of_two,
            }),
            ..self
        }
    }

    pub fn signature(&self) -> Signature<'_> {
        Signature::new(&self.params, self.result, &self.buffers)
    }

    #[inline]
    fn check_thread(&self) -> Result<(), TargetError> {
        match self.sve {
            None => Ok(()),
            Some(sve) => sve.check(),
        }
    }
}

impl Sve {
    fn check(self) -> Result<(), TargetError> {
        self.target.check_execution()?;
        if self.power_of_two
            && !splice_target::thread_sve_bytes().is_some_and(usize::is_power_of_two)
        {
            return Err(TargetError::VectorLengthNotPowerOfTwo);
        }
        Ok(())
    }
}

/// Descriptors of this many buffers live on the stack of a call.
const INLINE_SLICES: usize = 8;

/// [`Executable::call`]: check the arguments against the interface, describe
/// the slices, enter the code and keep only the result's bits.
#[inline]
pub(crate) fn call<E: Executable + ?Sized>(
    kernel: &E,
    args: &[u64],
    buffers: &mut [BufferArg<'_>],
) -> Result<u64, CallError> {
    let interface = kernel.interface();
    interface.check_thread()?;
    crate::validate_scalars(&interface.params, args)?;
    crate::validate_buffers(&interface.buffers, buffers)?;
    if buffers.is_empty() {
        // SAFETY: as below, with no buffers.
        let word = unsafe { kernel.enter(args, &[]) }?;
        return Ok(Constant::new(interface.result, word).bits());
    }
    let describe = |buffer: &mut BufferArg<'_>| SliceDescriptor {
        data: buffer.data(),
        len: buffer.len() as u64,
    };
    let empty = SliceDescriptor {
        data: std::ptr::null_mut(),
        len: 0,
    };
    let mut inline = [empty; INLINE_SLICES];
    let heap: Vec<SliceDescriptor>;
    let slices = if buffers.len() <= INLINE_SLICES {
        for (slot, buffer) in inline.iter_mut().zip(buffers.iter_mut()) {
            *slot = describe(buffer);
        }
        &inline[..buffers.len()]
    } else {
        heap = buffers.iter_mut().map(describe).collect();
        &heap
    };
    // SAFETY: the words and descriptors match the interface the kernel
    // was compiled for: one word per parameter that holds exactly its
    // value, and one descriptor per buffer of its kind, of a live slice
    // (exclusive when writable) or Rust context borrowed for this call.
    let word = unsafe { kernel.enter(args, slices) }?;
    // A backend may leave the bits above a narrow result undefined.
    Ok(Constant::new(interface.result, word).bits())
}
