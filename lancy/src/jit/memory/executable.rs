//! Owned executable mappings. Bytes are written before a mapping becomes executable.
use crate::CompileError;
use std::ffi::c_void;
use std::ptr::{self, NonNull};

#[cfg(target_os = "linux")]
use super::linux as os;
#[cfg(target_os = "windows")]
use super::windows as os;

pub(in crate::jit) struct Executable {
    pub(in crate::jit) address: NonNull<c_void>,
    pub(in crate::jit) len: usize,
}

// SAFETY: Once published the mapping is immutable. A caller borrows the owner
// for the duration of execution, so drop cannot race an active invocation.
unsafe impl Send for Executable {}
// SAFETY: shared references only read or execute the immutable mapping.
unsafe impl Sync for Executable {}

/// Installing nothing is a caller defect: every kernel has a body.
#[derive(Debug, thiserror::Error)]
#[error("cannot install empty machine code")]
struct EmptyCode;

impl Executable {
    pub(in crate::jit) fn new(code: &[u8]) -> Result<Self, CompileError> {
        if code.is_empty() {
            return Err(CompileError::internal(EmptyCode));
        }
        let len = code.len();
        let memory = Self {
            address: os::allocate(len)?,
            len,
        };
        // SAFETY: allocate returned an exclusive writable mapping of len bytes.
        unsafe { ptr::copy_nonoverlapping(code.as_ptr(), memory.address.as_ptr().cast(), len) };
        os::seal(&memory)?;
        Ok(memory)
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        os::release(self);
    }
}

pub(super) fn os_error(operation: &'static str) -> CompileError {
    CompileError::Os {
        operation,
        source: std::io::Error::last_os_error(),
    }
}
