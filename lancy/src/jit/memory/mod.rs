//! Owned executable mappings. Bytes are written before a mapping becomes executable.
use std::ffi::c_void;
use std::ptr::{self, NonNull};

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
use windows as os;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
use linux as os;

pub(crate) struct Executable {
    pub(crate) address: NonNull<c_void>,
    pub(crate) len: usize,
}

// SAFETY: Once published the mapping is immutable. A caller borrows the owner
// for the duration of execution, so drop cannot race an active invocation.
unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

impl Executable {
    pub(crate) fn new(code: &[u8]) -> Result<Self, String> {
        if code.is_empty() {
            return Err("empty machine code".into());
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

fn os_error(operation: &str) -> String {
    format!("{operation}: {}", std::io::Error::last_os_error())
}
