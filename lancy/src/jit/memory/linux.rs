use super::executable::{Executable, os_error};
use crate::CompileError;
use std::ffi::c_void;
use std::ptr::{self, NonNull};

pub(super) fn allocate(len: usize) -> Result<NonNull<c_void>, CompileError> {
    // SAFETY: an anonymous private mapping at an address the kernel chooses
    // aliases no existing memory.
    let p = unsafe {
        libc::mmap(
            ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return Err(os_error("mmap"));
    }
    NonNull::new(p).ok_or_else(|| {
        // SAFETY: `p` is the mapping of `len` bytes just created, unused.
        let unmapped = unsafe { libc::munmap(p, len) };
        debug_assert_eq!(unmapped, 0, "munmap of a fresh mapping");
        CompileError::Os {
            operation: "mmap",
            source: std::io::Error::other("mapping has a null base address"),
        }
    })
}

pub(super) fn seal(mem: &Executable) -> Result<(), CompileError> {
    #[cfg(target_arch = "aarch64")]
    synchronize(mem);
    // Publish the mapping only after mprotect succeeds.
    // SAFETY: `mem` owns the mapping of `mem.len` bytes; nothing else
    // references it before it is published.
    let sealed = unsafe {
        libc::mprotect(
            mem.address.as_ptr(),
            mem.len,
            libc::PROT_READ | libc::PROT_EXEC,
        )
    };
    if sealed != 0 {
        return Err(os_error("mprotect"));
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
fn synchronize(mem: &Executable) {
    use std::arch::asm;
    let start = mem.address.as_ptr() as usize;
    let end = start + mem.len;
    let ctr: usize;
    // SAFETY: Linux permits reading CTR_EL0 at EL0. Only lines overlapping
    // the live mapping are maintained. No mappings are written by this code.
    unsafe {
        asm!("mrs {ctr}, ctr_el0", ctr = out(reg) ctr, options(nostack, preserves_flags));
        if ctr & (1 << 28) == 0 {
            let line = 4 << ((ctr >> 16) & 15);
            for address in (start & !(line - 1)..end).step_by(line) {
                asm!("dc cvau, {p}", p = in(reg) address, options(nostack, preserves_flags));
            }
        }
        asm!("dsb ish", options(nostack, preserves_flags));
        if ctr & (1 << 29) == 0 {
            let line = 4 << (ctr & 15);
            for address in (start & !(line - 1)..end).step_by(line) {
                asm!("ic ivau, {p}", p = in(reg) address, options(nostack, preserves_flags));
            }
            asm!("dsb ish", options(nostack, preserves_flags));
        }
        asm!("isb", options(nostack, preserves_flags));
    }
}

pub(super) fn release(mem: &Executable) {
    // SAFETY: `mem` owns the mapping, and its owner is dropped, so no
    // invocation is running in it.
    let unmapped = unsafe { libc::munmap(mem.address.as_ptr(), mem.len) };
    debug_assert_eq!(unmapped, 0, "munmap of an owned mapping");
}
