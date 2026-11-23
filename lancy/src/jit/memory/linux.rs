use super::executable::{Executable, os_error};
use crate::CompileError;
use std::ffi::c_void;
use std::ptr::{self, NonNull};

unsafe extern "C" {
    fn mmap(
        address: *mut c_void,
        size: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn mprotect(address: *mut c_void, size: usize, prot: i32) -> i32;
    fn munmap(address: *mut c_void, size: usize) -> i32;
}

pub(super) fn allocate(len: usize) -> Result<NonNull<c_void>, CompileError> {
    const MAP_PRIVATE_ANONYMOUS: i32 = 0x02 | 0x20;
    const PROT_READ_WRITE: i32 = 1 | 2;
    let p = unsafe {
        mmap(
            ptr::null_mut(),
            len,
            PROT_READ_WRITE,
            MAP_PRIVATE_ANONYMOUS,
            -1,
            0,
        )
    };
    if p as isize == -1 {
        return Err(os_error("mmap"));
    }
    NonNull::new(p).ok_or_else(|| {
        unsafe { munmap(p, len) };
        CompileError::Os {
            operation: "mmap",
            source: std::io::Error::other("mapping has a null base address"),
        }
    })
}

pub(super) fn seal(mem: &Executable) -> Result<(), CompileError> {
    const PROT_READ_EXEC: i32 = 1 | 4;
    #[cfg(target_arch = "aarch64")]
    synchronize(mem);
    // Publish the mapping only after mprotect succeeds.
    if unsafe { mprotect(mem.address.as_ptr(), mem.len, PROT_READ_EXEC) } != 0 {
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
    unsafe { munmap(mem.address.as_ptr(), mem.len) };
}
