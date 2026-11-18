use super::{Executable, os_error};
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

pub(super) fn allocate(len: usize) -> Result<NonNull<c_void>, String> {
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
        "mmap returned a null base".into()
    })
}

pub(super) fn seal(mem: &Executable) -> Result<(), String> {
    const PROT_READ_EXEC: i32 = 1 | 4;
    // x86-64 has coherent instruction/data caches. The kernel publishes
    // this newly created mapping only after mprotect succeeds.
    if unsafe { mprotect(mem.address.as_ptr(), mem.len, PROT_READ_EXEC) } != 0 {
        return Err(os_error("mprotect"));
    }
    Ok(())
}

pub(super) fn release(mem: &Executable) {
    unsafe { munmap(mem.address.as_ptr(), mem.len) };
}
