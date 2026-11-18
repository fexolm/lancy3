use super::executable::{Executable, os_error};
use crate::CompileError;
use std::ffi::c_void;
use std::ptr::{self, NonNull};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualAlloc(address: *mut c_void, size: usize, kind: u32, protection: u32) -> *mut c_void;
    fn VirtualProtect(address: *mut c_void, size: usize, protection: u32, old: *mut u32) -> i32;
    fn VirtualFree(address: *mut c_void, size: usize, kind: u32) -> i32;
    fn GetCurrentProcess() -> *mut c_void;
    fn FlushInstructionCache(process: *mut c_void, address: *const c_void, size: usize) -> i32;
}

pub(super) fn allocate(len: usize) -> Result<NonNull<c_void>, CompileError> {
    const MEM_COMMIT_RESERVE: u32 = 0x1000 | 0x2000;
    const PAGE_READWRITE: u32 = 0x04;
    let p = unsafe { VirtualAlloc(ptr::null_mut(), len, MEM_COMMIT_RESERVE, PAGE_READWRITE) };
    NonNull::new(p).ok_or_else(|| os_error("VirtualAlloc"))
}

pub(super) fn seal(mem: &Executable) -> Result<(), CompileError> {
    const PAGE_EXECUTE_READ: u32 = 0x20;
    let address = mem.address.as_ptr();
    let mut old = 0;
    if unsafe { VirtualProtect(address, mem.len, PAGE_EXECUTE_READ, &mut old) } == 0 {
        return Err(os_error("VirtualProtect"));
    }
    if unsafe { FlushInstructionCache(GetCurrentProcess(), address, mem.len) } == 0 {
        return Err(os_error("FlushInstructionCache"));
    }
    Ok(())
}

pub(super) fn release(mem: &Executable) {
    const MEM_RELEASE: u32 = 0x8000;
    unsafe { VirtualFree(mem.address.as_ptr(), 0, MEM_RELEASE) };
}
