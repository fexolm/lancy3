use super::executable::{Executable, os_error};
use crate::CompileError;
use std::ffi::c_void;
use std::ptr::{self, NonNull};
use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc,
    VirtualFree, VirtualProtect,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

pub(super) fn allocate(len: usize) -> Result<NonNull<c_void>, CompileError> {
    // SAFETY: a new region at an address the system chooses aliases no
    // existing memory.
    let p = unsafe { VirtualAlloc(ptr::null(), len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) };
    NonNull::new(p).ok_or_else(|| os_error("VirtualAlloc"))
}

pub(super) fn seal(mem: &Executable) -> Result<(), CompileError> {
    let address = mem.address.as_ptr();
    let mut old = 0;
    // SAFETY: `mem` owns the region of `mem.len` bytes; nothing else
    // references it before it is published.
    if unsafe { VirtualProtect(address, mem.len, PAGE_EXECUTE_READ, &mut old) } == 0 {
        return Err(os_error("VirtualProtect"));
    }
    // SAFETY: flushing the instruction cache of this process's own region.
    if unsafe { FlushInstructionCache(GetCurrentProcess(), address, mem.len) } == 0 {
        return Err(os_error("FlushInstructionCache"));
    }
    Ok(())
}

pub(super) fn release(mem: &Executable) {
    // SAFETY: `mem` owns the region, and its owner is dropped, so no
    // invocation is running in it.
    let freed = unsafe { VirtualFree(mem.address.as_ptr(), 0, MEM_RELEASE) };
    debug_assert_ne!(freed, 0, "VirtualFree of an owned region");
}
