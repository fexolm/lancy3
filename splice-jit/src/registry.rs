//! The linked list is the ABI defined by GDB's JIT Compilation Interface.

use std::ptr::null_mut;
use std::sync::Mutex;

#[repr(C)]
struct Entry {
    next: *mut Self,
    previous: *mut Self,
    bytes: *const u8,
    size: u64,
}

#[repr(C)]
pub struct Descriptor {
    version: u32,
    action: u32,
    relevant: *mut Entry,
    first: *mut Entry,
}

static LOCK: Mutex<()> = Mutex::new(());

// MSVC puts regular symbols in a PDB. Export the two hooks so debuggers can
// find them through the PE symbol table as well.
#[cfg(all(windows, target_env = "msvc"))]
#[used]
#[unsafe(link_section = ".drectve")]
static EXPORTS: [u8; b" /EXPORT:__jit_debug_register_code /EXPORT:__jit_debug_descriptor,DATA "
    .len()] = *b" /EXPORT:__jit_debug_register_code /EXPORT:__jit_debug_descriptor,DATA ";

#[unsafe(no_mangle)]
pub static mut __jit_debug_descriptor: Descriptor = Descriptor {
    version: 1,
    action: 0,
    relevant: null_mut(),
    first: null_mut(),
};

#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn __jit_debug_register_code() {
    // The debugger sets a breakpoint on this symbol. Keep both the call and
    // the descriptor observable when the surrounding binary is optimized.
    std::hint::black_box(&raw const __jit_debug_descriptor);
}

pub(super) struct Registration {
    entry: *mut Entry,
    _bytes: Vec<u8>,
}

// SAFETY: the entry links are accessed only under LOCK. The bytes are
// immutable and both allocations stay alive until removal from the list.
unsafe impl Send for Registration {}
// SAFETY: there is no Rust API that mutates a shared Registration.
unsafe impl Sync for Registration {}

impl Registration {
    pub(super) fn new(bytes: Vec<u8>) -> Self {
        let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let descriptor = &raw mut __jit_debug_descriptor;
        // SAFETY: LOCK protects the descriptor and every linked entry. The
        // caller retains the bytes and the heap-allocated entry until Drop.
        let entry = unsafe {
            let next = (*descriptor).first;
            let entry = Box::into_raw(Box::new(Entry {
                next,
                previous: null_mut(),
                bytes: bytes.as_ptr(),
                size: bytes.len() as u64,
            }));
            if !next.is_null() {
                (*next).previous = entry;
            }
            (*descriptor).first = entry;
            notify(descriptor, entry, 1);
            entry
        };
        Self {
            entry,
            _bytes: bytes,
        }
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let descriptor = &raw mut __jit_debug_descriptor;
        // SAFETY: This guard exclusively owns entry. LOCK keeps its neighbors
        // alive while unlinking. GDB consumes the unregister action before the
        // symbol buffer and entry are freed.
        unsafe {
            let entry = Box::from_raw(self.entry);
            if entry.previous.is_null() {
                (*descriptor).first = entry.next;
            } else {
                (*entry.previous).next = entry.next;
            }
            if !entry.next.is_null() {
                (*entry.next).previous = entry.previous;
            }
            notify(descriptor, self.entry, 2);
        }
    }
}

unsafe fn notify(descriptor: *mut Descriptor, entry: *mut Entry, action: u32) {
    // SAFETY: the caller holds LOCK and entry is alive throughout the call.
    unsafe {
        (&raw mut (*descriptor).action).write_volatile(action);
        (&raw mut (*descriptor).relevant).write_volatile(entry);
        __jit_debug_register_code();
        (&raw mut (*descriptor).relevant).write_volatile(null_mut());
        (&raw mut (*descriptor).action).write_volatile(0);
    }
}
