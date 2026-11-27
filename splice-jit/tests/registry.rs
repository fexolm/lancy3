//! Registered symbol files reach the process-wide GDB JIT descriptor, which
//! is the one debuggers read even when LLVM supplies its definition (BE-05).
use splice_jit::{Publish, Published};
use std::cell::Cell;

#[repr(C)]
struct Entry {
    next: *const Entry,
    previous: *const Entry,
    bytes: *const u8,
    size: u64,
}

#[repr(C)]
struct Descriptor {
    version: u32,
    action: u32,
    relevant: *const Entry,
    first: *const Entry,
}

unsafe extern "C" {
    static __jit_debug_descriptor: Descriptor;
}

fn publish<M>(name: &str, address: u64, memory: M) -> Published<M> {
    let mut publish = Publish::new(true);
    publish.function(name, address, 16);
    publish.finish(false, memory).unwrap()
}

/// Code memory that records, when it is freed, how many symbol files are
/// still registered.
struct Memory<'a>(&'a Cell<Option<usize>>);

impl Drop for Memory<'_> {
    fn drop(&mut self) {
        let descriptor = &raw const __jit_debug_descriptor;
        let mut count = 0;
        // SAFETY: no other thread registers while the test runs.
        unsafe {
            let mut entry = (&raw const (*descriptor).first).read_volatile();
            while !entry.is_null() {
                count += 1;
                entry = (*entry).next;
            }
        }
        self.0.set(Some(count));
    }
}

#[test]
fn registrations_link_into_the_gdb_descriptor() {
    let descriptor = &raw const __jit_debug_descriptor;
    // SAFETY: only this test registers objects in this process, and the
    // descriptor is read between registrations.
    let first = || unsafe { (&raw const (*descriptor).first).read_volatile() };
    // SAFETY: as above; nothing writes the descriptor during these reads.
    unsafe {
        assert_eq!((*descriptor).version, 1);
        assert_eq!((*descriptor).action, 0);
    }
    assert!(first().is_null());

    let freed = Cell::new(None);
    let outer = publish("outer", 0x1000, Memory(&freed));
    let inner = publish("inner", 0x2000, ());
    // SAFETY: both entries stay registered while they are read.
    unsafe {
        let newest = &*first();
        let oldest = &*newest.next;
        assert!(newest.previous.is_null() && oldest.next.is_null());
        assert_eq!(oldest.previous, newest as *const Entry);
        for entry in [newest, oldest] {
            let bytes = std::slice::from_raw_parts(entry.bytes, entry.size as usize);
            assert_eq!(&bytes[..4], b"\x7fELF");
            let machine = u16::from_le_bytes([bytes[18], bytes[19]]);
            assert_eq!(
                machine,
                if cfg!(target_arch = "x86_64") {
                    62
                } else {
                    183
                }
            );
        }
        assert!((*descriptor).relevant.is_null());
    }
    drop(outer);
    // The debugger saw the removal before the code was freed.
    assert_eq!(freed.get(), Some(1));
    // SAFETY: the inner entry is still registered.
    unsafe {
        let only = &*first();
        assert!(only.next.is_null() && only.previous.is_null());
    }
    drop(inner);
    assert!(first().is_null());
}
