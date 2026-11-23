//! Registered symbol files reach the process-wide GDB JIT descriptor, which
//! is the one debuggers read even when LLVM supplies its definition (BE-05).
use splice_jit::gimli::write::FrameTable;
use splice_jit::{Function, Registration};

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

fn function(name: &str, address: u64) -> Function {
    Function {
        name: name.to_owned(),
        address,
        size: 16,
        statements: Vec::new(),
    }
}

#[test]
fn registrations_link_into_the_gdb_descriptor() {
    let descriptor = &raw const __jit_debug_descriptor;
    // SAFETY: only this test registers objects in this process, and the
    // descriptor is read between registrations.
    let first = || unsafe { (&raw const (*descriptor).first).read_volatile() };
    unsafe {
        assert_eq!((*descriptor).version, 1);
        assert_eq!((*descriptor).action, 0);
    }
    assert!(first().is_null());

    let frames = FrameTable::default();
    let outer = Registration::new(&[function("outer", 0x1000)], &frames).unwrap();
    let inner = Registration::new(&[function("inner", 0x2000)], &frames).unwrap();
    // SAFETY: both entries stay registered while they are read.
    unsafe {
        let newest = &*first();
        let oldest = &*newest.next;
        assert!(newest.previous.is_null() && oldest.next.is_null());
        assert_eq!(oldest.previous, newest as *const Entry);
        for entry in [newest, oldest] {
            let bytes = std::slice::from_raw_parts(entry.bytes, entry.size as usize);
            assert_eq!(&bytes[..4], b"\x7fELF");
        }
        assert!((*descriptor).relevant.is_null());
    }
    drop(outer);
    // SAFETY: the inner entry is still registered.
    unsafe {
        let only = &*first();
        assert!(only.next.is_null() && only.previous.is_null());
    }
    drop(inner);
    assert!(first().is_null());
}
