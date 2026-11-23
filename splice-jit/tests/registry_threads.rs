//! Concurrent registration and removal keep the GDB JIT list well formed.
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

/// Walk the list from `first`, checking each back link, and count entries.
/// # Safety
/// No thread may register or remove entries during the walk.
unsafe fn entries() -> usize {
    let descriptor = &raw const __jit_debug_descriptor;
    let mut count = 0;
    let mut previous = std::ptr::null();
    // SAFETY: the caller excludes concurrent changes; registered entries live.
    unsafe {
        let mut entry = (&raw const (*descriptor).first).read_volatile();
        while !entry.is_null() {
            assert_eq!((*entry).previous, previous);
            assert_eq!(std::slice::from_raw_parts((*entry).bytes, 4), b"\x7fELF");
            previous = entry;
            entry = (*entry).next;
            count += 1;
        }
    }
    count
}

#[test]
fn threads_register_and_drop_symbol_files_concurrently() {
    const THREADS: usize = 8;
    const ROUNDS: usize = 200;
    let frames = FrameTable::default();
    let kept: Vec<Vec<Registration>> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..THREADS)
            .map(|t| {
                let frames = &frames;
                scope.spawn(move || {
                    let mut kept = Vec::new();
                    for round in 0..ROUNDS {
                        let function = Function {
                            name: format!("kernel_{t}_{round}"),
                            address: 0x10_0000 * (t as u64 + 1) + 0x100 * round as u64,
                            size: 0x40,
                            statements: Vec::new(),
                        };
                        let registration = Registration::new(&[function], frames).unwrap();
                        // Keep every third file and drop the rest at once, so
                        // removals interleave with other threads' insertions.
                        if round % 3 == 0 {
                            kept.push(registration);
                        }
                    }
                    kept
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    let total: usize = kept.iter().map(Vec::len).sum();
    // SAFETY: all registering threads have joined.
    assert_eq!(unsafe { entries() }, total);
    drop(kept);
    // SAFETY: as above.
    assert_eq!(unsafe { entries() }, 0);
}
