use crate::hir::function::validate_buffers;
use crate::hir::{BufferArg, BufferKind};
use std::mem::MaybeUninit;

/// Read by the x64 emitter. Pointers live for one synchronous invocation.
#[repr(C)]
pub(super) struct Descriptor {
    data: *mut i32,
    len: usize,
}

const _: () = {
    assert!(std::mem::size_of::<*const ()>() == 2 * std::mem::size_of::<i32>());
    assert!(std::mem::offset_of!(Descriptor, data) == 0);
    assert!(std::mem::offset_of!(Descriptor, len) == 8);
};

// Moving a caller transfers its exclusive workspace. Descriptors are populated
// from the current borrows immediately before each call and are never exposed.
unsafe impl Send for Descriptor {}
unsafe impl Sync for Descriptor {}

pub(super) fn descriptors(
    kinds: &[BufferKind],
    buffers: &mut [BufferArg<'_>],
    views: &mut Vec<Descriptor>,
) -> Result<(), String> {
    validate_buffers(kinds, buffers)?;
    views.clear();
    for buffer in buffers {
        let (data, len) = match buffer {
            BufferArg::Read(s) => (s.as_ptr().cast_mut(), s.len()),
            BufferArg::Write(s) => (s.as_mut_ptr(), s.len()),
        };
        views.push(Descriptor { data, len });
    }
    Ok(())
}

pub(super) fn prepare_header(
    scratch: &mut [MaybeUninit<i32>],
    status: &mut i32,
    views: &[Descriptor],
) {
    // Root frame starts with a status pointer followed by one descriptor pointer
    // per buffer. A scratch word is only 4-byte aligned, even on x64.
    assert!(scratch.len() >= crate::lir::buffer_header_words(views.len()));
    let pointers = scratch.as_mut_ptr().cast::<*const ()>();
    // SAFETY: the assertion covers every 8-byte write. The descriptor vector
    // stays fixed and all originating slices are borrowed throughout execution.
    unsafe {
        pointers.write_unaligned(std::ptr::from_mut(status).cast());
        for (index, view) in views.iter().enumerate() {
            pointers
                .add(index + 1)
                .write_unaligned(std::ptr::from_ref(view).cast());
        }
    }
}
