//! Execute an isolated instruction-selection test inside the usual RX mapping.
use super::memory::Executable;

/// # Safety
/// The bytes must implement the host C ABI for one pointer to N 64-byte
/// slots, preserve nonvolatile registers and use only host-supported features.
pub(crate) unsafe fn with_kernel<const N: usize>(
    code: &[u8],
    run: impl FnOnce(&dyn Fn(&mut [[u8; 64]; N])),
) {
    let memory = Executable::new(code).unwrap();
    // SAFETY: the test emitter supplies the ABI above; the mapping stays live
    // until the callback returns, and the callback cannot retain the closure.
    let entry: unsafe extern "C" fn(*mut [[u8; 64]; N]) =
        unsafe { std::mem::transmute(memory.address.as_ptr()) };
    // SAFETY: as above; `slots` is the exclusive slot array of one call.
    run(&|slots| unsafe { entry(slots) });
}
