use super::super::Entry;

const CANARY: u64 = 0x1234_abcd_9876_4000;

/// Enter `entry` with canaries in x19..x28 and the low halves of d8..d15.
/// Bit `i` of the result is set if the i-th of those 18 registers changed.
pub(super) fn call_with_canaries(entry: Entry, args: *const u64, workspace: *mut u64) -> u64 {
    let changed: u64;
    // SAFETY: the block saves every callee-saved register it overwrites in
    // its own 16-byte aligned stack area and restores them before leaving.
    // The kernel entry follows AAPCS64; `clobber_abi("C")` covers the
    // registers it may change, and the caller vouches for `args`/`workspace`.
    unsafe {
        std::arch::asm!(
            "sub sp, sp, #160",
            "stp x19, x20, [sp, #0]",
            "stp x21, x22, [sp, #16]",
            "stp x23, x24, [sp, #32]",
            "stp x25, x26, [sp, #48]",
            "stp x27, x28, [sp, #64]",
            "stp d8, d9, [sp, #80]",
            "stp d10, d11, [sp, #96]",
            "stp d12, d13, [sp, #112]",
            "stp d14, d15, [sp, #128]",
            "str x10, [sp, #144]",
            "add x19, x10, #0",
            "add x20, x10, #1",
            "add x21, x10, #2",
            "add x22, x10, #3",
            "add x23, x10, #4",
            "add x24, x10, #5",
            "add x25, x10, #6",
            "add x26, x10, #7",
            "add x27, x10, #8",
            "add x28, x10, #9",
            "add x11, x10, #10", "fmov d8, x11",
            "add x11, x10, #11", "fmov d9, x11",
            "add x11, x10, #12", "fmov d10, x11",
            "add x11, x10, #13", "fmov d11, x11",
            "add x11, x10, #14", "fmov d12, x11",
            "add x11, x10, #15", "fmov d13, x11",
            "add x11, x10, #16", "fmov d14, x11",
            "add x11, x10, #17", "fmov d15, x11",
            "blr x9",
            "ldr x10, [sp, #144]",
            "mov x0, #0",
            ".irp n, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28",
            "add x11, x10, #(\\n - 19)",
            "cmp x\\n, x11",
            "cset x12, ne",
            "orr x0, x0, x12, lsl #(\\n - 19)",
            ".endr",
            ".irp n, 8, 9, 10, 11, 12, 13, 14, 15",
            "fmov x13, d\\n",
            "add x11, x10, #(\\n + 2)",
            "cmp x13, x11",
            "cset x12, ne",
            "orr x0, x0, x12, lsl #(\\n + 2)",
            ".endr",
            "ldp x19, x20, [sp, #0]",
            "ldp x21, x22, [sp, #16]",
            "ldp x23, x24, [sp, #32]",
            "ldp x25, x26, [sp, #48]",
            "ldp x27, x28, [sp, #64]",
            "ldp d8, d9, [sp, #80]",
            "ldp d10, d11, [sp, #96]",
            "ldp d12, d13, [sp, #112]",
            "ldp d14, d15, [sp, #128]",
            "add sp, sp, #160",
            inout("x0") args => changed,
            in("x1") workspace,
            inout("x9") entry => _,
            inout("x10") CANARY => _,
            clobber_abi("C"),
        );
    }
    changed
}
