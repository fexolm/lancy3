//! The machine-code layout of debug builds, for debugger and unwinder tables.
//!
//! Each compiled body of a debug build ([`CompileOptions::with_debug`]) is
//! described by a [`FunctionInfo`]: where its code lies, which machine
//! offsets carry its `debug_point` markers, where its debug slots live and
//! how to unwind its frame. Offsets count bytes from
//! [`Kernel::code_address`]; DWARF register numbers follow the target's
//! psABI.
//!
//! [`CompileOptions::with_debug`]: crate::CompileOptions::with_debug
//! [`Kernel::code_address`]: crate::Kernel::code_address
use std::ops::Range;

/// The machine-code layout of one compiled body.
#[derive(Debug)]
pub struct FunctionInfo {
    pub(crate) code: Range<u64>,
    pub(crate) slots: usize,
    pub(crate) points: Vec<Point>,
    pub(crate) frame: FrameInfo,
}

impl FunctionInfo {
    /// The body's code, in bytes from the kernel's code address.
    pub fn code(&self) -> Range<u64> {
        self.code.clone()
    }

    /// Byte offset of the body's `debug_store` slots from the workspace
    /// register (`r11` on x86-64, `x19` on AArch64).
    pub fn slots_offset(&self) -> usize {
        self.slots
    }

    /// The `debug_point` markers of the body, in code order. The first one
    /// is the body's own marker.
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// How to unwind the body's active frame.
    pub fn frame(&self) -> &FrameInfo {
        &self.frame
    }
}

/// A `debug_point N` marker and the code offset where it takes effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    /// The marker's operand `N`.
    pub id: u32,
    /// Bytes from the kernel's code address.
    pub offset: u64,
}

/// Stack-based unwind rules for the active portion of a body: after the
/// prologue and before the epilogue.
#[derive(Debug)]
pub struct FrameInfo {
    pub(crate) code: Range<u64>,
    pub(crate) cfa_from_sp: i32,
    pub(crate) saved: Vec<(u16, i32)>,
    pub(crate) workspace_saved: Vec<(u16, i64)>,
}

impl FrameInfo {
    /// The code the rules hold for, in bytes from the kernel's code address.
    pub fn code(&self) -> Range<u64> {
        self.code.clone()
    }

    /// The canonical frame address, in bytes above the stack pointer.
    pub fn cfa_from_sp(&self) -> i32 {
        self.cfa_from_sp
    }

    /// Registers saved on the stack: (DWARF register, byte offset from the
    /// canonical frame address).
    pub fn saved(&self) -> &[(u16, i32)] {
        &self.saved
    }

    /// Registers saved in the workspace: (DWARF register, byte offset from
    /// the workspace register).
    pub fn workspace_saved(&self) -> &[(u16, i64)] {
        &self.workspace_saved
    }
}
