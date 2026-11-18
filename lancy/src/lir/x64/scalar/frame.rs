//! The internal call ABI reserves RAX and RCX for instruction selection, RDX
//! for the workspace and the four caller-saved registers R8..R11 for values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Home {
    Register(u8),
    Slot(u32),
    Immediate(i32),
    Dead,
}
const _: () = assert!(std::mem::size_of::<Home>() <= 8);
pub(super) const LINK_SLOTS: usize = 2;
pub(crate) fn buffer_header_words(count: usize) -> usize {
    2 * (1 + count)
}

pub(super) struct Frame {
    pub args: Vec<Home>,
    pub has_status: bool,
    pub slots: usize,
    pub status_slot: usize,
    pub save_base: usize,
}

impl Frame {
    pub(super) fn buffer_slot(&self, buffer: usize) -> usize {
        self.status_slot + 2 + 2 * buffer
    }
}
