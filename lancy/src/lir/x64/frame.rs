//! The internal call ABI reserves RAX and RCX for instruction selection, RDX
//! for the workspace and the four caller-saved registers R8..R11 for values.
pub(super) const EAX: u8 = 0;
pub(super) const ECX: u8 = 1;
pub(super) const RDX: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Home {
    Register(u8),
    Slot(u32),
}
const _: () = assert!(std::mem::size_of::<Home>() <= 8);
pub(super) const LINK_SLOTS: usize = 1;

pub(super) struct Frame {
    pub args: Vec<Home>,
    pub slots: usize,
    pub status_slot: usize,
    pub save_base: usize,
}
