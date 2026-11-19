//! R11 points to a frame throughout allocated code. The host supplies the
//! status pointer in root[0]. A child starts with its continuation and a
//! pointer to the root, followed by incoming arguments and private spills.
//! Siblings reuse the same space immediately above their caller.
pub(super) const CHILD_ARGS: usize = 2;
pub(super) struct Frame {
    pub spill_base: usize,
    pub slots: usize,
}

impl Frame {
    pub fn new(root: bool, arity: usize, spills: usize) -> Self {
        let spill_base = if root { 1 } else { CHILD_ARGS + arity };
        Self {
            spill_base,
            slots: spill_base + spills,
        }
    }
}

// Win64 preserves the low 128 bits of XMM6..15. GPR saves occupy one word.
pub(super) fn save_words(reg: regalloc2::PReg) -> usize {
    if reg.class() == regalloc2::RegClass::Vector {
        2
    } else {
        1
    }
}
