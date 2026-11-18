//! Low-level instructions for one iteration of a column kernel. Both scalar
//! and SIMD loops consume the same stream. Registers and spills are assigned
//! during lowering; immediate values need no storage in the frame.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Operand {
    VectorReg(u8),
    Spill(u32),
    Immediate(i32),
}

#[derive(Clone, Copy)]
pub(super) enum Binary {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Eq,
    Gt,
}

#[derive(Clone, Copy)]
pub(super) enum Action {
    Load(u16),
    Binary(Binary, Operand, Operand),
    Select(Operand, Operand, Operand),
}

pub(super) struct Step {
    pub(super) dest: Operand,
    pub(super) action: Action,
}

pub(super) struct Batch {
    pub(super) steps: Vec<Step>,
    pub(super) result: Operand,
    pub(super) spill_slots: usize,
}
