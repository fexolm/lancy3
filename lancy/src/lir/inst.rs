//! Constraints exposed by target instructions to architecture-independent LIR passes.
use regalloc2::{MachineEnv, Operand, PRegSet, RegClass};

pub(super) trait TargetInst: Copy {
    /// What operands and clobbers depend on: the calling convention and the
    /// vector width the function is compiled for.
    type Machine: Copy;
    /// Append operands in the same order in which the emitter consumes allocations.
    fn operands(&self, machine: &Self::Machine, out: &mut Vec<Operand>);
    fn clobbers(&self, machine: &Self::Machine) -> PRegSet;
    /// The frame-local region this instruction addresses, if any.
    fn frame_use(&self, machine: &Self::Machine) -> Option<super::frame::FrameUse>;
    fn is_branch(&self) -> bool;
    fn is_ret(&self) -> bool;
    fn machine_env(machine: &Self::Machine) -> &'static MachineEnv;
    fn spillslot_size(machine: &Self::Machine, class: RegClass) -> usize;
}
