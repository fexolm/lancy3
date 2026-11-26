//! Constraints exposed by target instructions to architecture-independent LIR passes.
use regalloc2::{MachineEnv, Operand, PReg, PRegSet, RegClass};

/// A scratch register an instruction writes: any free register of a class,
/// or one the encoding fixes.
#[derive(Clone, Copy, Debug)]
pub(super) enum Temp {
    Any(RegClass),
    Fixed(PReg),
}

pub(super) trait TargetInst: Copy {
    /// What operands and clobbers depend on: the calling convention and the
    /// vector width the function is compiled for.
    type Machine: Copy;
    /// The function's table of instruction payloads too large to keep in
    /// the instruction itself, which holds an index into it.
    type Details: Default;
    /// Append one operand per value the instruction reads or writes; the
    /// emitter finds each allocation by its value (`regalloc::Allocations`).
    fn operands(&self, machine: &Self::Machine, details: &Self::Details, out: &mut Vec<Operand>);
    fn clobbers(&self, machine: &Self::Machine, details: &Self::Details) -> PRegSet;
    /// The scratch registers the emitter writes besides its outputs. The
    /// function adds a fresh temporary for each one after the instruction's
    /// operands, live from its start, so no input or output shares it; the
    /// emitter finds them with `Allocations::temp`.
    fn temps(&self, _machine: &Self::Machine) -> &'static [Temp] {
        &[]
    }
    /// The frame-local region this instruction addresses, if any.
    fn frame_use(&self, machine: &Self::Machine) -> Option<super::frame::FrameUse>;
    fn is_branch(&self) -> bool;
    fn is_ret(&self) -> bool;
    fn machine_env(machine: &Self::Machine) -> &'static MachineEnv;
    fn spillslot_size(machine: &Self::Machine, class: RegClass) -> usize;
}
