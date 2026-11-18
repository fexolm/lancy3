//! Constraints exposed by target instructions to architecture-independent LIR passes.
use regalloc2::{MachineEnv, Operand, PRegSet, RegClass};

pub(super) trait TargetInst {
    /// Append operands in the same order in which the emitter consumes allocations.
    fn operands(&self, out: &mut Vec<Operand>);
    fn clobbers(&self) -> PRegSet;
    fn is_branch(&self) -> bool;
    fn is_ret(&self) -> bool;
    fn machine_env() -> &'static MachineEnv;
    fn spillslot_size(class: RegClass) -> usize;
}
