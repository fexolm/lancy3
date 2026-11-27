use super::inst::{Alu, Details, Inst, convert};
use crate::{
    hir::{Binary, Lane},
    lir::schedule::{Schedulable, Scheduling},
    support::numeric::Action,
};

impl Schedulable for Inst {
    fn scheduling(&self, details: &Details) -> Scheduling {
        use Scheduling::{Barrier, Compute, Load, Ordered};
        match *self {
            Self::Load(..) | Self::PackedMemory(..) | Self::BroadcastMemory(..) => Load(5),
            Self::Masked(id) => {
                let access = details.masked[id];
                match (access.dst, access.index) {
                    (Some(_), Some(_)) => Load(9),
                    (Some(_), None) => Load(5),
                    (None, _) => Ordered(1),
                }
            }
            Self::CheckBounds(..)
            | Self::TrapIf(..)
            | Self::Trap(_)
            | Self::CheckMask(..)
            | Self::Store(..)
            | Self::CompactStore { .. }
            | Self::StoreArg(..) => Ordered(1),
            Self::DynamicLane { checked: true, .. } => Ordered(4),
            Self::DynamicLane { .. } | Self::CompactLookup(..) => Compute(5),
            Self::MaskCompare(..)
            | Self::MaskCompact(..)
            | Self::PairedShuffle(..)
            | Self::MaskPrefix(..)
            | Self::MaskCount(..)
            | Self::AnyMask(..)
            | Self::AllMask(..)
            | Self::SelectMask(_) => Compute(3),
            Self::Alu(Alu::Mul, ..) => Compute(3),
            Self::IntUnary(..) | Self::Saturating { .. } | Self::Classify(..) => Compute(3),
            // Soft casts write fixed registers.
            Self::Convert { from, to, .. } if convert::soft(from, to) => Barrier,
            Self::Convert { .. } => Compute(5),
            Self::Native(id) => {
                let native = details.natives[id].native;
                match native.action {
                    Action::Binary(Binary::Div) if native.input.lane().is_some_and(Lane::float) => {
                        Compute(12)
                    }
                    Action::Binary(Binary::Div | Binary::Rem) => Barrier,
                    Action::Binary(Binary::Mul) | Action::Ternary(_) => Compute(4),
                    _ => Compute(if native.separate_output() { 4 } else { 1 }),
                }
            }
            Self::DivImm { .. } => Compute(5),
            Self::BroadcastLane(..)
            | Self::ExtractLane(..)
            | Self::Iota(..)
            | Self::SplatLiteral(..)
            | Self::Select { .. }
            | Self::SelectVector { .. } => Compute(3),
            Self::Imm(..)
            | Self::Literal(..)
            | Self::Copy(..)
            | Self::BitMove(..)
            | Self::Alu(..)
            | Self::Compare(..)
            | Self::Shift { .. }
            | Self::ShiftDwords(..)
            | Self::VectorShiftImm(..)
            | Self::MaskLogic(..)
            | Self::MaskNot(..)
            | Self::MaskImmediate(..)
            | Self::Extend(..)
            | Self::Lea(..)
            | Self::CallScratch(..) => Compute(1),
            // Hardware division faults, and its high half is in a fixed register;
            // calls use stack areas and fixed registers. Entry arguments,
            // debug state and terminators keep their place.
            Self::Divide { .. }
            | Self::DivideHigh { .. }
            | Self::Fallback(_)
            | Self::Call { .. }
            | Self::RustCall(..)
            | Self::Argv(_)
            | Self::LoadArg { .. }
            | Self::DebugStore(..)
            | Self::DebugPoint(_)
            | Self::BrCmp(..)
            | Self::BranchAny(_)
            | Self::BranchAll(..)
            | Self::Jump
            | Self::Branch(_)
            | Self::Ret(_) => Barrier,
        }
    }
}
