use super::inst::{Alu, Inst};
use crate::{
    hir::{Binary, Lane},
    lir::schedule::{Schedulable, Scheduling},
    support::numeric::Action,
};

impl Schedulable for Inst {
    fn scheduling(&self) -> Scheduling {
        use Scheduling::{Barrier, Compute, Load, Ordered};
        match *self {
            Self::Load(..) | Self::PackedMemory(..) | Self::BroadcastMemory(..) => Load(5),
            Self::Masked {
                dst: Some(_),
                indices,
                ..
            } => Load(if indices.is_some() { 9 } else { 5 }),
            Self::CheckBounds(..)
            | Self::TrapIf(..)
            | Self::Trap(_)
            | Self::CheckMask(..)
            | Self::Store(..)
            | Self::Masked { dst: None, .. }
            | Self::CompactStore { .. }
            | Self::StoreArg(..) => Ordered(1),
            Self::DwordLane { checked: true, .. } => Ordered(4),
            Self::DwordLane { .. } | Self::CompactLookup(..) => Compute(5),
            Self::MaskCompare(..)
            | Self::MaskCompact(..)
            | Self::PairedShuffle(..)
            | Self::MaskPrefix(..)
            | Self::MaskCount(..)
            | Self::AnyMask(..)
            | Self::AllMask(..)
            | Self::SelectMask { .. } => Compute(3),
            Self::Alu(Alu::Mul, ..) | Self::IntBinary(Binary::Mul, ..) => Compute(3),
            Self::Native(native, ..) => match native.action {
                Action::Binary(Binary::Div) if native.input.lane().is_some_and(Lane::float) => {
                    Compute(12)
                }
                Action::Binary(Binary::Div | Binary::Rem) => Barrier,
                Action::Binary(Binary::Mul) | Action::Ternary(_) => Compute(4),
                _ => Compute(if native.separate_output() { 4 } else { 1 }),
            },
            Self::DivImm { .. } => Compute(5),
            Self::BroadcastLane(..)
            | Self::ExtractLane(..)
            | Self::Iota(..)
            | Self::SelectI32 { .. } => Compute(3),
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
            | Self::Movsxd(..)
            | Self::Lea(..)
            | Self::IntBinary(..)
            | Self::CallScratch(..) => Compute(1),
            // Hardware division faults, and CDQ feeds it in fixed registers;
            // inline lane expansions and calls use
            // stack areas and fixed registers. Entry arguments, debug state
            // and terminators keep their place.
            Self::IDiv { .. }
            | Self::Cdq { .. }
            | Self::Lanes { .. }
            | Self::Fallback { .. }
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
