use super::inst::{Alu, Inst};
use crate::{
    hir::{Binary, Lane, operation::Action},
    lir::schedule::{Schedulable, Scheduling},
};

impl Schedulable for Inst {
    fn scheduling(&self) -> Scheduling {
        use Scheduling::{Barrier, Compute, Ordered};
        match *self {
            Self::Load(..) | Self::PackedMemory(..) | Self::BroadcastMemory(..) => Ordered(5),
            Self::Masked {
                dst: Some(_),
                indices,
                ..
            } => Ordered(if indices.is_some() { 9 } else { 5 }),
            Self::CheckBounds(..)
            | Self::TrapIf(..)
            | Self::Trap(_)
            | Self::CheckMask(..)
            | Self::Store(..)
            | Self::Masked { .. }
            | Self::CompactStore { .. } => Ordered(1),
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
            Self::Alu(Alu::Mul, ..) => Compute(3),
            Self::Native(native, ..) => match native.action {
                Action::Binary(Binary::Div) if native.input.lane().is_some_and(Lane::float) => {
                    Compute(12)
                }
                Action::Binary(Binary::Div | Binary::Rem) => Barrier,
                Action::Binary(Binary::Mul) | Action::Ternary(_) => Compute(4),
                _ => Compute(if native.separate_output() { 4 } else { 1 }),
            },
            Self::DivImm { .. } => Compute(5),
            Self::BroadcastLane(..) | Self::Iota(..) | Self::SelectI32 { .. } => Compute(3),
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
            | Self::Lea(..) => Compute(1),
            _ => Barrier,
        }
    }
}
