use crate::lir::x64::Cpu;
use crate::{
    hir::operation::Action,
    hir::{Binary as B, Type, Unary},
    lir::x64::inst::simd::{
        Native,
        packed::{Recipe, float, integer, rearrange},
    },
};

impl Recipe {
    pub fn select(action: Action, input: Type, output: Type, c: Cpu) -> Option<Self> {
        integer::Rule::select(action, input, c)
            .map(Self::Int)
            .or_else(|| float::Rule::select(action, input, output, c).map(Self::Float))
            .or_else(|| rearrange::Rule::select(action, input, output, c).map(Self::Rearrange))
            .or_else(|| {
                let lane = input.lane()?;
                if lane.float()
                    || lane.bytes() > 8
                    || !(c.bytes == 32 && !c.isa.avx2
                        || c.bytes == 64 && !c.isa.bw && lane.bytes() < 4 && c.isa.avx2)
                {
                    return None;
                }
                match action {
                    // Logical operations already have full-width AVX forms.
                    Action::Binary(B::And | B::Or | B::Xor) | Action::Unary(Unary::Not) => None,
                    Action::Unary(_) | Action::Binary(_) => {
                        packed_half(action, input, output, c).map(|_| Self::Halves)
                    }
                    _ => None,
                }
            })
    }
}

pub(in crate::lir::x64) fn packed_half(
    action: Action,
    input: Type,
    output: Type,
    c: Cpu,
) -> Option<Native> {
    Native::select(
        action,
        input,
        output,
        Cpu {
            bytes: c.bytes / 2,
            ..c
        },
    )
}
