use crate::lir::x64::Cpu;
use crate::{
    hir::{Binary as B, Lane, Ternary, Type, Unary},
    lir::x64::inst::simd::{Native, lanewise::Lanewise, packed::Recipe},
    support::numeric::Action,
};

impl Native {
    pub(crate) fn select(action: Action, input: Type, output: Type, cpu: Cpu) -> Option<Self> {
        let lane = input.lane()?;
        let packed = matches!(input, Type::Vector(_));
        let bytes = if packed || matches!(action, Action::Splat) {
            cpu.bytes
        } else {
            lane.bytes()
        };
        let direct = Self {
            action,
            input,
            output,
            bytes,
            cpu,
            recipe: None,
        };
        if !packed && matches!(action, Action::Cast) {
            let to = output.lane()?;
            if !lane.float() && !to.float() && lane.bytes() <= 8 && to.bytes() <= 8 {
                return Some(direct);
            }
        }
        if packed && let Some(recipe) = Recipe::select(action, input, output, cpu) {
            return Some(Self {
                recipe: Some(recipe),
                ..direct
            });
        }
        if packed && supports_mask_op(action, lane.bytes(), cpu) {
            return Some(direct);
        }
        if matches!(lane, Lane::F16 | Lane::BF16) && !matches!(action, Action::Splat) {
            let supported = lane == Lane::F16
                && packed
                && cpu.isa.fp16
                && crate::lir::x64::inst::simd::evex(cpu)
                && matches!(
                    action,
                    Action::Binary(B::Add | B::Sub | B::Mul | B::Div)
                        | Action::Unary(Unary::Sqrt)
                        | Action::Ternary(Ternary::MulAdd)
                );
            return supported.then_some(direct);
        }
        if let Action::Binary(op) = action
            && let Some(form) = Lanewise::of(op, lane)
        {
            return form.supports(bytes, cpu).then_some(direct);
        }
        let int = packed && (bytes != 64 || lane.bytes() >= 4 || cpu.isa.bw);
        let supported = match action {
            Action::Splat => {
                lane.bytes() <= 8
                    && match bytes {
                        16 | 32 => true,
                        64 => lane.bytes() >= 4 || cpu.isa.bw,
                        _ => false,
                    }
            }
            Action::Compress | Action::Expand => {
                packed && crate::lir::x64::inst::simd::compact_lanes(cpu, lane)
            }
            Action::Unary(Unary::Abs) if lane.signed() => {
                int && match lane.bytes() {
                    1 | 2 | 4 => bytes != 16 || cpu.isa.ssse3,
                    8 => bytes == 64,
                    _ => false,
                }
            }
            Action::Binary(op @ (B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge))
                if !lane.float() =>
            {
                int && lane.bytes() <= 8
                    && (bytes == 64
                        || match op {
                            B::Eq => lane.bytes() != 8 || bytes == 32 || cpu.isa.sse41,
                            B::Gt => {
                                lane.signed() && (lane.bytes() != 8 || bytes == 32 || cpu.isa.sse42)
                            }
                            _ => false,
                        })
            }
            Action::Unary(Unary::Popcnt) => {
                packed
                    && bytes == 64
                    && match lane.bytes() {
                        1 | 2 => cpu.isa.bitalg,
                        4 | 8 => cpu.isa.vpopcnt,
                        _ => false,
                    }
            }
            Action::Unary(Unary::LeadingZeros) => {
                packed && bytes == 64 && cpu.isa.cd && matches!(lane.bytes(), 4 | 8)
            }
            Action::Unary(Unary::Floor | Unary::Ceil | Unary::Trunc | Unary::Nearest) => {
                lane.float() && (cpu.isa.sse41 || bytes >= 32)
            }
            Action::Binary(op @ (B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge))
                if packed && lane.float() =>
            {
                bytes != 16 || !matches!(op, B::Gt | B::Ge)
            }
            Action::Cast => {
                cpu.isa.avx10 >= 2
                    && matches!(
                        (input, output),
                        (Type::Vector(Lane::F32), Type::Vector(Lane::I32 | Lane::U32))
                            | (
                                Type::Vector(Lane::F64),
                                Type::Vector(Lane::I64 | Lane::U64 | Lane::Isize | Lane::Usize)
                            )
                    )
            }
            Action::Unary(Unary::Sqrt) => lane.float(),
            Action::Ternary(Ternary::MulAdd) => lane.float() && (cpu.isa.fma || bytes == 64),
            _ => false,
        };
        supported.then_some(direct)
    }
}

fn supports_mask_op(action: Action, lane_bytes: usize, cpu: Cpu) -> bool {
    let signs = match lane_bytes {
        1 | 2 => cpu.isa.bw,
        4 | 8 => true,
        _ => false,
    };
    match action {
        Action::AnyTrue | Action::Unary(Unary::Not) => true,
        // VPTESTMD and the 16-bit KNOTW/KORTESTW forms need only AVX-512F.
        // Narrower vectors compare with zero; PCMPEQQ needs SSE4.1.
        Action::AllTrue => match cpu.bytes {
            16 => lane_bytes < 8 || (lane_bytes == 8 && cpu.isa.sse41),
            32 => lane_bytes <= 8,
            _ => lane_bytes == 4 || signs,
        },
        Action::Ternary(Ternary::Select) => match cpu.bytes {
            16 => lane_bytes <= 8,
            32 => lane_bytes <= 8,
            _ => signs,
        },
        _ => false,
    }
}
