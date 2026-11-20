use crate::{
    hir::operation::Action,
    hir::{Binary as B, Lane, Ternary, Type, Unary},
    lir::x64::inst::simd::{Native, packed::Recipe},
};
use splice_target::Features;

impl Native {
    pub fn select(action: Action, input: Type, output: Type, cpu: Features) -> Option<Self> {
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
        if matches!(lane, Lane::F16 | Lane::BF16) {
            let supported = lane == Lane::F16
                && packed
                && cpu.fp16
                && (bytes == 64 || (bytes == 32 && cpu.avx10 >= 1))
                && matches!(
                    action,
                    Action::Binary(B::Add | B::Sub | B::Mul | B::Div)
                        | Action::Unary(Unary::Sqrt)
                        | Action::Ternary(Ternary::MulAdd)
                );
            return supported.then_some(direct);
        }
        let int =
            packed && (bytes != 32 || cpu.avx2) && (bytes != 64 || lane.bytes() >= 4 || cpu.bw);
        let supported = match action {
            Action::Splat => {
                lane.bytes() <= 8
                    && match bytes {
                        16 => true,
                        32 => cpu.avx2,
                        64 => lane.bytes() >= 4 || cpu.bw,
                        _ => false,
                    }
            }
            Action::Compress | Action::Expand => {
                bytes == 64
                    && packed
                    && match lane.bytes() {
                        1 | 2 => cpu.vbmi2 && cpu.bw,
                        4 | 8 => true,
                        _ => false,
                    }
            }
            Action::Unary(Unary::Abs) if lane.signed() => {
                int && match lane.bytes() {
                    1 | 2 | 4 => bytes != 16 || cpu.ssse3,
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
                            B::Eq => lane.bytes() != 8 || bytes == 32 || cpu.sse41,
                            B::Gt => {
                                lane.signed() && (lane.bytes() != 8 || bytes == 32 || cpu.sse42)
                            }
                            _ => false,
                        })
            }
            Action::Unary(Unary::Popcnt) => {
                packed
                    && bytes == 64
                    && match lane.bytes() {
                        1 | 2 => cpu.bitalg,
                        4 | 8 => cpu.vpopcnt,
                        _ => false,
                    }
            }
            Action::Unary(Unary::LeadingZeros) => {
                packed && bytes == 64 && cpu.cd && matches!(lane.bytes(), 4 | 8)
            }
            Action::Unary(Unary::Floor | Unary::Ceil | Unary::Trunc | Unary::Nearest) => {
                lane.float() && (cpu.sse41 || bytes >= 32)
            }
            Action::Binary(B::SaturatingAdd | B::SaturatingSub) => int && lane.bytes() <= 2,
            Action::Binary(B::Min | B::Max) => {
                int && !lane.float()
                    && match lane.bytes() {
                        1 => !lane.signed() || bytes != 16 || cpu.sse41,
                        2 => lane.signed() || bytes != 16 || cpu.sse41,
                        4 => bytes != 16 || cpu.sse41,
                        8 => bytes == 64,
                        _ => false,
                    }
            }
            Action::Binary(op @ (B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge))
                if packed && lane.float() =>
            {
                bytes != 16 || !matches!(op, B::Gt | B::Ge)
            }
            Action::Cast => {
                cpu.avx10 >= 2
                    && matches!(
                        (input, output),
                        (Type::Vector(Lane::F32), Type::Vector(Lane::I32 | Lane::U32))
                            | (
                                Type::Vector(Lane::F64),
                                Type::Vector(Lane::I64 | Lane::U64 | Lane::Isize | Lane::Usize)
                            )
                    )
            }
            Action::Binary(B::Add | B::Sub) => lane.float() || (int && lane.bytes() <= 8),
            Action::Binary(B::Mul) => {
                lane.float()
                    || (int
                        && match lane.bytes() {
                            2 => true,
                            4 => bytes != 16 || cpu.sse41,
                            8 => bytes == 64 && cpu.dq,
                            _ => false,
                        })
            }
            Action::Binary(B::Div) | Action::Unary(Unary::Sqrt) => lane.float(),
            Action::Binary(B::And | B::Or | B::Xor) => packed,
            Action::Ternary(Ternary::MulAdd) => lane.float() && (cpu.fma || bytes == 64),
            _ => false,
        };
        supported.then_some(direct)
    }
}

fn supports_mask_op(action: Action, lane_bytes: usize, cpu: Features) -> bool {
    let signs = match lane_bytes {
        1 | 2 => cpu.bw,
        4 | 8 => true,
        _ => false,
    };
    match action {
        Action::AnyTrue | Action::Unary(Unary::Not) => true,
        // VPTESTMD and the 16-bit KNOTW/KORTESTW forms need only AVX-512F.
        Action::AllTrue => cpu.bytes == 64 && (lane_bytes == 4 || signs),
        Action::Ternary(Ternary::Select) => match cpu.bytes {
            16 => lane_bytes <= 8,
            32 => lane_bytes <= 8 && (cpu.avx2 || lane_bytes >= 4),
            _ => signs,
        },
        _ => false,
    }
}
