use crate::lir::x64::Cpu;
use crate::{
    hir::operation::Action,
    hir::{Binary as B, Type, Unary},
    lir::x64::inst::simd::packed::float::Rule,
};

impl Rule {
    pub(in crate::lir::x64) fn select(
        action: Action,
        input: Type,
        output: Type,
        c: Cpu,
    ) -> Option<Self> {
        let e = input.lane()?;
        let to = output.lane()?;
        if matches!(action, Action::Cast) && e.bytes() == to.bytes() && e.float() != to.float() {
            let int = if e.float() { to } else { e };
            let w = int.bytes();
            if matches!(w, 4 | 8)
                && (w == 4 && int.signed()
                    || (c.bytes == 64 || c.isa.avx10 >= 1) && (w == 4 || c.isa.dq))
                && (!e.float() || c.isa.avx10 < 2)
            {
                return Some(Self::Convert);
            }
        }
        if !e.float() {
            return None;
        }
        Some(match action {
            Action::Unary(u @ (Unary::Abs | Unary::Neg)) => Self::Sign(u),
            Action::Binary(B::CopySign) => Self::CopySign,
            _ if e.bytes() < 4 => return None,
            Action::Binary(B::Ordered) => Self::Compare(7, false),
            Action::Binary(B::Unordered) => Self::Compare(3, false),
            Action::Binary(B::Gt) if c.bytes == 16 => Self::Compare(1, true),
            Action::Binary(B::Ge) if c.bytes == 16 => Self::Compare(2, true),
            Action::Unary(
                u @ (Unary::IsNan
                | Unary::IsNormal
                | Unary::IsInfinite
                | Unary::IsFinite
                | Unary::IsSubnormal
                | Unary::SignBit),
            ) => Self::Class(u),
            Action::Unary(Unary::Recip) => Self::Recip,
            Action::Unary(Unary::Fract) if c.bytes >= 32 || c.isa.sse41 => Self::Fract,
            Action::Unary(Unary::Round) if c.bytes >= 32 || c.isa.sse41 => Self::Round,
            Action::Binary(op @ (B::Min | B::Max)) => Self::MinMax(op),
            _ => return None,
        })
    }
}
