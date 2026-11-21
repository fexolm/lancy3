use super::Rules;
use crate::{
    hir::operation::Action,
    hir::{Binary as B, Lane, Op, Simd, Ternary, Type, Unary},
    lir::x64::inst::{Inst, simd::packed::integer::Rule},
};
use splice_target::Features;

impl Rules<'_> {
    pub(super) fn immediate_vector_shift(&mut self, op: Op) -> Option<Inst> {
        let Op::Simd(Simd::Bin(
            kind @ (B::Shl | B::Shr | B::RotateLeft | B::RotateRight),
            src,
            count,
        )) = op
        else {
            return None;
        };
        let lane = self.ty(src).lane()?;
        if !matches!(lane, Lane::I32 | Lane::U32) || (self.cpu.bytes == 32 && !self.cpu.avx2) {
            return None;
        }
        let Op::Simd(Simd::Splat(scalar)) = self.ctx.defining_op(count)? else {
            return None;
        };
        let count = (self.dword_constant(scalar)? & 31) as u8;
        let dst = self.dst();
        let src = self.input(src);
        Some(if count == 0 {
            Inst::Copy(dst, src)
        } else {
            Inst::VectorShiftImm(kind, lane, dst, src, count)
        })
    }
}

impl Rule {
    pub(in crate::lir::x64) fn select(action: Action, input: Type, c: Features) -> Option<Self> {
        let lane = input.lane()?;
        let n = lane.bytes();
        if matches!(action, Action::Ternary(Ternary::BitSelect)) {
            return Some(Self::BitSelect);
        }
        if lane.float() || n > 8 || c.bytes == 32 && !c.avx2 || c.bytes == 64 && n < 4 && !c.bw {
            return None;
        }
        Some(match action {
            Action::Unary(Unary::Neg) => Self::Neg,
            Action::Unary(Unary::Abs)
                if !lane.signed() || (n == 8 && c.bytes != 64) || (c.bytes == 16 && !c.ssse3) =>
            {
                Self::Abs
            }
            Action::Binary(B::Mul)
                if n == 1
                    || n == 8 && !(c.bytes == 64 && c.dq)
                    || n == 4 && c.bytes == 16 && !c.sse41 =>
            {
                Self::Mul
            }
            Action::Binary(op @ (B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge))
                if c.bytes != 64
                    && !matches!(
                        (op, n, lane.signed()),
                        (B::Eq, 1 | 2 | 4, _) | (B::Gt, 1 | 2 | 4, true)
                    ) =>
            {
                Self::Cmp(op)
            }
            Action::Binary(op @ (B::Min | B::Max))
                if c.bytes != 64
                    && (n == 8
                        || c.bytes == 16
                            && !c.sse41
                            && !matches!((n, lane.signed()), (1, false) | (2, true))) =>
            {
                Self::MinMax(op)
            }
            Action::Binary(op @ (B::Shl | B::Shr | B::RotateLeft | B::RotateRight))
                if c.bytes >= 32 && (n >= 4 || n == 2 && c.bytes == 64) =>
            {
                Self::Shift(op)
            }
            Action::Unary(Unary::TrailingZeros)
                if c.bytes == 64 && (n >= 4 && c.vpopcnt || n <= 2 && c.bitalg) =>
            {
                Self::Ctz
            }
            Action::Binary(op @ (B::SaturatingAdd | B::SaturatingSub)) if n >= 4 => {
                Self::Saturate(op)
            }
            _ => return None,
        })
    }
}
