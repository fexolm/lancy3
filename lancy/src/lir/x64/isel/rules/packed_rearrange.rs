use crate::{
    hir::operation::Action,
    hir::{Binary as B, Lane, Type},
    lir::x64::inst::simd::packed::rearrange::Rule,
};
use splice_target::Features;

impl Rule {
    pub(in crate::lir::x64) fn select(
        action: Action,
        input: Type,
        output: Type,
        c: Features,
    ) -> Option<Self> {
        let e = input.lane()?;
        let to = output.lane()?;
        let n = e.bytes();
        Some(match action {
            Action::Widen(_) | Action::Narrow
                if matches!((e, to), (Lane::F32, Lane::F64) | (Lane::F64, Lane::F32))
                    || matches!((e, to), (Lane::F32, Lane::F16) | (Lane::F16, Lane::F32))
                        && (c.bytes == 64 || c.f16c)
                    || matches!((e, to), (Lane::F32, Lane::BF16) | (Lane::BF16, Lane::F32))
                        && (c.bytes == 16 || c.avx2) =>
            {
                Self::FloatWidth
            }
            Action::Shuffle
                if c.bytes == 64
                    && (matches!(n, 4 | 8) || n == 2 && c.bw || n == 1 && c.bw && c.vbmi)
                    || c.bytes == 32 && c.avx2 && n == 4 =>
            {
                Self::Shuffle
            }
            Action::Widen(high)
                if !e.float()
                    && !to.float()
                    && to.bytes() == 2 * n
                    && to.bytes() <= 8
                    && (c.bytes == 16
                        || c.bytes == 32 && c.avx2
                        || c.bytes == 64 && (n >= 2 || c.bw)) =>
            {
                Self::Widen(high)
            }
            Action::Narrow
                if !e.float()
                    && !to.float()
                    && n == 2 * to.bytes()
                    && n <= 8
                    && (c.bytes == 16
                        || c.bytes == 32 && c.avx2
                        || c.bytes == 64 && (n >= 4 || c.bw)) =>
            {
                Self::Narrow
            }
            Action::Reduce(op)
                if !e.float()
                    && n <= 8
                    && (c.bytes == 16 || c.avx2)
                    && match op {
                        B::Add | B::And | B::Or | B::Xor => true,
                        B::Mul => n == 2 || n == 4 && c.sse41,
                        B::Min | B::Max => {
                            n <= 4 && (c.sse41 || matches!((n, e.signed()), (1, false) | (2, true)))
                        }
                        _ => false,
                    } =>
            {
                Self::Reduce(op)
            }
            _ => return None,
        })
    }
}
