use super::Rules;
use crate::{
    hir::{Binary, Lane, Op, Simd, Ternary, Value},
    lir::x64::inst::{Inst, VectorMask, simd::comparison_predicate},
};
use regalloc2::VReg;

impl Rules<'_> {
    pub(super) fn vector_mask(&mut self, value: Value) -> VectorMask<VReg> {
        self.comparison_mask(value)
            .unwrap_or_else(|| VectorMask::Lanes(self.input(value)))
    }

    pub(super) fn masked_numeric(&mut self, op: Op) -> Option<Inst> {
        match op {
            Op::Simd(Simd::Ternary(Ternary::Select, cond, yes, no)) => {
                Some(Inst::SelectComparison {
                    lane: self.ty(yes).lane()?,
                    mask: self.comparison_mask(cond)?,
                    yes: self.input(yes),
                    no: self.input(no),
                    dst: self.dst(),
                })
            }
            Op::Simd(Simd::AnyTrue(value)) => Some(Inst::AnyComparison(
                self.dst(),
                self.comparison_mask(value)?,
            )),
            _ => None,
        }
    }

    fn comparison_mask(&mut self, value: Value) -> Option<VectorMask<VReg>> {
        if self.cpu.bytes != 64 {
            return None;
        }
        let ((lane, predicate, lhs, rhs), active) = if let Some(cmp) = self.comparison(value) {
            (cmp, None)
        } else {
            // A comparison yields all ones or zero. AND with arbitrary bits
            // can use a writemask derived by the consumer, which knows if the
            // lane's sign bit or any nonzero bit makes it active.
            let Op::Simd(Simd::Bin(Binary::And, a, b)) = self.ctx.single_use(value)? else {
                return None;
            };
            [(a, b), (b, a)].into_iter().find_map(|(cmp, active)| {
                let cmp = self.comparison(cmp)?;
                // Keep floating comparisons unpredicated so the same lanes
                // participate in the comparison.
                (!cmp.0.float()).then_some((cmp, Some(active)))
            })?
        };
        Some(VectorMask::Compare {
            lane,
            predicate,
            lhs: self.input(lhs),
            rhs: self.input(rhs),
            active: active.map(|v| self.input(v)),
        })
    }

    fn comparison(&self, value: Value) -> Option<(Lane, i32, Value, Value)> {
        let mut op = self.ctx.single_use(value)?;
        if let Op::Simd(Simd::Bitcast(to, from)) = op {
            if to.bytes() != self.ty(from).lane()?.bytes() {
                return None;
            }
            op = self.ctx.single_use(from)?;
        }
        let Op::Simd(Simd::Bin(kind, lhs, rhs)) = op else {
            return None;
        };
        let lane = self.ty(lhs).lane()?;
        let supported = match lane.bytes() {
            1 | 2 => self.cpu.bw && !lane.float(),
            4 | 8 => true,
            _ => false,
        };
        supported.then_some((lane, comparison_predicate(lane, kind)?, lhs, rhs))
    }
}
