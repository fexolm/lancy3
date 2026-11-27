use super::super::masks::{MaskForm, mask_form};
use super::Rules;
use crate::{
    CompileError,
    hir::{Binary, Lane, Op, Simd, Ternary, Type, Value},
    lir::x64::inst::{Inst, MaskSelect, VectorMask, simd::comparison_predicate},
};
use regalloc2::{RegClass, VReg};

impl Rules<'_> {
    pub(super) fn vector_mask(&mut self, value: Value) -> VectorMask<VReg> {
        self.opmask(value)
            .or_else(|| self.comparison_mask(value))
            .unwrap_or_else(|| VectorMask::Lanes(self.input(value)))
    }

    pub(super) fn masked_numeric(&mut self, op: Op) -> Option<Inst> {
        if let Some(result) = self.result.filter(|&v| self.masks.contains(v)) {
            let lane = |v| self.ty(v).lane().unwrap();
            let constant = |v| match self.ctx.defining_op(v)? {
                Op::Const(n) => Some(u128::from(n as u32)),
                Op::Literal(_, bits) => Some(bits),
                _ => None,
            };
            match mask_form(&op, lane(result), lane, constant) {
                Some(MaskForm::Logic(kind, x, y)) => {
                    return Some(Inst::MaskLogic(
                        lane(x),
                        kind,
                        self.dst(),
                        self.input(x),
                        self.input(y),
                    ));
                }
                Some(MaskForm::Compare(lane, predicate, x, y)) => {
                    return Some(Inst::MaskCompare(
                        lane,
                        predicate,
                        self.dst(),
                        self.input(x),
                        self.input(y),
                    ));
                }
                Some(MaskForm::Not(x)) => {
                    return Some(Inst::MaskNot(lane(x), self.dst(), self.input(x)));
                }
                Some(MaskForm::Constant(all)) => {
                    let Op::Simd(Simd::Splat(x)) = op else {
                        unreachable!("a constant predicate is a splat")
                    };
                    let bits = if all {
                        u64::MAX >> (64 - self.cpu.bytes / lane(x).bytes())
                    } else {
                        0
                    };
                    return Some(Inst::MaskImmediate(self.dst(), bits));
                }
                Some(MaskForm::Copy(_)) | None => {}
            }
        }
        match op {
            // A scalar select has no lanes for a predicate register.
            Op::Ternary(Ternary::Select, cond, yes, no)
                if matches!(self.ty(yes), Type::Vector(_)) =>
            {
                let select = MaskSelect {
                    lane: self.ty(yes).lane()?,
                    mask: self.opmask(cond).or_else(|| self.comparison_mask(cond))?,
                    yes: self.input(yes),
                    no: self.input(no),
                    dst: self.dst(),
                };
                Some(self.body.details_mut().select(select))
            }
            Op::Simd(Simd::AnyTrue(value)) => Some(Inst::AnyMask(
                self.dst(),
                self.opmask(value).or_else(|| self.comparison_mask(value))?,
            )),
            Op::Simd(Simd::AllTrue(value)) if self.masks.contains(value) => Some(Inst::AllMask(
                self.ty(value).lane()?,
                self.dst(),
                self.input(value),
            )),
            Op::Simd(compact @ (Simd::Compress(value, mask) | Simd::Expand(value, mask)))
                if self.masks.contains(mask) =>
            {
                Some(Inst::MaskCompact(
                    self.ty(value).lane()?,
                    matches!(compact, Simd::Expand(..)),
                    self.dst(),
                    self.input(value),
                    self.input(mask),
                ))
            }
            _ => None,
        }
    }

    /// `canonical::expand` leaves `FirstLanes` only in a predicate register
    /// on a machine with BMI2.
    pub(super) fn first_lanes(&mut self, lane: Lane, count: Value) -> Result<Inst, CompileError> {
        let lanes = (self.cpu.bytes / lane.bytes()) as u8;
        if let Some(count) = self.dword_constant(count) {
            let count = (count as i32).clamp(0, lanes.into());
            return Ok(Inst::MaskImmediate(self.dst(), (1u64 << count) - 1));
        }
        Ok(Inst::MaskPrefix(
            lanes,
            self.dst(),
            self.input(count),
            self.body.temp(RegClass::Int)?,
        ))
    }

    fn opmask(&mut self, value: Value) -> Option<VectorMask<VReg>> {
        self.masks
            .contains(value)
            .then(|| VectorMask::Register(self.ty(value).lane().unwrap(), self.input(value)))
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
            let Op::Bin(Binary::And, a, b) = self.ctx.single_use(value)? else {
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
        if let Op::Bitcast(Type::Vector(to), from) = op {
            if to.bytes() != self.ty(from).lane()?.bytes() {
                return None;
            }
            op = self.ctx.single_use(from)?;
        }
        let Op::Bin(kind, lhs, rhs) = op else {
            return None;
        };
        let lane = self.ty(lhs).lane()?;
        let supported = match lane.bytes() {
            1 | 2 => self.cpu.isa.bw && !lane.float(),
            4 | 8 => true,
            _ => false,
        };
        supported.then_some((lane, comparison_predicate(lane, kind)?, lhs, rhs))
    }
}
