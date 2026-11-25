use super::super::masks::constant_predicate;
use super::Rules;
use crate::{
    hir::{Binary, Lane, Op, Simd, Ternary, Type, Unary, Value},
    lir::x64::inst::{Inst, VectorMask, simd::comparison_predicate},
};
use regalloc2::VReg;

impl Rules<'_> {
    pub(super) fn vector_mask(&mut self, value: Value) -> VectorMask<VReg> {
        self.opmask(value)
            .or_else(|| self.comparison_mask(value))
            .unwrap_or_else(|| VectorMask::Lanes(self.input(value)))
    }

    pub(super) fn masked_numeric(&mut self, op: Op) -> Option<Inst> {
        if self.result.is_some_and(|v| self.masks.contains(v)) {
            match op {
                Op::Bin(kind @ (Binary::And | Binary::Or | Binary::Xor), x, y) => {
                    return Some(Inst::MaskLogic(
                        self.ty(x).lane()?,
                        kind,
                        self.dst(),
                        self.input(x),
                        self.input(y),
                    ));
                }
                Op::Bin(kind, x, y) => {
                    let lane = self.ty(x).lane()?;
                    if kind == Binary::Lt
                        && matches!(lane, Lane::I32 | Lane::U32)
                        && let Some(prefix) = self.prefix_mask(x, y)
                    {
                        return Some(prefix);
                    }
                    return Some(Inst::MaskCompare(
                        lane,
                        comparison_predicate(lane, kind)?,
                        self.dst(),
                        self.input(x),
                        self.input(y),
                    ));
                }
                Op::Unary(Unary::Not, x) => {
                    return Some(Inst::MaskNot(self.ty(x).lane()?, self.dst(), self.input(x)));
                }
                Op::Simd(Simd::Splat(x)) => {
                    let lane = self.ty(x).lane()?;
                    let bits = match self.ctx.defining_op(x)? {
                        Op::Const(n) => u128::from(n as u32),
                        Op::Literal(_, bits) => bits,
                        _ => return None,
                    };
                    let bits = if constant_predicate(lane, bits)? {
                        u64::MAX >> (64 - self.cpu.bytes / lane.bytes())
                    } else {
                        0
                    };
                    return Some(Inst::MaskImmediate(self.dst(), bits));
                }
                _ => {}
            }
        }
        match op {
            Op::Ternary(Ternary::Select, cond, yes, no) => Some(Inst::SelectMask {
                lane: self.ty(yes).lane()?,
                mask: self.opmask(cond).or_else(|| self.comparison_mask(cond))?,
                yes: self.input(yes),
                no: self.input(no),
                dst: self.dst(),
            }),
            Op::Simd(Simd::AnyTrue(value)) => Some(Inst::AnyMask(
                self.dst(),
                self.opmask(value).or_else(|| self.comparison_mask(value))?,
            )),
            Op::Simd(Simd::AllTrue(value)) if self.masks.contains(value) => Some(Inst::AllMask(
                self.ty(value).lane()?,
                self.dst(),
                self.input(value),
            )),
            Op::Simd(Simd::Reduce(Binary::Add, value))
                if self.ty(value) == Type::Vector(Lane::I32) && self.cpu.isa.popcnt =>
            {
                let Op::Unary(Unary::Neg, mask) = self.ctx.single_use(value)? else {
                    return None;
                };
                self.masks
                    .contains(mask)
                    .then(|| Inst::MaskCount(self.dst(), self.input(mask)))
            }
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

    fn opmask(&mut self, value: Value) -> Option<VectorMask<VReg>> {
        self.masks
            .contains(value)
            .then(|| VectorMask::Register(self.ty(value).lane().unwrap(), self.input(value)))
    }

    fn prefix_mask(&mut self, iota: Value, right: Value) -> Option<Inst> {
        if !matches!(
            self.ctx.defining_op(iota),
            Some(Op::Simd(Simd::Iota(Lane::I32 | Lane::U32)))
        ) {
            return None;
        }
        let Op::Simd(Simd::Splat(count)) = self.ctx.defining_op(right)? else {
            return None;
        };
        let lanes = self.cpu.bytes as u32 / 4;
        if let Some(count) = self.dword_constant(count) {
            let count = if self.ty(iota) == Type::Vector(Lane::I32) && (count as i32) < 0 {
                0
            } else {
                count.min(lanes)
            };
            return Some(Inst::MaskImmediate(self.dst(), (1u64 << count) - 1));
        }
        if !self.cpu.isa.bmi2 {
            return None;
        }
        let unsigned = match self.ctx.defining_op(count) {
            Some(Op::Bitcast(Type::I32, v)) if self.ty(v) == Type::U32 => v,
            _ => count,
        };
        let Op::Bin(Binary::Min, a, b) = self.ctx.defining_op(unsigned)? else {
            return None;
        };
        if self.ty(unsigned) != Type::U32 {
            return None;
        }
        [a, b].into_iter().any(|v| {
            self.dword_constant(v).is_some_and(|v| v <= lanes)
                || matches!(self.ctx.defining_op(v), Some(Op::Bitcast(Type::U32, n)) if
                    matches!(self.ctx.defining_op(n), Some(Op::Simd(Simd::LaneCount(lane))) if lane.bytes() >= 4))
        }).then(|| Inst::MaskPrefix(self.dst(), self.input(unsigned)))
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
