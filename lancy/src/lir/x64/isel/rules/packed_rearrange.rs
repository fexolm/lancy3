use super::Rules;
use crate::lir::x64::Cpu;
use crate::{
    hir::Value,
    hir::operation::Action,
    hir::{Binary as B, Lane, Op, Simd, Type},
    lir::x64::inst::{
        Inst,
        simd::packed::rearrange::{PairedShuffle, Rule},
    },
};
use regalloc2::RegClass;

impl Rules<'_> {
    pub(super) fn dynamic_dword_lane(
        &mut self,
        op: Op,
    ) -> Result<Option<Inst>, crate::CompileError> {
        let (value, index, replacement) = match op {
            Op::Simd(Simd::ExtractLane(x, i)) => (x, i, None),
            Op::Simd(Simd::InsertLane(x, i, y)) => (x, i, Some(y)),
            _ => return Ok(None),
        };
        let lane = self.ty(value).lane().unwrap();
        if lane.bytes() != 4 || self.cpu.bytes == 16 || self.cpu.bytes == 32 && !self.cpu.isa.avx2 {
            return Ok(None);
        }
        let count = self.cpu.bytes as u32 / 4;
        let constant = self.dword_constant(index);
        if constant.is_some_and(|n| n >= count) {
            return Ok(None);
        }
        let masked = matches!(self.ctx.defining_op(index), Some(Op::Bin(B::And, a, b))
            if [a, b].into_iter().any(|v| self.dword_constant(v).is_some_and(|n| n < count)));
        let checked = !masked && constant.is_none();
        let temp = if replacement.is_some() && self.cpu.bytes == 32 {
            Some(self.body.temp(RegClass::Vector)?)
        } else {
            None
        };
        Ok(Some(Inst::DwordLane {
            lane,
            checked,
            dst: self.dst(),
            value: self.input(value),
            index: self.input(index),
            replacement: replacement.map(|v| self.input(v)),
            temp,
        }))
    }

    pub(super) fn paired_shuffle(&mut self, op: Op) -> Option<Inst> {
        let Op::Simd(Simd::Shuffle(lhs, rhs, index)) = op else {
            return None;
        };
        if self.ty(lhs).lane()?.bytes() != 4 || self.cpu.bytes == 32 && !self.cpu.isa.avx2 {
            return None;
        }
        let n = self.cpu.bytes / 4;
        let index = self.constant_indices(index, 5)?;
        let form = [
            PairedShuffle::Interleave(false),
            PairedShuffle::Interleave(true),
            PairedShuffle::Deinterleave(false),
            PairedShuffle::Deinterleave(true),
        ]
        .into_iter()
        .find(|s| {
            index[..n]
                .iter()
                .enumerate()
                .all(|(i, &j)| j as usize == s.index(i, n))
        })?;
        Some(Inst::PairedShuffle(
            form,
            self.dst(),
            self.input(lhs),
            self.input(rhs),
        ))
    }

    fn constant_indices(&self, value: Value, depth: u8) -> Option<[u32; 16]> {
        match self.ctx.defining_op(value)? {
            Op::Simd(Simd::Iota(Lane::I32 | Lane::U32)) => Some(std::array::from_fn(|i| i as u32)),
            Op::Simd(Simd::Splat(value)) => Some([self.dword_constant(value)?; 16]),
            Op::Simd(Simd::Bin(op, a, b)) if depth != 0 => {
                let left = self.constant_indices(a, depth - 1)?;
                let right = self.constant_indices(b, depth - 1)?;
                let combine = match op {
                    B::Add => u32::wrapping_add,
                    B::Sub => u32::wrapping_sub,
                    B::Mul => u32::wrapping_mul,
                    B::And => |a, b| a & b,
                    B::Shr if self.ty(a) == Type::Vector(Lane::I32) => {
                        |a, b| (a as i32).wrapping_shr(b) as u32
                    }
                    B::Shr => u32::wrapping_shr,
                    _ => return None,
                };
                Some(std::array::from_fn(|i| combine(left[i], right[i])))
            }
            _ => None,
        }
    }

    pub(super) fn immediate_rearrangement(&mut self, op: Op) -> Option<Inst> {
        match op {
            Op::Simd(Simd::ExtractLane(vector, index)) => {
                let lane = self.ty(vector).lane()?;
                let index = self.dword_constant(index)?;
                ((index as usize) < self.cpu.bytes / lane.bytes())
                    .then(|| Inst::ExtractLane(lane, self.dst(), self.input(vector), index as u8))
            }
            Op::Simd(Simd::Shuffle(vector, _, indices))
                if self.ty(vector).lane()?.bytes() == 4
                    && (self.cpu.bytes != 32 || self.cpu.isa.avx2) =>
            {
                let Op::Simd(Simd::Bin(B::Sub, sequence, offset)) =
                    self.ctx.defining_op(indices)?
                else {
                    return None;
                };
                if !matches!(
                    self.ctx.defining_op(sequence),
                    Some(Op::Simd(Simd::Iota(_)))
                ) {
                    return None;
                }
                let Op::Simd(Simd::Splat(offset)) = self.ctx.defining_op(offset)? else {
                    return None;
                };
                let count = self.dword_constant(offset)?;
                // Wrapped negative indices lie beyond both tables. All other
                // indices are in the first table for a shift of 1..lane_count.
                (count != 0 && (count as usize) < self.cpu.bytes / 4)
                    .then(|| Inst::ShiftDwords(self.dst(), self.input(vector), count as u8))
            }
            _ => None,
        }
    }
}

impl Rule {
    pub(in crate::lir::x64) fn select(
        action: Action,
        input: Type,
        output: Type,
        c: Cpu,
    ) -> Option<Self> {
        let e = input.lane()?;
        let to = output.lane()?;
        let n = e.bytes();
        Some(match action {
            Action::Widen(_) | Action::Narrow
                if matches!((e, to), (Lane::F32, Lane::F64) | (Lane::F64, Lane::F32))
                    || matches!((e, to), (Lane::F32, Lane::F16) | (Lane::F16, Lane::F32))
                        && (c.bytes == 64 || c.isa.f16c)
                    || matches!((e, to), (Lane::F32, Lane::BF16) | (Lane::BF16, Lane::F32))
                        && (c.bytes == 16 || c.isa.avx2) =>
            {
                Self::FloatWidth
            }
            Action::Shuffle
                if crate::lir::x64::inst::simd::evex(c)
                    && (matches!(n, 4 | 8)
                        || n == 2 && c.isa.bw
                        || n == 1 && c.isa.bw && c.isa.vbmi)
                    || c.bytes == 32 && c.isa.avx2 && n == 4 =>
            {
                Self::Shuffle
            }
            Action::Widen(high)
                if !e.float()
                    && !to.float()
                    && to.bytes() == 2 * n
                    && to.bytes() <= 8
                    && (c.bytes == 16
                        || c.bytes == 32 && c.isa.avx2
                        || c.bytes == 64 && (n >= 2 || c.isa.bw)) =>
            {
                Self::Widen(high)
            }
            Action::Narrow
                if !e.float()
                    && !to.float()
                    && n == 2 * to.bytes()
                    && n <= 8
                    && (c.bytes == 16
                        || c.bytes == 32 && c.isa.avx2
                        || c.bytes == 64 && (n >= 4 || c.isa.bw)) =>
            {
                Self::Narrow
            }
            Action::Reduce(op)
                if !e.float()
                    && n <= 8
                    && (c.bytes == 16
                        || c.isa.avx2
                        || cfg!(feature = "experimental-simd") && c.bytes == 32)
                    && match op {
                        B::Add | B::And | B::Or | B::Xor => true,
                        B::Mul => n == 2 || n == 4 && c.isa.sse41,
                        B::Min | B::Max => {
                            n <= 4
                                && (c.isa.sse41
                                    || matches!((n, e.signed()), (1, false) | (2, true)))
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
