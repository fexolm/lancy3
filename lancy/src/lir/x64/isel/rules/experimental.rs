//! Opt-in x64 instruction selection rules for manually vectorized kernels.
use super::Rules;
use crate::{
    CompileError,
    hir::{Binary, Op, Simd, Value},
    lir::x64::inst::Inst,
};

impl Rules<'_> {
    pub(super) fn experimental_numeric(&mut self, op: Op) -> Result<Option<Inst>, CompileError> {
        if let Op::Simd(Simd::Shuffle(x, y, indices)) = op
            && self.ty(x).lane().unwrap().bytes() == 4
            && let Some(Op::Simd(Simd::Splat(index))) = self.ctx.defining_op(indices)
            && let Some(index) = self.ctx.constant(index)
            && let Ok(index) = u8::try_from(index)
        {
            let lanes = self.cpu.bytes as u8 / 4;
            if index < 2 * lanes {
                let src = self.input(if index < lanes { x } else { y });
                return Ok(Some(Inst::BroadcastLane(self.dst(), src, index % lanes)));
            }
        }
        // VEX and EVEX accept unaligned packed memory operands. Keep the load
        // at the same point in the effect order by covering adjacent loads only.
        if self.cpu.bytes >= 32 {
            match op {
                Op::Simd(Simd::Bin(
                    kind @ (Binary::Add | Binary::And | Binary::Or | Binary::Xor),
                    x,
                    y,
                )) => {
                    for (loaded, other) in [(x, y), (y, x)] {
                        if let Some(Op::Simd(Simd::Load(lane, ptr))) =
                            self.ctx.adjacent_single_use(loaded)
                            && (kind != Binary::Add
                                || !lane.float() && matches!(lane.bytes(), 4 | 8))
                        {
                            let addr = self.address(ptr)?;
                            self.ctx.cover_adjacent_load(loaded);
                            return Ok(Some(Inst::PackedMemory(
                                kind,
                                lane,
                                self.dst(),
                                self.input(other),
                                addr,
                            )));
                        }
                    }
                }
                Op::Simd(Simd::Splat(v)) => {
                    if let Some(Op::Load(ty, ptr)) = self.ctx.adjacent_single_use(v)
                        && let Some(lane) = ty.lane()
                        && matches!(lane.bytes(), 4 | 8)
                    {
                        let addr = self.address(ptr)?;
                        self.ctx.cover_adjacent_load(v);
                        return Ok(Some(Inst::BroadcastMemory(lane, self.dst(), addr)));
                    }
                }
                _ => {}
            }
        }
        Ok(None)
    }

    pub(super) fn experimental_all_true(&mut self, v: Value) -> Option<Inst> {
        let lane = self.ty(v).lane().unwrap();
        let supported = match self.cpu.bytes {
            16 => lane.bytes() <= 4 || lane.bytes() == 8 && self.cpu.isa.sse41,
            32 => lane.bytes() <= 8,
            64 => matches!(lane.bytes(), 4 | 8) || (lane.bytes() < 4 && self.cpu.isa.bw),
            _ => false,
        };
        supported.then(|| Inst::BranchAll(lane, self.input(v)))
    }
}
