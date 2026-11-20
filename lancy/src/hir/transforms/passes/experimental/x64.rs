//! Opt-in instruction selection rules for manually vectorized kernels.
use super::Rules;
use crate::{
    CompileError,
    hir::{Binary, Op, Simd, Type, Value},
    lir::x64::inst::{Inst, VectorMask},
};
use regalloc2::VReg;

impl Rules<'_> {
    pub(super) fn experimental_numeric(&mut self, op: Op) -> Result<Option<Inst>, CompileError> {
        if let Op::Simd(Simd::Shuffle(x, y, indices)) = op
            && self.ty(x).lane().unwrap().bytes() == 4
            && (self.cpu.bytes != 32 || self.cpu.avx2)
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
        if self.cpu.bytes == 64 || self.cpu.bytes == 32 && self.cpu.avx2 {
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

    pub(super) fn experimental_memory_mask(
        &mut self,
        ty: Type,
        mask: Value,
    ) -> Option<VectorMask<VReg>> {
        if self.cpu.bytes == 64
            && matches!(ty.lane().unwrap().bytes(), 4 | 8)
            && let Some(Op::Simd(Simd::Bin(kind, lhs, rhs))) = self.ctx.single_use(mask)
            && let Some(lane) = self.ty(lhs).lane()
            && !lane.float()
        {
            let predicate = match kind {
                Binary::Eq => Some(0),
                Binary::Lt => Some(1),
                Binary::Le => Some(2),
                Binary::Ne => Some(4),
                Binary::Ge => Some(5),
                Binary::Gt => Some(6),
                _ => None,
            };
            predicate.map(|predicate| VectorMask::Compare {
                lhs: self.input(lhs),
                rhs: self.input(rhs),
                predicate,
                signed: lane.signed(),
            })
        } else {
            None
        }
    }

    pub(super) fn experimental_all_true(&mut self, v: Value) -> Option<Inst> {
        let lane = self.ty(v).lane().unwrap();
        let supported = match self.cpu.bytes {
            16 => lane.bytes() <= 4 || lane.bytes() == 8 && self.cpu.sse41,
            32 => self.cpu.avx2 && lane.bytes() <= 8,
            64 => matches!(lane.bytes(), 4 | 8) || (lane.bytes() < 4 && self.cpu.bw),
            _ => false,
        };
        supported.then(|| Inst::BranchAll(lane, self.input(v)))
    }
}
