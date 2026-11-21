use super::{Rules, reg};
use crate::{
    CompileError,
    hir::{Op, Scale, Simd, Type, Value, VectorAddress},
    lir::x64::inst::{Address, Inst, VectorMask, simd::evex_masked_memory},
};
use regalloc2::RegClass;

impl Rules<'_> {
    pub(super) fn bounds_check(&mut self, index: Value, length: Value) -> Inst {
        if self.ctx.constant(length) == Some(1)
            && let Some(Op::Simd(Simd::AnyTrue(mask))) = self.ctx.single_use(index)
        {
            return Inst::CheckMask(self.vector_mask(mask));
        }
        Inst::CheckBounds(self.input(index), self.input(length))
    }

    pub(super) fn pointer_offset(
        &mut self,
        base: Value,
        index: Value,
        scale: Scale,
    ) -> Result<Inst, CompileError> {
        Ok(Inst::Lea(self.dst(), self.offset(base, index, scale)?))
    }

    pub(super) fn load(&mut self, ty: Type, ptr: Value) -> Result<Inst, CompileError> {
        Ok(Inst::Load(ty, self.dst(), self.address(ptr)?))
    }

    pub(super) fn store(&mut self, ptr: Value, value: Value) -> Result<Inst, CompileError> {
        Ok(Inst::Store(
            self.ty(value),
            self.address(ptr)?,
            self.input(value),
        ))
    }

    pub(super) fn masked_memory(
        &mut self,
        address: VectorAddress,
        mask: Value,
        value: Value,
    ) -> Result<Inst, CompileError> {
        let ty = self.ty(value);
        let dst = self.result.map(|v| reg(self.f, v));
        let base = self.input(address.base);
        let indices = address.indices.map(|v| self.input(v));
        // AVX2 gather needs a scratch vector for its destructive mask.
        let temp = if self.cpu.bytes == 32
            && self.cpu.avx10 == 0
            && self.cpu.avx2
            && address.indices.is_some()
            && dst.is_some()
            && matches!(ty.lane().unwrap().bytes(), 4 | 8)
        {
            Some(self.body.temp(RegClass::Vector)?)
        } else {
            None
        };
        let mask = if evex_masked_memory(self.cpu, ty.lane().unwrap().bytes(), indices.is_some()) {
            self.vector_mask(mask)
        } else {
            VectorMask::Lanes(self.input(mask))
        };
        Ok(Inst::Masked {
            ty,
            base,
            indices,
            temp,
            mask,
            value: self.input(value),
            dst,
        })
    }

    pub(super) fn address(&mut self, v: Value) -> Result<Address, CompileError> {
        match self.ctx.single_use(v) {
            Some(Op::PtrOffset(base, index, scale)) => self.offset(base, index, scale),
            _ => Ok(Address {
                base: self.input(v),
                index: None,
                disp: 0,
            }),
        }
    }

    fn offset(&mut self, base: Value, index: Value, scale: Scale) -> Result<Address, CompileError> {
        let base = self.input(base);
        if let Some(disp) = self
            .ctx
            .constant(index)
            .and_then(|i| i.checked_mul(1 << scale as u8))
        {
            return Ok(Address {
                base,
                index: None,
                disp,
            });
        }
        let wide = self.body.temp(RegClass::Int)?;
        let index = self.input(index);
        self.out.push(Inst::Movsxd(wide, index));
        Ok(Address {
            base,
            index: Some((wide, scale)),
            disp: 0,
        })
    }
}
