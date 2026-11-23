use super::Rules;
use crate::{
    CompileError,
    hir::{Binary, Lane, Op, Scale, Simd, Type, Unary, Value, VectorAddress},
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
        let dst = self.result.map(|v| self.masks.reg(self.f, v));
        if dst.is_none()
            && address.indices.is_none()
            && let Some(inst) = self.compact_store(address.base, mask, value)?
        {
            return Ok(inst);
        }
        let base = self.input(address.base);
        let indices = address.indices.map(|v| self.input(v));
        let size = ty.lane().unwrap().bytes();
        let evex = evex_masked_memory(self.cpu, size, indices.is_some());
        // AVX2 gather needs a scratch vector for its destructive mask.
        let temp = if self.cpu.bytes == 32
            && self.cpu.isa.avx10 == 0
            && address.indices.is_some()
            && dst.is_some()
            && matches!(ty.lane().unwrap().bytes(), 4 | 8)
        {
            Some(self.body.temp(RegClass::Vector)?)
        } else {
            None
        };
        let mask = if evex {
            self.vector_mask(mask)
        } else {
            VectorMask::Lanes(self.input(mask))
        };
        let vex = self.cpu.bytes == 32 && matches!(size, 4 | 8);
        let zero = dst.is_some()
            && (evex || vex)
            && matches!(self.ctx.defining_op(value), Some(Op::Simd(Simd::Splat(x))) if
                matches!(self.ctx.defining_op(x), Some(Op::Const(0) | Op::Literal(_, 0))));
        Ok(Inst::Masked {
            ty,
            base,
            indices,
            temp,
            mask,
            value: (!zero).then(|| self.input(value)),
            dst,
        })
    }

    fn compact_store(
        &mut self,
        base: Value,
        enabled: Value,
        result: Value,
    ) -> Result<Option<Inst>, CompileError> {
        if self.ty(result) != Type::Vector(Lane::I32)
            || !self.masks.contains(enabled)
            || !self.cpu.isa.bmi2
        {
            return Ok(None);
        }
        let Some(Op::Simd(Simd::Compress(data, original))) = self.ctx.single_use(result) else {
            return Ok(None);
        };
        if !self.masks.contains(original) {
            return Ok(None);
        }
        let Some(Op::Simd(Simd::Bin(Binary::And, a, b))) = self.ctx.defining_op(enabled) else {
            return Ok(None);
        };
        let (Some(a), Some(b)) = (self.prefix_limit(a), self.prefix_limit(b)) else {
            return Ok(None);
        };
        let exact_count = |limit| {
            matches!(self.ctx.defining_op(limit), Some(Op::Simd(Simd::Reduce(Binary::Add, flags)))
                if self.ctx.defining_op(flags) == Some(Op::Simd(Simd::Unary(Unary::Neg, original))))
        };
        // Both predicates are prefixes, and the count limits the prefix to
        // real elements. Deposit its bits into the source lane positions.
        if !exact_count(a) && !exact_count(b) {
            return Ok(None);
        }
        Ok(Some(Inst::CompactStore {
            base: self.input(base),
            data: self.input(data),
            original: self.input(original),
            enabled: self.input(enabled),
            temp: self.body.temp(RegClass::Int)?,
        }))
    }

    fn prefix_limit(&self, value: Value) -> Option<Value> {
        let Op::Simd(Simd::Bin(Binary::Lt, index, limit)) = self.ctx.defining_op(value)? else {
            return None;
        };
        if self.ctx.defining_op(index) != Some(Op::Simd(Simd::Iota(Lane::I32))) {
            return None;
        }
        match self.ctx.defining_op(limit)? {
            Op::Simd(Simd::Splat(value)) => Some(value),
            _ => None,
        }
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
