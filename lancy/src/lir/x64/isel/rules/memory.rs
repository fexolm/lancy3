use super::Rules;
use crate::support::slotmap::DenseMap;
use crate::{
    CompileError,
    hir::{Function, Op, Scale, Simd, Type, Value, VectorAddress},
    lir::x64::inst::{
        Address, Inst, Int, VectorMask,
        simd::{MaskedMemory, evex_masked_memory},
    },
};
use regalloc2::{RegClass, VReg};

/// Sign extensions of `i32` indices to pointer width. An index that several
/// address computations use is extended once, right after its definition,
/// and every address reuses the wide register; another one is extended at
/// its use, which keeps the wide register short-lived.
pub(in crate::lir::x64) struct Indices {
    shared: DenseMap<Value, bool>,
    wide: DenseMap<Value, Option<VReg>>,
}

impl Indices {
    pub(in crate::lir::x64) fn new(f: &Function) -> Self {
        let mut uses = DenseMap::filled(f.value_count(), 0u8);
        for b in f.blocks() {
            for inst in f.instructions(b) {
                if let Op::PtrOffset(_, index, _) = *f.op(inst) {
                    uses[index] = uses[index].saturating_add(1);
                }
            }
        }
        let shared = DenseMap::from_fn(f.value_count(), |v| uses[v] > 1);
        Self {
            shared,
            wide: DenseMap::filled(f.value_count(), None),
        }
    }

    /// The extension of `v` that selected addresses use, if any: the
    /// caller places it right after the definition of `v`.
    pub(in crate::lir::x64) fn extension(&self, v: Value, narrow: VReg) -> Option<Inst> {
        Some(Inst::Extend(Int::I32, self.wide[v]?, narrow))
    }
}

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
        let access = MaskedMemory {
            ty,
            base,
            index: indices,
            temp,
            mask,
            value: (!zero).then(|| self.input(value)),
            dst,
        };
        Ok(self.body.details_mut().masked(access))
    }

    /// `canonical::expand` leaves a compressing store only where
    /// `native_compress_store` holds and its mask is a predicate register.
    pub(super) fn compress_store(
        &mut self,
        base: Value,
        mask: Value,
        value: Value,
        limit: Value,
    ) -> Result<Inst, CompileError> {
        Ok(Inst::CompactStore {
            base: self.input(base),
            data: self.input(value),
            mask: self.input(mask),
            limit: self.input(limit),
            temps: [
                self.body.temp(RegClass::Int)?,
                self.body.temp(RegClass::Int)?,
            ],
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
        let wide = if self.indices.shared[index] {
            // Extended once where the index is defined (see `Indices`).
            self.ctx.require(index);
            match self.indices.wide[index] {
                Some(wide) => wide,
                None => {
                    let wide = self.body.temp(RegClass::Int)?;
                    self.indices.wide[index] = Some(wide);
                    wide
                }
            }
        } else {
            let wide = self.body.temp(RegClass::Int)?;
            let index = self.input(index);
            self.out.push(Inst::Extend(Int::I32, wide, index));
            wide
        };
        Ok(Address {
            base,
            index: Some((wide, scale)),
            disp: 0,
        })
    }
}
