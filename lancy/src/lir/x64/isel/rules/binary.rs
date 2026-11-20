use super::Rules;
use crate::{
    CompileError,
    hir::{Binary, Op, Type, Value},
    lir::x64::inst::{Alu, Cond, Inst, Rhs, class},
};
use regalloc2::{RegClass, VReg};

impl Rules<'_> {
    pub(super) fn binary(
        &mut self,
        kind: Binary,
        a: Value,
        b: Value,
    ) -> Result<Inst, CompileError> {
        let ty = self.ty(a);
        let dst = self.dst();
        if ty == Type::I32
            && let Some(inst) = self.i32_binary(kind, dst, a, b)?
        {
            return Ok(inst);
        }
        if matches!(
            kind,
            Binary::Add | Binary::Sub | Binary::Mul | Binary::And | Binary::Or | Binary::Xor
        ) && class(ty) == RegClass::Int
            && !ty.lane().unwrap().float()
        {
            return Ok(Inst::IntBinary(kind, ty, dst, self.input(a), self.input(b)));
        }
        self.numeric(Op::Bin(kind, a, b))
    }

    pub(super) fn rhs(&mut self, v: Value) -> Rhs {
        match self.ctx.constant(v) {
            Some(n) => Rhs::Imm(n),
            None => Rhs::Reg(self.input(v)),
        }
    }

    fn i32_binary(
        &mut self,
        kind: Binary,
        dst: VReg,
        mut a: Value,
        mut b: Value,
    ) -> Result<Option<Inst>, CompileError> {
        if let Some(cc) = Cond::from_binary(kind) {
            return Ok(Some(Inst::Compare(cc, dst, self.input(a), self.rhs(b))));
        }
        if matches!(
            kind,
            Binary::Add | Binary::Mul | Binary::And | Binary::Or | Binary::Xor
        ) && self.ctx.constant(a).is_some()
            && self.ctx.constant(b).is_none()
        {
            std::mem::swap(&mut a, &mut b);
        }
        Ok(Some(match kind {
            Binary::Div | Binary::Rem => self.i32_division(kind, dst, a, b)?,
            Binary::Shl | Binary::Shr => Inst::Shift {
                left: kind == Binary::Shl,
                dst,
                value: self.input(a),
                count: self.rhs(b),
            },
            Binary::Add => Inst::Alu(Alu::Add, dst, self.input(a), self.rhs(b)),
            Binary::Sub => Inst::Alu(Alu::Sub, dst, self.input(a), self.rhs(b)),
            Binary::Mul => match self.ctx.constant(b).map(|n| n as u32) {
                Some(n) if n.is_power_of_two() => Inst::Shift {
                    left: true,
                    dst,
                    value: self.input(a),
                    count: Rhs::Imm(n.trailing_zeros() as i32),
                },
                _ => Inst::Alu(Alu::Mul, dst, self.input(a), self.rhs(b)),
            },
            Binary::And => Inst::Alu(Alu::And, dst, self.input(a), self.rhs(b)),
            Binary::Or => Inst::Alu(Alu::Or, dst, self.input(a), self.rhs(b)),
            Binary::Xor => Inst::Alu(Alu::Xor, dst, self.input(a), self.rhs(b)),
            Binary::Min => Inst::Alu(Alu::Min, dst, self.input(a), Rhs::Reg(self.input(b))),
            Binary::Max => Inst::Alu(Alu::Max, dst, self.input(a), Rhs::Reg(self.input(b))),
            _ => return Ok(None),
        }))
    }
}

impl Cond {
    pub fn from_binary(op: Binary) -> Option<Self> {
        Some(match op {
            Binary::Eq => Self::Eq,
            Binary::Ne => Self::Ne,
            Binary::Lt => Self::Lt,
            Binary::Le => Self::Le,
            Binary::Gt => Self::Gt,
            Binary::Ge => Self::Ge,
            _ => return None,
        })
    }
}
