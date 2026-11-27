use super::Rules;
use crate::{
    CompileError,
    hir::{Binary, Op, Value},
    lir::x64::inst::{Alu, Cond, Inst, Int, Rhs, Shift},
};
use regalloc2::{RegClass, VReg};

impl Rules<'_> {
    pub(super) fn binary(
        &mut self,
        kind: Binary,
        a: Value,
        b: Value,
    ) -> Result<Inst, CompileError> {
        match Int::of(self.ty(a)) {
            Some(int) => self.integer_binary(kind, int, a, b),
            None => self.numeric(Op::Bin(kind, a, b)),
        }
    }

    /// An integer operand: an immediate for an `i32` constant.
    pub(super) fn rhs(&mut self, v: Value) -> Rhs {
        match self.ctx.constant(v) {
            Some(n) => Rhs::Imm(n),
            None => Rhs::Reg(self.input(v)),
        }
    }

    fn integer_binary(
        &mut self,
        kind: Binary,
        int: Int,
        mut a: Value,
        mut b: Value,
    ) -> Result<Inst, CompileError> {
        let dst = self.dst();
        if let Some(cc) = Cond::from_binary(kind, int.signed) {
            return Ok(Inst::Compare(
                cc,
                int.bytes,
                dst,
                self.input(a),
                self.rhs(b),
            ));
        }
        if matches!(
            kind,
            Binary::Add | Binary::Mul | Binary::And | Binary::Or | Binary::Xor
        ) && self.ctx.constant(a).is_some()
            && self.ctx.constant(b).is_none()
        {
            std::mem::swap(&mut a, &mut b);
        }
        Ok(match kind {
            Binary::Div | Binary::Rem if int == Int::I32 => self.i32_division(kind, dst, a, b)?,
            Binary::Div | Binary::Rem => self.division(kind == Binary::Rem, int, dst, a, b)?,
            Binary::Mul => match self.ctx.constant(b).map(|n| n as u32) {
                Some(n) if n.is_power_of_two() => Inst::Shift {
                    op: Shift::Left,
                    int,
                    dst,
                    value: self.input(a),
                    count: Rhs::Imm(n.trailing_zeros() as i32),
                },
                _ => Inst::Alu(Alu::Mul, int, dst, self.input(a), self.rhs(b)),
            },
            Binary::SaturatingAdd | Binary::SaturatingSub => Inst::Saturating {
                sub: kind == Binary::SaturatingSub,
                int,
                dst,
                lhs: self.input(a),
                rhs: self.input(b),
            },
            _ => {
                if let Some(op) = Shift::from_binary(kind) {
                    let count = match self.ctx.constant(b) {
                        Some(n) => Rhs::Imm(n & (int.bits() - 1)),
                        None => Rhs::Reg(self.input(b)),
                    };
                    return Ok(Inst::Shift {
                        op,
                        int,
                        dst,
                        value: self.input(a),
                        count,
                    });
                }
                let op = Alu::from_binary(kind).expect("verified integer operation");
                let rhs = if op.register_only() {
                    Rhs::Reg(self.input(b))
                } else {
                    self.rhs(b)
                };
                Inst::Alu(op, int, dst, self.input(a), rhs)
            }
        })
    }

    /// Divide in hardware registers. Narrow operands are first extended,
    /// so the 32-bit division is exact; the type still decides overflow.
    pub(super) fn division(
        &mut self,
        remainder: bool,
        int: Int,
        dst: VReg,
        a: Value,
        b: Value,
    ) -> Result<Inst, CompileError> {
        let operand = |rules: &mut Self, v: Value| -> Result<VReg, CompileError> {
            let reg = rules.input(v);
            if int.bytes >= 4 {
                return Ok(reg);
            }
            let wide = rules.body.temp(RegClass::Int)?;
            rules.out.push(Inst::Extend(int, wide, reg));
            Ok(wide)
        };
        let lo = operand(self, a)?;
        let divisor = operand(self, b)?;
        let hi = self.body.temp(RegClass::Int)?;
        self.out.push(Inst::DivideHigh { int, lo, hi });
        Ok(Inst::Divide {
            int,
            remainder,
            dst,
            lo,
            hi,
            divisor,
        })
    }
}
