use super::Rules;
use crate::{
    CompileError,
    hir::{Binary, Value},
    lir::x64::inst::{Inst, Int, Quotient},
};
use regalloc2::VReg;

impl Rules<'_> {
    pub(super) fn i32_division(
        &mut self,
        kind: Binary,
        dst: VReg,
        a: Value,
        b: Value,
    ) -> Result<Inst, CompileError> {
        let remainder = kind == Binary::Rem;
        match self.dword_constant(b).map(|n| n as i32) {
            Some(1) => {
                return Ok(if remainder {
                    Inst::Imm(dst, 0)
                } else {
                    Inst::Copy(dst, self.input(a))
                });
            }
            Some(divisor) if !matches!(divisor, 0 | -1) => {
                return Ok(Inst::DivImm {
                    remainder,
                    dst,
                    value: self.input(a),
                    divisor,
                    quotient: Quotient::for_divisor(divisor),
                });
            }
            // Division by zero and signed overflow use the checked register form.
            _ => {}
        }
        self.division(remainder, Int::I32, dst, a, b)
    }
}

impl Quotient {
    // The caller has handled ±1 and zero.
    fn for_divisor(divisor: i32) -> Self {
        let magnitude = divisor.unsigned_abs();
        let log = magnitude.ilog2();
        if magnitude.is_power_of_two() {
            return Self::PowerOfTwo(log as i32);
        }
        let d = u64::from(magnitude);
        let mut shift = 31 + log;
        let scale = 1u64 << shift;
        let mut multiplier = scale.div_ceil(d);
        // For n >= 0, the error in n*m/scale must be strictly below 1/d.
        // For n < 0 the result is rounded down, then incremented to
        // truncate toward zero. The reciprocal rounds up strictly for all
        // non-powers of two. One extra bit always meets the error bound.
        if multiplier * d - scale > 1u64 << log {
            shift += 1;
            multiplier = (1u64 << shift).div_ceil(d);
        }
        Self::Reciprocal {
            multiplier: multiplier as u32,
            shift: shift as i32,
        }
    }
}
