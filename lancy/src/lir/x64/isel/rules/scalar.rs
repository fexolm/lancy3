//! Scalars in general registers: integers, and the sign and class bit
//! operations of 16-bit floats, which have no arithmetic there.
use super::Rules;
use crate::{
    CompileError,
    hir::{Binary as B, Lane, Op, Ternary, Type, Unary as U, Value},
    lir::x64::inst::{Alu, Inst, Int, IntUnary, Rhs, Shift},
};
use regalloc2::{RegClass, VReg};

const HALF_BITS: Int = Int {
    bytes: 2,
    signed: false,
};

impl Rules<'_> {
    pub(super) fn general_scalar(&mut self, op: Op) -> Result<Option<Inst>, CompileError> {
        Ok(Some(match op {
            Op::Unary(kind, x) => {
                let ty = self.ty(x);
                if let Some(int) = Int::of(ty)
                    && let Some(unary) = IntUnary::from_unary(kind)
                {
                    let temp = if unary.needs_temp(self.cpu.isa) {
                        Some(self.body.temp(RegClass::Int)?)
                    } else {
                        None
                    };
                    Inst::IntUnary(unary, int, self.dst(), self.input(x), temp)
                } else if let Some(lane) = half(ty) {
                    let (dst, x) = (self.dst(), self.input(x));
                    match kind {
                        U::Not => Inst::IntUnary(IntUnary::Not, HALF_BITS, dst, x, None),
                        U::Neg => Inst::Alu(Alu::Xor, HALF_BITS, dst, x, Rhs::Imm(0x8000)),
                        U::Abs => Inst::Alu(Alu::And, HALF_BITS, dst, x, Rhs::Imm(0x7fff)),
                        U::SignBit => {
                            let wide = self.body.temp(RegClass::Int)?;
                            let int = Int {
                                signed: true,
                                ..HALF_BITS
                            };
                            self.out.push(Inst::Extend(int, wide, x));
                            Inst::Shift {
                                op: Shift::Right,
                                int: Int::I32,
                                dst,
                                value: wide,
                                count: Rhs::Imm(31),
                            }
                        }
                        U::IsNan | U::IsInfinite | U::IsFinite | U::IsNormal | U::IsSubnormal => {
                            Inst::Classify(kind, lane, dst, x)
                        }
                        _ => return Ok(None),
                    }
                } else {
                    return Ok(None);
                }
            }
            Op::Bin(kind @ (B::And | B::Or | B::Xor), a, b) if half(self.ty(a)).is_some() => {
                let op = Alu::from_binary(kind).unwrap();
                Inst::Alu(
                    op,
                    HALF_BITS,
                    self.dst(),
                    self.input(a),
                    Rhs::Reg(self.input(b)),
                )
            }
            Op::Bin(B::CopySign, a, b) if half(self.ty(a)).is_some() => {
                let magnitude = self.masked(a, 0x7fff)?;
                let sign = self.masked(b, 0x8000)?;
                Inst::Alu(Alu::Or, HALF_BITS, self.dst(), magnitude, Rhs::Reg(sign))
            }
            Op::Cast(to, x) if is_float_cast(self.ty(x), to) => {
                let (from, to) = (self.ty(x).lane().unwrap(), to.lane().unwrap());
                if from == to {
                    return Ok(Some(Inst::Copy(self.dst(), self.input(x))));
                }
                let temp = match crate::lir::x64::inst::convert::temp(from, to) {
                    Some(class) => Some(self.body.temp(class)?),
                    None => None,
                };
                Inst::Convert {
                    from,
                    to,
                    dst: self.dst(),
                    src: self.input(x),
                    temp,
                }
            }
            // c ^ (mask & (b ^ c))
            Op::Ternary(Ternary::BitSelect, mask, b, c) => {
                let ty = self.ty(b);
                let Some(int) = Int::of(ty).or(half(ty).map(|_| HALF_BITS)) else {
                    return Ok(None);
                };
                let (mask, b, c) = (self.input(mask), self.input(b), self.input(c));
                let differ = self.body.temp(RegClass::Int)?;
                self.out
                    .push(Inst::Alu(Alu::Xor, int, differ, b, Rhs::Reg(c)));
                let chosen = self.body.temp(RegClass::Int)?;
                self.out
                    .push(Inst::Alu(Alu::And, int, chosen, differ, Rhs::Reg(mask)));
                Inst::Alu(Alu::Xor, int, self.dst(), chosen, Rhs::Reg(c))
            }
            _ => return Ok(None),
        }))
    }

    fn masked(&mut self, v: Value, bits: i32) -> Result<VReg, CompileError> {
        let out = self.body.temp(RegClass::Int)?;
        let v = self.input(v);
        self.out
            .push(Inst::Alu(Alu::And, HALF_BITS, out, v, Rhs::Imm(bits)));
        Ok(out)
    }
}

/// A scalar cast of at most eight bytes from or to a float.
fn is_float_cast(from: Type, to: Type) -> bool {
    let small =
        |ty: Type| !matches!(ty, Type::Vector(_)) && ty.lane().is_some_and(|l| l.bytes() <= 8);
    small(from) && small(to) && (from.lane().unwrap().float() || to.lane().unwrap().float())
}

fn half(ty: Type) -> Option<Lane> {
    match ty {
        Type::F16 => Some(Lane::F16),
        Type::BF16 => Some(Lane::BF16),
        _ => None,
    }
}
