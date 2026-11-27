use super::{A, C, Cond, Emitter, O, PredQual, R, Result, operations::sve_lane_bytes};
use crate::hir::{Type, Unary};

#[derive(Clone, Copy, Debug)]
pub(super) enum Form {
    Integer(Unary, usize),
    Neon(C, A),
    Sve(C, A),
    NeonPopcnt(usize),
    NeonCtz(A),
    NeonCountZeros64(Unary),
    SveCtz(A),
    HalfSign(Unary, bool, bool),
    Float(FloatOp, FloatRegs),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum FloatOp {
    Native(C),
    Fract(C),
    Recip,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum FloatRegs {
    Scalar(A),
    Neon(A),
    Sve(A),
}

fn v(n: u8, arr: A) -> O {
    O::vec(R::v(n).unwrap(), arr)
}

fn z(n: u8, arr: A) -> O {
    O::vec(R::z(n).unwrap(), arr)
}

impl Emitter<'_> {
    pub(super) fn unary(&mut self, form: Form, src: u8, dst: u8) -> Result {
        match form {
            Form::Integer(op, size) => self.int_unary(op, size, src, dst),
            Form::Neon(code, arr) => self.a.put(code, &[v(dst, arr), v(src, arr)]),
            Form::Sve(code, arr) => {
                self.predicate(R::P1, arr, self.config.bytes / sve_lane_bytes(arr))?;
                self.a.put(
                    code,
                    &[z(dst, arr), O::pred(R::P1, PredQual::Merge), z(src, arr)],
                )
            }
            Form::SveCtz(arr) => {
                self.predicate(R::P1, arr, self.config.bytes / sve_lane_bytes(arr))?;
                let p = O::pred(R::P1, PredQual::Merge);
                let d = z(dst, arr);
                self.a.put(C::Rbit_z_p_z, &[d, p, z(src, arr)])?;
                self.a.put(C::Clz_z_p_z, &[d, p, d])
            }
            Form::NeonPopcnt(size) => {
                let d = v(dst, A::B16);
                self.a.put(C::Cnt_asimd, &[d, v(src, A::B16)])?;
                for pair in [A::B16, A::H8, A::S4, A::D2]
                    .windows(2)
                    .take(size.trailing_zeros() as usize)
                {
                    self.a
                        .put(C::Uaddlp_asimd, &[v(dst, pair[1]), v(dst, pair[0])])?;
                }
                Ok(())
            }
            Form::NeonCtz(arr) => {
                let d = v(dst, A::B16);
                self.a.put(C::Rbit_asimd, &[d, v(src, A::B16)])?;
                if arr != A::B16 {
                    let reverse = if arr == A::H8 {
                        C::Rev16_asimd
                    } else {
                        C::Rev32_asimd
                    };
                    self.a.put(reverse, &[d, d])?;
                }
                self.a.put(C::Clz_asimd, &[v(dst, arr), v(dst, arr)])
            }
            Form::NeonCountZeros64(op) => {
                self.a.put(
                    C::Umov_lane,
                    &[O::reg(R::X9), O::vec(R::v(src).unwrap(), A::D2), O::imm(1)],
                )?;
                self.float_move(Type::F64, false, src, 10)?;
                self.int_unary(op, 8, 9, 9)?;
                self.int_unary(op, 8, 10, 10)?;
                self.float_move(Type::F64, true, dst, 10)?;
                self.a.put(
                    C::Ins_lane,
                    &[O::vec(R::v(dst).unwrap(), A::D2), O::reg(R::X9), O::imm(1)],
                )
            }
            Form::HalfSign(op, packed, scalable) => self.half_sign(op, packed, scalable, src, dst),
            Form::Float(op, regs) => self.float_unary(op, regs, src, dst),
        }
    }

    pub(super) fn int_unary(&mut self, op: Unary, size: usize, src: u8, dst: u8) -> Result {
        use Unary::*;
        let wide = size == 8;
        let choose = |word, double| if wide { double } else { word };
        let reg = |n| O::reg(if wide { R::x(n) } else { R::w(n) }.unwrap());
        let (mut s, d, zero) = (reg(src), reg(dst), reg(31));
        let bits = (size * 8) as u64;
        match op {
            Neg => self.a.put(
                choose(C::Sub_32_addsub_shift, C::Sub_64_addsub_shift),
                &[d, zero, s],
            )?,
            Not => self.a.put(
                choose(C::Orn_32_log_shift, C::Orn_64_log_shift),
                &[d, zero, s],
            )?,
            Abs => {
                if size < 4 {
                    self.a.put(
                        C::Sbfm_32m_bitfield,
                        &[reg(13), s, O::imm(0), O::imm(bits - 1)],
                    )?;
                    s = reg(13);
                }
                self.a.put(
                    choose(C::Subs_32_addsub_imm, C::Subs_64_addsub_imm),
                    &[zero, s, O::imm(0)],
                )?;
                self.a.put(
                    choose(C::Csneg_32_condsel, C::Csneg_64_condsel),
                    &[d, s, s, O::cond(Cond::Pl)],
                )?;
            }
            LeadingZeros => {
                self.a.put(choose(C::Clz_32, C::Clz_64), &[d, s])?;
                if size < 4 {
                    self.a
                        .put(C::Sub_32_addsub_imm, &[d, d, O::imm(32 - bits)])?;
                }
                return Ok(());
            }
            TrailingZeros => {
                if size < 4 {
                    self.a.put(C::Orr_32_log_imm, &[d, s, O::imm(1 << bits)])?;
                    s = d;
                }
                self.a.put(choose(C::Rbit_32, C::Rbit_64), &[d, s])?;
                return self.a.put(choose(C::Clz_32, C::Clz_64), &[d, d]);
            }
            Popcnt => {
                self.float_move(if wide { Type::F64 } else { Type::F32 }, true, 16, src)?;
                let b = v(16, A::B16);
                self.a.put(C::Cnt_asimd, &[b, b])?;
                if size != 1 {
                    self.a.put(C::Addv_asimd, &[O::reg(R::V16), b])?;
                }
                return self.float_move(Type::F32, false, 16, dst);
            }
            _ => unreachable!("verified integer unary"),
        }
        if size < 4 {
            self.a
                .put(C::Ubfm_32m_bitfield, &[d, d, O::imm(0), O::imm(bits - 1)])?;
        }
        Ok(())
    }

    fn half_sign(&mut self, op: Unary, packed: bool, scalable: bool, src: u8, dst: u8) -> Result {
        let negate = op == Unary::Neg;
        if !packed {
            let (src, dst) = (O::reg(R::w(src).unwrap()), O::reg(R::w(dst).unwrap()));
            return if negate {
                self.a.put(C::Eor_32_log_imm, &[dst, src, O::imm(0x8000)])
            } else {
                self.a
                    .put(C::Ubfm_32m_bitfield, &[dst, src, O::imm(0), O::imm(14)])
            };
        }
        self.a.imm(R::W9, 0x8000)?;
        if scalable {
            self.a.put(C::Dup_z_r, &[z(16, A::H), O::reg(R::W9)])?;
            let code = if negate { C::Eor_z_zz_ } else { C::Bic_z_zz_ };
            self.a.put(code, &[z(dst, A::D), z(src, A::D), z(16, A::D)])
        } else {
            self.a
                .put(C::Dup_asimdgen, &[v(16, A::H8), O::reg(R::W9)])?;
            let code = if negate {
                C::Eor_asimdsame_only
            } else {
                C::Bic_asimdsame
            };
            self.a
                .put(code, &[v(dst, A::B16), v(src, A::B16), v(16, A::B16)])
        }
    }

    fn float_unary(&mut self, op: FloatOp, regs: FloatRegs, src: u8, dst: u8) -> Result {
        let (arr, sve) = match regs {
            FloatRegs::Scalar(arr) | FloatRegs::Neon(arr) => (arr, false),
            FloatRegs::Sve(arr) => (arr, true),
        };
        let reg = |n| if sve { z(n, arr) } else { v(n, arr) };
        let (s, d, t) = (reg(src), reg(dst), reg(16));
        let p = O::pred(R::P1, PredQual::Merge);
        if sve {
            self.predicate(R::P1, arr, self.config.bytes / sve_lane_bytes(arr))?;
        }
        match op {
            FloatOp::Native(code) => {
                if sve {
                    self.a.put(code, &[d, p, s])
                } else {
                    self.a.put(code, &[d, s])
                }
            }
            FloatOp::Fract(code) => match regs {
                FloatRegs::Scalar(_) => {
                    self.a.put(code, &[t, s])?;
                    self.a.put(C::Fsub_scalar, &[d, s, t])
                }
                FloatRegs::Neon(_) => {
                    self.a.put(code, &[t, s])?;
                    self.a.put(C::Fsub_asimdsame, &[d, s, t])
                }
                FloatRegs::Sve(_) => {
                    self.a.put(code, &[t, p, s])?;
                    self.a.put(C::Fsub_z_zz_, &[d, s, t])
                }
            },
            FloatOp::Recip => {
                let one = O::imm(0x70); // FP immediate encoding for 1.0.
                match regs {
                    FloatRegs::Scalar(_) => {
                        self.a.put(C::Fmov_imm_scalar, &[t, one])?;
                        self.a.put(C::Fdiv_scalar, &[d, t, s])
                    }
                    FloatRegs::Neon(_) => {
                        self.a.put(C::Fmov_imm_asimd, &[t, one])?;
                        self.a.put(C::Fdiv_asimdsame, &[d, t, s])
                    }
                    FloatRegs::Sve(_) if src == dst => {
                        self.a.put(C::Fmov_imm_z, &[t, one])?;
                        self.a.put(C::Fdivr_z_p_zz_, &[d, p, d, t])
                    }
                    FloatRegs::Sve(_) => {
                        self.a.put(C::Fmov_imm_z, &[d, one])?;
                        self.a.put(C::Fdiv_z_p_zz_, &[d, p, d, s])
                    }
                }
            }
        }
    }
}
