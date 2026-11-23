use super::{A, C, Cond, Emitter, O, R, Result};
use crate::hir::{Lane, Type, Unary};

#[derive(Clone, Copy, Debug)]
pub(super) enum Action {
    Binary(C),
    Compare(Cond),
    Unary(Unary),
    Cast(Lane),
    MulAdd,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Form {
    pub lane: Lane,
    pub packed: bool,
    pub scalable: bool,
    pub action: Action,
}

fn f(n: u8, arr: A) -> O {
    O::vec(R::v(n).unwrap(), arr)
}

fn w(n: u8) -> O {
    O::reg(R::w(n).unwrap())
}

fn infinity(lane: Lane) -> u64 {
    if lane == Lane::F16 { 0x7c00 } else { 0x7f80 }
}

fn minimum_normal(lane: Lane) -> u64 {
    if lane == Lane::F16 { 0x0400 } else { 0x0080 }
}

impl Emitter<'_> {
    pub(super) fn half(&mut self, form: Form, a: u8, extra: [Option<u8>; 2], dst: u8) -> Result {
        if !form.packed {
            return self.half_lane(form, a, extra, dst);
        }
        let registers = [extra[0].map(|_| 10), extra[1].map(|_| 11)];
        if form.scalable {
            return self.sve_each_lane(A::H, |e| {
                e.sve_lane_to_int(A::H, a, 9)?;
                for (i, v) in extra.into_iter().enumerate() {
                    if let Some(v) = v {
                        e.sve_lane_to_int(A::H, v, 10 + i as u8)?;
                    }
                }
                e.half_lane(form, 9, registers, 12)?;
                e.sve_lane_from_int(A::H, 12, dst, R::P1)
            });
        }
        for lane in 0..8 {
            self.neon_lane_to_int(a, 9, 2, lane)?;
            for (i, v) in extra.into_iter().enumerate() {
                if let Some(v) = v {
                    self.neon_lane_to_int(v, 10 + i as u8, 2, lane)?;
                }
            }
            self.half_lane(form, 9, registers, 12)?;
            self.neon_lane_from_int(12, dst, 2, lane)?;
        }
        Ok(())
    }

    pub(super) fn half_lane(
        &mut self,
        form: Form,
        a: u8,
        [b, c]: [Option<u8>; 2],
        dst: u8,
    ) -> Result {
        use Unary::*;
        let lane = form.lane;
        let d = f(16, A::D);
        let rhs = f(17, A::D);
        self.decode_half(lane, a, 16)?;
        if let Some(b) = b {
            self.decode_half(lane, b, 17)?;
        }
        match form.action {
            // FMINNM/FMAXNM alone would turn a signaling NaN operand into NaN.
            Action::Binary(code @ (C::Fminnm_scalar | C::Fmaxnm_scalar)) => {
                self.ordered_minmax(code == C::Fminnm_scalar, d, rhs)?;
            }
            Action::Binary(code) => self.a.put(code, &[d, d, rhs])?,
            Action::Compare(cond) => {
                self.a.put(C::Fcmp_scalar, &[d, rhs])?;
                return self.bool_mask(cond, dst);
            }
            Action::Unary(op) => {
                let code = match op {
                    Sqrt => C::Fsqrt_scalar,
                    Floor => C::Frintm_scalar,
                    Ceil => C::Frintp_scalar,
                    Trunc => C::Frintz_scalar,
                    Round => C::Frinta_scalar,
                    Nearest => C::Frintn_scalar,
                    Fract => {
                        self.a.put(C::Frintz_scalar, &[rhs, d])?;
                        self.a.put(C::Fsub_scalar, &[d, d, rhs])?;
                        return self.encode_half(lane, 16, dst);
                    }
                    Recip => {
                        self.a.put(C::Fmov_imm_scalar, &[rhs, O::imm(0x70)])?;
                        self.a.put(C::Fdiv_scalar, &[d, rhs, d])?;
                        return self.encode_half(lane, 16, dst);
                    }
                    _ => unreachable!("selected half unary operation"),
                };
                self.a.put(code, &[d, d])?;
            }
            Action::Cast(to) => return self.encode_half(to, 16, dst),
            Action::MulAdd => {
                self.decode_half(lane, c.unwrap(), 18)?;
                let addend = f(18, A::D);
                self.a.put(C::Fmul_scalar, &[d, d, rhs])?;
                if lane == Lane::F16 {
                    self.a.put(C::Fadd_scalar, &[d, d, addend])?;
                } else {
                    self.a.put(C::Fadd_scalar, &[f(19, A::D), d, addend])?;
                    return self.encode_bfloat(19, dst, true);
                }
            }
        }
        self.encode_half(lane, 16, dst)
    }

    pub(super) fn half_cast(&mut self, from: Lane, to: Lane, src: u8, dst: u8) -> Result {
        let source = match from {
            Lane::F64 => src,
            Lane::F32 => {
                self.a
                    .put(C::Fcvt_d_s, &[O::reg(R::V16), O::reg(R::v(src).unwrap())])?;
                16
            }
            Lane::F16 | Lane::BF16 => {
                let target = if to == Lane::F64 { dst } else { 16 };
                self.decode_half(from, src, target)?;
                target
            }
            _ => unreachable!("selected floating-point cast"),
        };
        match to {
            Lane::F64 => Ok(()),
            Lane::F32 => self.a.put(
                C::Fcvt_s_d,
                &[O::reg(R::v(dst).unwrap()), O::reg(R::v(source).unwrap())],
            ),
            _ => self.encode_half(to, source, dst),
        }
    }

    pub(super) fn decode_half(&mut self, lane: Lane, src: u8, dst: u8) -> Result {
        self.canonical_half(lane, src, 13)?;
        if lane == Lane::F16 {
            self.float_move(Type::F32, true, dst, 13)?;
            self.a.put(C::Fcvt_d_h, &[f(dst, A::D), f(dst, A::H)])
        } else {
            self.a.put(
                C::Ubfm_32m_bitfield,
                &[w(13), w(13), O::imm(16), O::imm(15)],
            )?;
            self.float_move(Type::F32, true, dst, 13)?;
            let v = O::reg(R::v(dst).unwrap());
            self.a.put(C::Fcvt_d_s, &[v, v])
        }
    }

    fn encode_half(&mut self, lane: Lane, src: u8, dst: u8) -> Result {
        if lane == Lane::F16 {
            self.a.put(C::Fcvt_h_d, &[f(16, A::H), f(src, A::D)])?;
            self.float_move(Type::F32, false, 16, dst)?;
            self.a
                .put(C::And_32_log_imm, &[w(dst), w(dst), O::imm(0xffff)])?;
            return self.canonical_half(lane, dst, dst);
        }
        self.encode_bfloat(src, dst, false)
    }

    fn encode_bfloat(&mut self, src: u8, dst: u8, from_fma: bool) -> Result {
        let d = f(src, A::D);
        let ordered = self.a.label();
        let done = self.a.label();
        self.a.put(C::Fcmp_scalar, &[d, d])?;
        self.a.skip_if(Cond::Vs)?;
        self.a.jump(ordered);
        self.float_move(Type::F64, false, src, 13)?;
        self.a.put(
            C::Ubfm_64m_bitfield,
            &[O::reg(R::X13), O::reg(R::X13), O::imm(63), O::imm(63)],
        )?;
        self.a
            .put(C::Ubfm_32m_bitfield, &[w(13), w(13), O::imm(17), O::imm(0)])?;
        self.a
            .put(C::Orr_32_log_imm, &[w(dst), w(13), O::imm(0x7fc0)])?;
        self.a.jump(done);
        self.a.bind(ordered);
        // Round to odd before dropping the low halfword, so the final nearest-even
        // rounding can still distinguish an exact midpoint from its neighbors.
        let tmp = if from_fma { 17 } else { 16 };
        self.a.put(C::Fcvtxn_s_d, &[f(tmp, A::S), d])?;
        self.float_move(Type::F32, false, tmp, 13)?;
        if from_fma {
            self.adjust_bfloat_midpoint()?;
        }
        self.a.put(
            C::Ubfm_32m_bitfield,
            &[w(14), w(13), O::imm(16), O::imm(16)],
        )?;
        self.a.put(C::Add_32_addsub_shift, &[w(13), w(13), w(14)])?;
        self.a.imm(R::W14, 0x7fff)?;
        self.a.put(C::Add_32_addsub_shift, &[w(13), w(13), w(14)])?;
        self.a.put(
            C::Ubfm_32m_bitfield,
            &[w(dst), w(13), O::imm(16), O::imm(31)],
        )?;
        self.a.bind(done);
        Ok(())
    }

    fn adjust_bfloat_midpoint(&mut self) -> Result {
        let done = self.a.label();
        self.a
            .put(C::And_32_log_imm, &[w(14), w(13), O::imm(0xffff)])?;
        self.a.imm(R::W15, 0x8000)?;
        self.a
            .put(C::Subs_32_addsub_shift, &[O::reg(R::WZR), w(14), w(15)])?;
        self.a.skip_if(Cond::Eq)?;
        self.a.jump(done);
        // The low bit of an inexact FCVTXN result is odd, so a midpoint
        // reaching this branch was represented exactly in binary32. TwoSum
        // recovers the tail lost in the preceding binary64 addition.
        let (p, b, c, s) = (f(16, A::D), f(17, A::D), f(18, A::D), f(19, A::D));
        self.a.put(C::Fsub_scalar, &[b, s, p])?;
        self.a.put(C::Fsub_scalar, &[c, c, b])?;
        self.a.put(C::Fsub_scalar, &[b, s, b])?;
        self.a.put(C::Fsub_scalar, &[p, p, b])?;
        self.a.put(C::Fadd_scalar, &[p, p, c])?;
        self.float_move(Type::F64, false, 16, 15)?;
        let (error, magnitude) = (O::reg(R::X15), O::reg(R::X14));
        self.a.put(
            C::Ubfm_64m_bitfield,
            &[magnitude, error, O::imm(0), O::imm(62)],
        )?;
        self.a.put(
            C::Subs_64_addsub_imm,
            &[O::reg(R::XZR), magnitude, O::imm(0)],
        )?;
        self.a.skip_if(Cond::Ne)?;
        self.a.jump(done);
        self.a.put(
            C::Ubfm_64m_bitfield,
            &[error, error, O::imm(32), O::imm(63)],
        )?;
        self.a.put(C::Eor_32_log_shift, &[w(15), w(15), w(13)])?;
        self.a
            .put(C::Subs_32_addsub_imm, &[O::reg(R::WZR), w(15), O::imm(0)])?;
        self.a
            .put(C::Add_32_addsub_imm, &[w(14), w(13), O::imm(1)])?;
        self.a
            .put(C::Sub_32_addsub_imm, &[w(15), w(13), O::imm(1)])?;
        self.a.put(
            C::Csel_32_condsel,
            &[w(13), w(15), w(14), O::cond(Cond::Mi)],
        )?;
        self.a.bind(done);
        Ok(())
    }

    fn canonical_half(&mut self, lane: Lane, src: u8, dst: u8) -> Result {
        self.a
            .put(C::And_32_log_imm, &[w(14), w(src), O::imm(0x7fff)])?;
        self.a.imm(R::W15, infinity(lane))?;
        self.a
            .put(C::Subs_32_addsub_shift, &[O::reg(R::WZR), w(14), w(15)])?;
        self.a
            .put(C::And_32_log_imm, &[w(14), w(src), O::imm(0x8000)])?;
        let nan = infinity(lane) | minimum_normal(lane) >> 1;
        self.a
            .put(C::Orr_32_log_imm, &[w(14), w(14), O::imm(nan)])?;
        self.a.put(
            C::Csel_32_condsel,
            &[w(dst), w(14), w(src), O::cond(Cond::Hi)],
        )
    }
}
