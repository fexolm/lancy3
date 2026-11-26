use super::{A, C, Cond, Emitter, O, R, Result, arrangements, groups::z};
use crate::hir::{Lane, Unary};

#[derive(Clone, Copy, Debug)]
pub(super) enum Action {
    MinMax(bool),
    CopySign,
    Classify(Unary),
    Signum,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Form {
    pub lane: Lane,
    pub packed: bool,
    pub scalable: bool,
    pub action: Action,
}

/// Fraction bits, exponent bias and sign bit of a float lane.
pub(super) fn float_layout(lane: Lane) -> (u64, u64, u64) {
    let (fraction, bias) = match lane {
        Lane::F16 => (10, 15),
        Lane::BF16 => (7, 127),
        Lane::F32 => (23, 127),
        Lane::F64 => (52, 1023),
        _ => unreachable!("selected floating type"),
    };
    (fraction, bias, 1 << (lane.bytes() * 8 - 1))
}

fn x(n: u8) -> O {
    O::reg(R::x(n).unwrap())
}

impl Emitter<'_> {
    pub(super) fn float_special(
        &mut self,
        form: Form,
        src: u8,
        rhs: Option<u8>,
        dst: u8,
    ) -> Result {
        let size = form.lane.bytes();
        let (neon, scalable) = arrangements(size);
        if !form.packed {
            let ty = form.lane.scalar();
            for (from, to) in [(Some(src), 9), (rhs, 10)] {
                if let Some(from) = from {
                    if size == 2 {
                        self.a
                            .put(C::And_64_log_imm, &[x(to), x(from), O::imm(0xffff)])?;
                    } else {
                        self.float_move(ty, false, from, to)?;
                    }
                }
            }
            self.float_special_lane(form)?;
            if size == 2 || matches!(form.action, Action::Classify(_)) {
                let w = |n| O::reg(R::w(n).unwrap());
                return self
                    .a
                    .put(C::Orr_32_log_shift, &[w(dst), O::reg(R::WZR), w(12)]);
            }
            return self.float_move(ty, true, dst, 12);
        }
        if matches!(form.action, Action::Classify(Unary::SignBit)) {
            if form.scalable {
                return self.a.put(
                    C::Asr_z_zi,
                    &[
                        z(dst, scalable),
                        z(src, scalable),
                        O::imm((size * 8 - 1) as u64),
                    ],
                );
            }
            let v = |n| O::vec(R::v(n).unwrap(), neon);
            return self.a.put(C::Sshr_sign_asimdshf, &[v(dst), v(src)]);
        }
        if form.scalable {
            return self.sve_each_lane(scalable, |e| {
                e.sve_lane_to_int(scalable, src, 9)?;
                if let Some(rhs) = rhs {
                    e.sve_lane_to_int(scalable, rhs, 10)?;
                }
                e.float_special_lane(form)?;
                e.sve_lane_from_int(scalable, 12, dst, R::P1)
            });
        }
        for lane in 0..16 / size {
            self.neon_lane_to_int(src, 9, size, lane)?;
            if let Some(rhs) = rhs {
                self.neon_lane_to_int(rhs, 10, size, lane)?;
            }
            self.float_special_lane(form)?;
            self.neon_lane_from_int(12, dst, size, lane)?;
        }
        Ok(())
    }

    fn float_special_lane(&mut self, form: Form) -> Result {
        let (fraction, bias, sign) = float_layout(form.lane);
        let min_normal = 1 << fraction;
        let infinity = sign - min_normal;
        let (a, b, out, mag, tmp) = (x(9), x(10), x(12), x(13), x(14));
        let zr = x(31);
        if let Action::MinMax(min) = form.action {
            let ty = form.lane.scalar();
            let arr = if form.lane == Lane::F32 { A::S } else { A::D };
            self.float_move(ty, true, 16, 9)?;
            self.float_move(ty, true, 17, 10)?;
            self.ordered_minmax(min, O::vec(R::V16, arr), O::vec(R::V17, arr))?;
            return self.float_move(ty, false, 16, 12);
        }
        if matches!(form.action, Action::Classify(Unary::SignBit)) {
            let bit = O::imm(sign.trailing_zeros().into());
            return self.a.put(C::Sbfm_64m_bitfield, &[out, a, bit, bit]);
        }
        self.a.put(C::And_64_log_imm, &[mag, a, O::imm(sign - 1)])?;
        match form.action {
            Action::CopySign => {
                self.a.put(C::And_64_log_imm, &[tmp, b, O::imm(sign)])?;
                self.a.put(C::Orr_64_log_shift, &[out, mag, tmp])
            }
            Action::Signum => {
                self.a.imm(R::X14, infinity)?;
                self.a.put(C::Subs_64_addsub_shift, &[zr, mag, tmp])?;
                self.a.put(C::And_64_log_imm, &[tmp, a, O::imm(sign)])?;
                self.a.imm(R::X15, bias << fraction)?;
                self.a.put(C::Orr_64_log_shift, &[tmp, tmp, x(15)])?;
                self.a.imm(R::X15, infinity | (min_normal >> 1))?;
                self.a
                    .put(C::Csel_64_condsel, &[out, x(15), tmp, O::cond(Cond::Hi)])
            }
            Action::Classify(op) => {
                use Unary::*;
                let (offset, limit, cond) = match op {
                    IsNan => (0, infinity, Cond::Hi),
                    IsInfinite => (0, infinity, Cond::Eq),
                    IsFinite => (0, infinity, Cond::Cc),
                    IsNormal => (min_normal, infinity - min_normal, Cond::Cc),
                    IsSubnormal => (1, min_normal - 1, Cond::Cc),
                    _ => unreachable!("selected FP classification"),
                };
                if offset != 0 {
                    self.a.imm(R::X14, offset)?;
                    self.a.put(C::Sub_64_addsub_shift, &[mag, mag, tmp])?;
                }
                self.a.imm(R::X14, limit)?;
                self.a.put(C::Subs_64_addsub_shift, &[zr, mag, tmp])?;
                self.bool_mask(cond, 12)?;
                self.a
                    .put(C::Sbfm_64m_bitfield, &[out, out, O::imm(0), O::imm(31)])
            }
            Action::MinMax(_) => unreachable!("handled with FP comparison"),
        }
    }
}
