use super::integer_float::float_layout;
use super::{C, Cond, Emitter, O, R, Result, ShiftKind, arrangements};
use crate::hir::Lane;

#[derive(Clone, Copy, Debug)]
pub(super) struct Form {
    pub lane: Lane,
    pub packed: bool,
    pub scalable: bool,
}

fn x(n: u8) -> O {
    O::reg(R::x(n).unwrap())
}

impl Emitter<'_> {
    pub(super) fn float_remainder(&mut self, form: Form, lhs: u8, rhs: u8, dst: u8) -> Result {
        let size = form.lane.bytes();
        if !form.packed {
            let ty = form.lane.scalar();
            if size == 2 {
                let w = |r| O::reg(R::w(r).unwrap());
                let zero = O::reg(R::WZR);
                self.a.put(C::Orr_32_log_shift, &[w(9), zero, w(lhs)])?;
                self.a.put(C::Orr_32_log_shift, &[w(10), zero, w(rhs)])?;
                self.remainder_bits(form.lane)?;
                return self.a.put(C::Orr_32_log_shift, &[w(dst), zero, w(9)]);
            }
            self.float_move(ty, false, lhs, 9)?;
            self.float_move(ty, false, rhs, 10)?;
            self.remainder_bits(form.lane)?;
            return self.float_move(ty, true, dst, 9);
        }
        if form.scalable {
            let arr = arrangements(size).1;
            return self.sve_each_lane(arr, |e| {
                e.sve_lane_to_int(arr, lhs, 9)?;
                e.sve_lane_to_int(arr, rhs, 10)?;
                e.remainder_bits(form.lane)?;
                e.sve_lane_from_int(arr, 9, dst, R::P1)
            });
        }
        for lane in 0..16 / size {
            self.neon_lane_to_int(lhs, 9, size, lane)?;
            self.neon_lane_to_int(rhs, 10, size, lane)?;
            self.remainder_bits(form.lane)?;
            self.neon_lane_from_int(9, dst, size, lane)?;
        }
        Ok(())
    }

    fn remainder_bits(&mut self, lane: Lane) -> Result {
        let (fraction, _, sign) = float_layout(lane);
        let hidden = 1_u64 << fraction;
        let infinity = sign - hidden;
        let quiet = hidden >> 1;
        let (a, b, sign_a, ea, eb, temp, one, chunk, sign_b) =
            (x(9), x(10), x(11), x(12), x(13), x(14), x(15), x(16), x(17));
        let zr = O::reg(R::XZR);
        let invalid = self.a.label();
        let nan_a = self.a.label();
        let nan_b = self.a.label();
        let pack = self.a.label();
        let loop_mod = self.a.label();
        let with_sign = self.a.label();
        let done = self.a.label();

        self.a.put(C::And_64_log_imm, &[sign_a, a, O::imm(sign)])?;
        self.a.put(C::And_64_log_imm, &[sign_b, b, O::imm(sign)])?;
        self.a.put(C::And_64_log_imm, &[a, a, O::imm(sign - 1)])?;
        self.a.put(C::And_64_log_imm, &[b, b, O::imm(sign - 1)])?;
        self.a.imm(R::X14, infinity)?;
        self.a.put(C::Subs_64_addsub_shift, &[zr, a, temp])?;
        self.a.skip_if(Cond::Ls)?;
        self.a.jump(nan_a);
        self.a.put(C::Subs_64_addsub_shift, &[zr, b, temp])?;
        self.a.skip_if(Cond::Ls)?;
        self.a.jump(nan_b);
        self.a.put(C::Subs_64_addsub_shift, &[zr, a, temp])?;
        self.a.skip_if(Cond::Ne)?;
        self.a.jump(invalid);
        self.a.put(C::Subs_64_addsub_imm, &[zr, b, O::imm(0)])?;
        self.a.skip_if(Cond::Ne)?;
        self.a.jump(invalid);
        self.a.put(C::Subs_64_addsub_shift, &[zr, a, b])?;
        self.a.skip_if(Cond::Cs)?;
        self.a.jump(with_sign);

        // Subnormals share the least normal exponent and have no hidden bit.
        // After reducing the significand, append zero bits for the exponent
        // difference in chunks that fit in a single unsigned division.
        self.a.imm(R::X15, 1)?;
        for (value, exp) in [(a, ea), (b, eb)] {
            self.a.put(
                C::Ubfm_64m_bitfield,
                &[exp, value, O::imm(fraction), O::imm(63)],
            )?;
            self.a
                .put(C::And_64_log_imm, &[value, value, O::imm(hidden - 1)])?;
            self.a.put(C::Subs_64_addsub_imm, &[zr, exp, O::imm(0)])?;
            self.a
                .put(C::Orr_64_log_imm, &[temp, value, O::imm(hidden)])?;
            self.a
                .put(C::Csel_64_condsel, &[value, temp, value, O::cond(Cond::Ne)])?;
            self.a
                .put(C::Csel_64_condsel, &[exp, exp, one, O::cond(Cond::Ne)])?;
        }
        let max_chunk = 63 - fraction;
        self.a.imm(R::X16, max_chunk)?;
        self.a.put(C::Sub_64_addsub_shift, &[ea, ea, eb])?;
        self.a.bind(loop_mod);
        self.a.put(C::Udiv_64_dp_2src, &[temp, a, b])?;
        self.a.put(C::Msub_64a_dp_3src, &[a, temp, b, a])?;
        self.a.put(C::Subs_64_addsub_imm, &[zr, a, O::imm(0)])?;
        self.a.skip_if(Cond::Ne)?;
        self.a.jump(with_sign);
        self.a.put(C::Subs_64_addsub_imm, &[zr, ea, O::imm(0)])?;
        self.a.skip_if(Cond::Ne)?;
        self.a.jump(pack);
        self.a.put(C::Subs_64_addsub_shift, &[zr, ea, chunk])?;
        self.a
            .put(C::Csel_64_condsel, &[temp, ea, chunk, O::cond(Cond::Ls)])?;
        self.a.put(C::Lslv_64_dp_2src, &[a, a, temp])?;
        self.a.put(C::Sub_64_addsub_shift, &[ea, ea, temp])?;
        self.a.jump(loop_mod);

        self.a.bind(pack);
        // The explicit significand adds the low bit of a normal exponent.
        // Reaching the least exponent also packs subnormals without rounding.
        self.a.put(C::Clz_64, &[temp, a])?;
        self.a
            .put(C::Sub_64_addsub_imm, &[temp, temp, O::imm(max_chunk)])?;
        self.a.put(C::Sub_64_addsub_imm, &[eb, eb, O::imm(1)])?;
        self.a.put(C::Subs_64_addsub_shift, &[zr, temp, eb])?;
        self.a
            .put(C::Csel_64_condsel, &[temp, temp, eb, O::cond(Cond::Ls)])?;
        self.a.put(C::Lslv_64_dp_2src, &[a, a, temp])?;
        self.a.put(C::Sub_64_addsub_shift, &[eb, eb, temp])?;
        self.a.put(
            C::Add_64_addsub_shift,
            &[a, a, O::shifted(R::X13, ShiftKind::Lsl, fraction as u8)],
        )?;
        self.a.jump(with_sign);

        self.a.bind(nan_b);
        self.a.put(C::Orr_64_log_shift, &[a, zr, b])?;
        self.a.put(C::Orr_64_log_shift, &[sign_a, zr, sign_b])?;
        self.a.bind(nan_a);
        if lane.bytes() == 2 {
            self.a.imm(R::X9, infinity | quiet)?;
        } else {
            self.a.put(C::Orr_64_log_imm, &[a, a, O::imm(quiet)])?;
        }
        self.a.jump(with_sign);
        self.a.bind(invalid);
        self.a.imm(R::X9, infinity | quiet)?;
        self.a.jump(done);
        self.a.bind(with_sign);
        self.a.put(C::Orr_64_log_shift, &[a, a, sign_a])?;
        self.a.bind(done);
        Ok(())
    }
}
