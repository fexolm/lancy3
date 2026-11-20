use super::{A, C, Cond, Emitter, O, R, Result, ShiftKind};
use crate::hir::{Binary, Lane, Type};

#[derive(Clone, Copy, Debug)]
pub(super) struct Form {
    pub from: Lane,
    pub to: Lane,
    pub packed: bool,
    pub scalable: bool,
}

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
    pub(super) fn integer_float_cast(&mut self, form: Form, src: u8, dst: u8) -> Result {
        let Form {
            from,
            to,
            packed,
            scalable,
        } = form;
        if packed {
            // HIR casts preserve lane width, so only the half formats reach this path.
            if scalable {
                return self.sve_each_lane(A::H, |e| {
                    e.sve_lane_to_int(A::H, src, 9)?;
                    e.integer_float_half_lane(form, 9, 12)?;
                    e.sve_lane_from_int(A::H, 12, dst, R::P1)
                });
            }
            for lane in 0..8 {
                self.neon_lane_to_int(src, 9, 2, lane)?;
                self.integer_float_half_lane(form, 9, 12)?;
                self.neon_lane_from_int(12, dst, 2, lane)?;
            }
            return Ok(());
        }
        if to.bytes() == 16 {
            if from.bytes() == 2 {
                self.a.put(C::Orr_64_log_shift, &[x(9), x(31), x(src)])?;
            } else {
                self.float_move(from.scalar(), false, src, 9)?;
            }
            self.float_to_wide(from, to.signed())?;
            return self.pair_from_gpr(dst, 9, 10);
        }
        if from.bytes() == 16 {
            self.pair_to_gpr(src, 9, 10)?;
            self.integer_to_float_bits(from.signed(), to)?;
            if to.bytes() > 2 {
                return self.float_move(to.scalar(), true, dst, 9);
            }
            return self.a.put(C::Orr_64_log_shift, &[x(dst), x(31), x(9)]);
        }
        self.integer_float_half_lane(form, src, dst)
    }

    pub(super) fn integer_float_half_lane(&mut self, form: Form, src: u8, dst: u8) -> Result {
        let Form { from, to, .. } = form;
        if from.float() {
            self.decode_half(from, src, 16)?;
            let code = if to.signed() {
                C::Fcvtzs_scalar
            } else {
                C::Fcvtzu_scalar
            };
            return self.float_cast(code, Type::F64, to.scalar(), 16, dst);
        }
        let base = if from.signed() { Type::I64 } else { Type::U64 };
        self.int_cast(from.scalar(), base, src, 9)?;
        if from.signed() {
            self.a
                .put(C::Sbfm_64m_bitfield, &[x(10), x(9), O::imm(63), O::imm(63)])?;
        } else {
            self.a.imm(R::X10, 0)?;
        }
        self.integer_to_float_bits(from.signed(), to)?;
        self.a.put(C::Orr_64_log_shift, &[x(dst), x(31), x(9)])
    }

    pub(super) fn float_to_wide(&mut self, from: Lane, signed: bool) -> Result {
        let (fraction, bias, sign) = float_layout(from);
        let hidden = 1 << fraction;
        let zero = self.a.label();
        let saturate = self.a.label();
        let right = self.a.label();
        let apply_sign = self.a.label();
        let done = self.a.label();
        self.a.put(
            C::Sbfm_64m_bitfield,
            &[
                x(16),
                x(9),
                O::imm(sign.trailing_zeros().into()),
                O::imm(sign.trailing_zeros().into()),
            ],
        )?;
        if !signed {
            self.a
                .put(C::Subs_64_addsub_imm, &[x(31), x(16), O::imm(0)])?;
            self.a.jump_if(Cond::Mi, zero)?;
        }
        self.a
            .put(C::And_64_log_imm, &[x(9), x(9), O::imm(sign - 1)])?;
        self.a.imm(R::X11, sign - hidden)?;
        self.a.put(C::Subs_64_addsub_shift, &[x(31), x(9), x(11)])?;
        self.a.jump_if(Cond::Hi, zero)?;
        self.a.jump_if(Cond::Eq, saturate)?;
        self.a.put(
            C::Ubfm_64m_bitfield,
            &[x(13), x(9), O::imm(fraction), O::imm(63)],
        )?;
        self.a
            .put(C::Subs_64_addsub_imm, &[x(13), x(13), O::imm(bias)])?;
        self.a.jump_if(Cond::Mi, zero)?;
        self.a.put(
            C::Subs_64_addsub_imm,
            &[x(31), x(13), O::imm(if signed { 127 } else { 128 })],
        )?;
        self.a.jump_if(Cond::Cs, saturate)?;
        self.a
            .put(C::And_64_log_imm, &[x(9), x(9), O::imm(hidden - 1)])?;
        self.a
            .put(C::Orr_64_log_imm, &[x(9), x(9), O::imm(hidden)])?;
        self.a.imm(R::X10, 0)?;
        self.a
            .put(C::Subs_64_addsub_imm, &[x(13), x(13), O::imm(fraction)])?;
        self.a.jump_if(Cond::Mi, right)?;
        self.wide_shift_pair(Binary::Shl, false)?;
        self.a.jump(apply_sign);
        self.a.bind(right);
        self.a.put(C::Sub_64_addsub_shift, &[x(13), x(31), x(13)])?;
        self.a.put(C::Lsrv_64_dp_2src, &[x(9), x(9), x(13)])?;
        self.a.bind(apply_sign);
        if signed {
            self.negate_pair_if_negative(9, 10, 16)?;
        }
        self.a.jump(done);
        self.a.bind(saturate);
        if signed {
            self.a.put(C::Orn_64_log_shift, &[x(9), x(31), x(16)])?;
            self.a.imm(R::X10, i64::MAX as u64)?;
            self.a.put(C::Eor_64_log_shift, &[x(10), x(10), x(16)])?;
        } else {
            self.a.imm(R::X9, u64::MAX)?;
            self.a.imm(R::X10, u64::MAX)?;
        }
        self.a.jump(done);
        self.a.bind(zero);
        self.a.imm(R::X9, 0)?;
        self.a.imm(R::X10, 0)?;
        self.a.bind(done);
        Ok(())
    }

    pub(super) fn integer_to_float_bits(&mut self, signed: bool, to: Lane) -> Result {
        let (fraction, bias, sign) = float_layout(to);
        let high_nonzero = self.a.label();
        let normalized = self.a.label();
        let increment = self.a.label();
        let pack = self.a.label();
        let infinity = self.a.label();
        let done = self.a.label();
        if signed {
            self.a.put(
                C::Sbfm_64m_bitfield,
                &[x(16), x(10), O::imm(63), O::imm(63)],
            )?;
            self.negate_pair_if_negative(9, 10, 16)?;
            self.a
                .put(C::And_64_log_imm, &[x(16), x(16), O::imm(sign)])?;
        }
        self.a
            .put(C::Subs_64_addsub_imm, &[x(31), x(10), O::imm(0)])?;
        self.a.jump_if(Cond::Ne, high_nonzero)?;
        self.a
            .put(C::Subs_64_addsub_imm, &[x(31), x(9), O::imm(0)])?;
        self.a.jump_if(Cond::Eq, done)?;
        self.a.put(C::Clz_64, &[x(13), x(9)])?;
        self.a.imm(R::X14, bias + 62)?;
        self.a.put(C::Sub_64_addsub_shift, &[x(14), x(14), x(13)])?;
        self.a.put(C::Lslv_64_dp_2src, &[x(10), x(9), x(13)])?;
        self.a.imm(R::X9, 0)?;
        self.a.jump(normalized);
        self.a.bind(high_nonzero);
        self.a.put(C::Clz_64, &[x(13), x(10)])?;
        self.a.imm(R::X14, bias + 126)?;
        self.a.put(C::Sub_64_addsub_shift, &[x(14), x(14), x(13)])?;
        self.wide_shift_pair(Binary::Shl, false)?;
        self.a.bind(normalized);
        if to == Lane::F16 {
            self.a
                .put(C::Subs_64_addsub_imm, &[x(31), x(14), O::imm(29)])?;
            self.a.jump_if(Cond::Hi, infinity)?;
        }
        // The top bit is now at bit 127. The tail is compared to a half ulp,
        // retaining the lower word as sticky data even across the word boundary.
        self.a.put(
            C::Ubfm_64m_bitfield,
            &[x(11), x(10), O::imm(63 - fraction), O::imm(63)],
        )?;
        self.a.put(
            C::Orr_64_log_shift,
            &[
                x(12),
                x(31),
                O::shifted(R::X10, ShiftKind::Lsl, (fraction + 1) as u8),
            ],
        )?;
        self.a.imm(R::X15, 1 << 63)?;
        self.a
            .put(C::Subs_64_addsub_shift, &[x(31), x(12), x(15)])?;
        self.a.jump_if(Cond::Hi, increment)?;
        self.a.jump_if(Cond::Cc, pack)?;
        self.a.put(C::And_64_log_imm, &[x(12), x(11), O::imm(1)])?;
        self.a.put(C::Orr_64_log_shift, &[x(12), x(12), x(9)])?;
        self.a
            .put(C::Subs_64_addsub_imm, &[x(31), x(12), O::imm(0)])?;
        self.a.jump_if(Cond::Eq, pack)?;
        self.a.bind(increment);
        self.a
            .put(C::Add_64_addsub_imm, &[x(11), x(11), O::imm(1)])?;
        self.a.bind(pack);
        self.a.put(
            C::Add_64_addsub_shift,
            &[
                x(9),
                x(11),
                O::shifted(R::X14, ShiftKind::Lsl, fraction as u8),
            ],
        )?;
        if to == Lane::F16 {
            let with_sign = self.a.label();
            self.a.jump(with_sign);
            self.a.bind(infinity);
            self.a.imm(R::X9, sign - (1 << fraction))?;
            self.a.bind(with_sign);
        }
        if signed {
            self.a.put(C::Orr_64_log_shift, &[x(9), x(9), x(16)])?;
        }
        self.a.bind(done);
        Ok(())
    }
}
