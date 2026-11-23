use super::{A, C, Cond, Emitter, O, PredQual, R, Result, invert};
use crate::Trap;
use crate::hir::{Binary, Type, Unary};

#[derive(Clone, Copy, Debug)]
pub(super) struct MathForm {
    pub op: Binary,
    pub signed: bool,
    pub packed: bool,
    pub scalable: bool,
}

fn x(n: u8) -> O {
    O::reg(R::x(n).unwrap())
}

impl MathForm {
    fn comparison(self) -> bool {
        matches!(
            self.op,
            Binary::Eq | Binary::Ne | Binary::Lt | Binary::Le | Binary::Gt | Binary::Ge
        )
    }
}

impl Emitter<'_> {
    pub(super) fn wide_math(&mut self, form: MathForm, a: u8, b: u8, dst: u8) -> Result {
        if form.scalable {
            return self.wide_lanes(a, Some(b), dst, |e| e.wide_math_pair(form));
        }
        self.pair_to_gpr(a, 9, 10)?;
        self.pair_to_gpr(b, 13, 14)?;
        self.wide_math_pair(form)?;
        if form.comparison() && !form.packed {
            self.a.put(
                C::Orr_32_log_shift,
                &[O::reg(R::w(dst).unwrap()), O::reg(R::WZR), O::reg(R::W9)],
            )
        } else {
            self.pair_from_gpr(dst, 9, 10)
        }
    }

    fn wide_lanes(
        &mut self,
        a: u8,
        b: Option<u8>,
        dst: u8,
        operation: impl FnOnce(&mut Self) -> Result,
    ) -> Result {
        self.sve_each_lane(A::D, |e| {
            let p = |r| O::vec(r, A::B);
            e.a.put(
                C::Orr_p_p_pp,
                &[p(R::P2), O::pred(R::P1, PredQual::Zero), p(R::P1), p(R::P1)],
            )?;
            e.sve_lane_to_int(A::D, a, 9)?;
            if let Some(b) = b {
                e.sve_lane_to_int(A::D, b, 13)?;
            }
            e.sve_next_lane(A::D)?;
            e.sve_lane_to_int(A::D, a, 10)?;
            if let Some(b) = b {
                e.sve_lane_to_int(A::D, b, 14)?;
            }
            operation(e)?;
            e.sve_lane_from_int(A::D, 9, dst, R::P2)?;
            e.sve_lane_from_int(A::D, 10, dst, R::P1)
        })
    }

    pub(super) fn wide_unary(&mut self, op: Unary, scalable: bool, src: u8, dst: u8) -> Result {
        if scalable {
            return self.wide_lanes(src, None, dst, |e| e.wide_unary_pair(op));
        }
        self.pair_to_gpr(src, 9, 10)?;
        self.wide_unary_pair(op)?;
        self.pair_from_gpr(dst, 9, 10)
    }

    fn wide_unary_pair(&mut self, op: Unary) -> Result {
        use Unary::*;
        match op {
            Neg => {
                self.a.put(C::Subs_64_addsub_shift, &[x(9), x(31), x(9)])?;
                self.a.put(C::Sbc_64, &[x(10), x(31), x(10)])
            }
            Abs => self.negate_pair_if_negative(9, 10, 10),
            Popcnt => {
                self.int_unary(op, 8, 9, 11)?;
                self.int_unary(op, 8, 10, 12)?;
                self.a.put(C::Add_64_addsub_shift, &[x(9), x(11), x(12)])?;
                self.a.imm(R::X10, 0)
            }
            LeadingZeros | TrailingZeros => {
                let (first, second) = if op == LeadingZeros { (10, 9) } else { (9, 10) };
                self.int_unary(op, 8, first, 11)?;
                self.int_unary(op, 8, second, 12)?;
                self.a
                    .put(C::Add_64_addsub_imm, &[x(12), x(12), O::imm(64)])?;
                self.a
                    .put(C::Subs_64_addsub_imm, &[x(31), x(first), O::imm(0)])?;
                self.a
                    .put(C::Csel_64_condsel, &[x(9), x(12), x(11), O::cond(Cond::Eq)])?;
                self.a.imm(R::X10, 0)
            }
            _ => unreachable!("selected 128-bit unary operation"),
        }
    }

    // Operands (X9:X10 and X13:X14) and the result (X9:X10) are low word first.
    fn wide_math_pair(&mut self, form: MathForm) -> Result {
        use Binary::*;
        match form.op {
            Add => {
                self.a.put(C::Adds_64_addsub_shift, &[x(9), x(9), x(13)])?;
                self.a.put(C::Adc_64, &[x(10), x(10), x(14)])
            }
            And | Or | Xor => {
                let code = match form.op {
                    And => C::And_64_log_shift,
                    Or => C::Orr_64_log_shift,
                    _ => C::Eor_64_log_shift,
                };
                self.a.put(code, &[x(9), x(9), x(13)])?;
                self.a.put(code, &[x(10), x(10), x(14)])
            }
            Mul => {
                self.a.put(C::Umulh_64, &[x(11), x(9), x(13)])?;
                self.a
                    .put(C::Madd_64a_dp_3src, &[x(11), x(9), x(14), x(11)])?;
                self.a
                    .put(C::Madd_64a_dp_3src, &[x(10), x(10), x(13), x(11)])?;
                self.a.put(C::Madd_64a_dp_3src, &[x(9), x(9), x(13), x(31)])
            }
            Div | Rem => self.wide_divide_pair(form.signed, form.op == Rem),
            Shl | Shr | RotateLeft | RotateRight => self.wide_shift_pair(form.op, form.signed),
            SaturatingAdd | SaturatingSub => {
                self.wide_saturate_pair(form.signed, form.op == SaturatingSub)
            }
            op => {
                let cond = self.wide_compare_pair(op, form.signed)?;
                if matches!(op, Min | Max) {
                    self.a
                        .put(C::Csel_64_condsel, &[x(9), x(9), x(13), O::cond(cond)])?;
                    self.a
                        .put(C::Csel_64_condsel, &[x(10), x(10), x(14), O::cond(cond)])
                } else {
                    self.bool_mask(cond, 9)?;
                    if form.packed {
                        self.a
                            .put(C::Sbfm_64m_bitfield, &[x(9), x(9), O::imm(0), O::imm(31)])?;
                        self.a.put(C::Orr_64_log_shift, &[x(10), x(31), x(9)])?;
                    }
                    Ok(())
                }
            }
        }
    }

    pub(super) fn wide_reduce(&mut self, form: MathForm, src: u8, dst: u8) -> Result {
        if !form.scalable {
            return self.copy_vec(dst, src, false);
        }
        self.pair_to_gpr(src, 9, 10)?;
        self.sve_lanes_after(A::D, 2, |e| {
            e.sve_lane_to_int(A::D, src, 13)?;
            e.sve_next_lane(A::D)?;
            e.sve_lane_to_int(A::D, src, 14)?;
            e.wide_math_pair(form)
        })?;
        self.pair_from_gpr(dst, 9, 10)
    }

    pub(super) fn wide_shift_pair(&mut self, op: Binary, signed: bool) -> Result {
        let rotate = matches!(op, Binary::RotateLeft | Binary::RotateRight);
        let left = matches!(op, Binary::Shl | Binary::RotateLeft);
        let count = O::reg(R::W13);
        self.a
            .put(C::And_32_log_imm, &[count, count, O::imm(127)])?;
        let small = self.a.label();
        let done = self.a.label();
        self.a
            .put(C::Subs_64_addsub_imm, &[x(31), x(13), O::imm(64)])?;
        self.a.skip_if(Cond::Cs)?;
        self.a.jump(small);
        if rotate {
            for (dst, src) in [(11, 9), (9, 10), (10, 11)] {
                self.a.put(C::Orr_64_log_shift, &[x(dst), x(31), x(src)])?;
            }
        } else {
            if left {
                self.a.put(C::Lslv_64_dp_2src, &[x(10), x(9), x(13)])?;
                self.a.imm(R::X9, 0)?;
            } else {
                let right = if signed {
                    C::Asrv_64_dp_2src
                } else {
                    C::Lsrv_64_dp_2src
                };
                self.a.put(right, &[x(9), x(10), x(13)])?;
                if signed {
                    self.a.put(
                        C::Sbfm_64m_bitfield,
                        &[x(10), x(10), O::imm(63), O::imm(63)],
                    )?;
                } else {
                    self.a.imm(R::X10, 0)?;
                }
            }
            self.a.jump(done);
        }
        self.a.bind(small);
        if rotate {
            self.a.put(C::And_32_log_imm, &[count, count, O::imm(63)])?;
        }
        self.a
            .put(C::Subs_64_addsub_imm, &[x(31), x(13), O::imm(0)])?;
        self.a.skip_if(Cond::Ne)?;
        self.a.jump(done);
        self.a.put(C::Sub_64_addsub_shift, &[x(15), x(31), x(13)])?;
        let (shift, cross) = if left {
            (C::Lslv_64_dp_2src, C::Lsrv_64_dp_2src)
        } else {
            (C::Lsrv_64_dp_2src, C::Lslv_64_dp_2src)
        };
        if rotate {
            self.a.put(cross, &[x(11), x(10), x(15)])?;
            self.a.put(cross, &[x(12), x(9), x(15)])?;
            self.a.put(shift, &[x(9), x(9), x(13)])?;
            self.a.put(shift, &[x(10), x(10), x(13)])?;
            self.a.put(C::Orr_64_log_shift, &[x(9), x(9), x(11)])?;
            self.a.put(C::Orr_64_log_shift, &[x(10), x(10), x(12)])?;
        } else if left {
            self.a.put(cross, &[x(11), x(9), x(15)])?;
            self.a.put(shift, &[x(9), x(9), x(13)])?;
            self.a.put(shift, &[x(10), x(10), x(13)])?;
            self.a.put(C::Orr_64_log_shift, &[x(10), x(10), x(11)])?;
        } else {
            self.a.put(cross, &[x(11), x(10), x(15)])?;
            self.a.put(shift, &[x(9), x(9), x(13)])?;
            let upper = if signed { C::Asrv_64_dp_2src } else { shift };
            self.a.put(upper, &[x(10), x(10), x(13)])?;
            self.a.put(C::Orr_64_log_shift, &[x(9), x(9), x(11)])?;
        }
        self.a.bind(done);
        Ok(())
    }

    fn wide_saturate_pair(&mut self, signed: bool, subtract: bool) -> Result {
        let (low, high) = if subtract {
            (C::Subs_64_addsub_shift, C::Sbcs_64)
        } else {
            (C::Adds_64_addsub_shift, C::Adcs_64)
        };
        self.a.put(low, &[x(11), x(9), x(13)])?;
        self.a.put(high, &[x(12), x(10), x(14)])?;
        let (low_bound, high_bound, fits) = if signed {
            self.a.put(
                C::Sbfm_64m_bitfield,
                &[x(15), x(10), O::imm(63), O::imm(63)],
            )?;
            self.a.put(C::Orn_64_log_shift, &[x(16), x(31), x(15)])?;
            self.a.imm(R::X17, i64::MAX as u64)?;
            self.a.put(C::Eor_64_log_shift, &[x(15), x(15), x(17)])?;
            (16, 15, Cond::Vc)
        } else if subtract {
            (31, 31, Cond::Cs)
        } else {
            self.a.put(C::Orn_64_log_shift, &[x(15), x(31), x(31)])?;
            (15, 15, Cond::Cc)
        };
        self.a.put(
            C::Csel_64_condsel,
            &[x(9), x(11), x(low_bound), O::cond(fits)],
        )?;
        self.a.put(
            C::Csel_64_condsel,
            &[x(10), x(12), x(high_bound), O::cond(fits)],
        )
    }

    fn wide_compare_pair(&mut self, op: Binary, signed: bool) -> Result<Cond> {
        use Binary::*;
        if matches!(op, Eq | Ne) {
            self.a.put(C::Eor_64_log_shift, &[x(11), x(9), x(13)])?;
            self.a.put(C::Eor_64_log_shift, &[x(12), x(10), x(14)])?;
            self.a.put(C::Orr_64_log_shift, &[x(11), x(11), x(12)])?;
            self.a
                .put(C::Subs_64_addsub_imm, &[x(31), x(11), O::imm(0)])?;
            return Ok(if op == Eq { Cond::Eq } else { Cond::Ne });
        }
        let (a, ah, b, bh) = if matches!(op, Gt | Le) {
            (13, 14, 9, 10)
        } else {
            (9, 10, 13, 14)
        };
        self.a.put(C::Subs_64_addsub_shift, &[x(31), x(a), x(b)])?;
        self.a.put(C::Sbcs_64, &[x(31), x(ah), x(bh)])?;
        let less = if signed { Cond::Lt } else { Cond::Cc };
        Ok(if matches!(op, Ge | Le | Max) {
            invert(less)
        } else {
            less
        })
    }

    fn wide_divide_pair(&mut self, signed: bool, remainder: bool) -> Result {
        self.a.put(C::Orr_64_log_shift, &[x(11), x(13), x(14)])?;
        self.a
            .put(C::Subs_64_addsub_imm, &[x(31), x(11), O::imm(0)])?;
        self.trap_if(Cond::Eq, Trap::DivisionByZero)?;
        if signed {
            self.a.imm(R::X11, 1 << 63)?;
            self.a.put(C::Eor_64_log_shift, &[x(11), x(10), x(11)])?;
            self.a.put(C::Orr_64_log_shift, &[x(11), x(11), x(9)])?;
            for word in [13, 14] {
                self.a
                    .put(C::Add_64_addsub_imm, &[x(12), x(word), O::imm(1)])?;
                self.a.put(C::Orr_64_log_shift, &[x(11), x(11), x(12)])?;
            }
            self.a
                .put(C::Subs_64_addsub_imm, &[x(31), x(11), O::imm(0)])?;
            self.trap_if(Cond::Eq, Trap::DivisionOverflow)?;
            let sign = if remainder {
                10
            } else {
                self.a.put(C::Eor_64_log_shift, &[x(11), x(10), x(14)])?;
                11
            };
            self.float_move(Type::F64, true, 16, sign)?;
            self.negate_pair_if_negative(9, 10, 10)?;
            self.negate_pair_if_negative(13, 14, 14)?;
        }

        let wide = self.a.label();
        let done = self.a.label();
        self.a.put(C::Orr_64_log_shift, &[x(11), x(10), x(14)])?;
        self.a
            .put(C::Subs_64_addsub_imm, &[x(31), x(11), O::imm(0)])?;
        self.a.skip_if(Cond::Eq)?;
        self.a.jump(wide);
        let quotient = if remainder { 11 } else { 9 };
        self.a
            .put(C::Udiv_64_dp_2src, &[x(quotient), x(9), x(13)])?;
        if remainder {
            self.a
                .put(C::Msub_64a_dp_3src, &[x(9), x(11), x(13), x(9)])?;
        }
        self.a.jump(done);

        self.a.bind(wide);
        self.a.imm(R::X11, 0)?;
        self.a.imm(R::X12, 0)?;
        self.a.imm(R::X17, 128)?;
        let step = self.a.label();
        self.a.bind(step);
        // Shift the dividend's next bit into the remainder. The vacated
        // low bit in X9 becomes the new quotient bit.
        self.a.put(C::Adds_64_addsub_shift, &[x(9), x(9), x(9)])?;
        self.a.put(C::Adcs_64, &[x(10), x(10), x(10)])?;
        self.a.put(C::Adcs_64, &[x(11), x(11), x(11)])?;
        self.a.put(C::Adc_64, &[x(12), x(12), x(12)])?;
        self.a
            .put(C::Subs_64_addsub_shift, &[x(15), x(11), x(13)])?;
        self.a.put(C::Sbcs_64, &[x(16), x(12), x(14)])?;
        self.a.put(
            C::Csel_64_condsel,
            &[x(11), x(15), x(11), O::cond(Cond::Cs)],
        )?;
        self.a.put(
            C::Csel_64_condsel,
            &[x(12), x(16), x(12), O::cond(Cond::Cs)],
        )?;
        self.a.put(C::Adc_64, &[x(9), x(9), x(31)])?;
        self.a
            .put(C::Subs_64_addsub_imm, &[x(17), x(17), O::imm(1)])?;
        self.a.skip_if(Cond::Eq)?;
        self.a.jump(step);
        if remainder {
            self.a.put(C::Orr_64_log_shift, &[x(9), x(31), x(11)])?;
            self.a.put(C::Orr_64_log_shift, &[x(10), x(31), x(12)])?;
        }
        self.a.bind(done);
        if signed {
            self.float_move(Type::F64, false, 16, 11)?;
            self.negate_pair_if_negative(9, 10, 11)?;
        }
        Ok(())
    }

    pub(super) fn negate_pair_if_negative(&mut self, low: u8, high: u8, sign: u8) -> Result {
        let done = self.a.label();
        self.a
            .put(C::Subs_64_addsub_imm, &[x(31), x(sign), O::imm(0)])?;
        self.a.skip_if(Cond::Mi)?;
        self.a.jump(done);
        self.a
            .put(C::Subs_64_addsub_shift, &[x(low), x(31), x(low)])?;
        self.a.put(C::Sbc_64, &[x(high), x(31), x(high)])?;
        self.a.bind(done);
        Ok(())
    }

    pub(super) fn wide_add(
        &mut self,
        subtract: bool,
        scalable: bool,
        a: u8,
        b: u8,
        dst: u8,
    ) -> Result {
        if scalable {
            let z = |n| O::vec(R::z(n).unwrap(), A::D);
            let p = |r| O::vec(r, A::D);
            let scratch = z(16);
            let code = if subtract { C::Sub_z_zz_ } else { C::Add_z_zz_ };
            self.a.put(code, &[scratch, z(a), z(b)])?;
            self.predicate(R::P2, A::D, self.config.bytes / 8)?;
            let (hi, lo) = if subtract {
                (z(b), z(a))
            } else {
                (z(a), scratch)
            };
            self.a.put(
                C::Cmphi_p_p_zz,
                &[p(R::P1), O::pred(R::P2, PredQual::Zero), hi, lo],
            )?;
            // Carry/borrow from each even doubleword affects only its paired odd doubleword.
            self.a.put(C::Pfalse_p, &[O::vec(R::P3, A::B)])?;
            self.a.put(C::Trn1_p_pp, &[p(R::P1), p(R::P3), p(R::P1)])?;
            self.a
                .put(C::Cpy_one_z_p, &[z(dst), O::pred(R::P1, PredQual::Zero)])?;
            return self.a.put(code, &[z(dst), scratch, z(dst)]);
        }
        self.pair_to_gpr(a, 9, 10)?;
        self.pair_to_gpr(b, 13, 14)?;
        let (low, high) = if subtract {
            (C::Subs_64_addsub_shift, C::Sbc_64)
        } else {
            (C::Adds_64_addsub_shift, C::Adc_64)
        };
        let r = O::reg;
        self.a.put(low, &[r(R::X9), r(R::X9), r(R::X13)])?;
        self.a.put(high, &[r(R::X10), r(R::X10), r(R::X14)])?;
        self.pair_from_gpr(dst, 9, 10)
    }

    pub(super) fn wide_cast(&mut self, from: Type, to: Type, src: u8, dst: u8) -> Result {
        match (from.bytes(16), to.bytes(16)) {
            (16, 16) => self.copy_vec(dst, src, false),
            (16, 8) => self.float_move(Type::F64, false, src, dst),
            (16, _) => {
                self.float_move(Type::F64, false, src, 9)?;
                self.int_cast(Type::U64, to, 9, dst)
            }
            (_, 16) => {
                let (low, high) = if from.lane().unwrap().signed() {
                    self.int_cast(from, Type::I64, src, 9)?;
                    self.a.put(
                        C::Sbfm_64m_bitfield,
                        &[O::reg(R::X10), O::reg(R::X9), O::imm(63), O::imm(63)],
                    )?;
                    (9, 10)
                } else {
                    (src, 31)
                };
                self.pair_from_gpr(dst, low, high)
            }
            _ => unreachable!("wide integer cast"),
        }
    }

    pub(super) fn pair_to_gpr(&mut self, src: u8, low: u8, high: u8) -> Result {
        self.float_move(Type::F64, false, src, low)?;
        self.a.put(
            C::Mov_x_v_d1,
            &[O::reg(R::x(high).unwrap()), O::reg(R::v(src).unwrap())],
        )
    }

    pub(super) fn pair_from_gpr(&mut self, dst: u8, low: u8, high: u8) -> Result {
        self.float_move(Type::F64, true, dst, low)?;
        self.a.put(
            C::Mov_v_d1_x,
            &[O::reg(R::v(dst).unwrap()), O::reg(R::x(high).unwrap())],
        )
    }
}
