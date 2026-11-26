use super::groups::z;
use super::inst::{IntBinary, IntOp, Packed, Select};
use super::{A, C, Cond, Emitter, O, PredQual, R, Result, ShiftKind, invert};
use crate::{
    Trap,
    hir::{Lane, Type},
    lir::fallback,
    support::numeric::Action,
};

impl Emitter<'_> {
    pub(super) fn fma(&mut self, code: C, arr: A, [a, b, c]: [u8; 3], dst: u8) -> Result {
        let v = |n| O::vec(R::v(n).unwrap(), arr);
        if code == C::Fmadd_scalar {
            return self.a.put(code, &[v(dst), v(a), v(b), v(c)]);
        }
        let acc = if dst != c && (dst == a || dst == b) {
            16
        } else {
            dst
        };
        let sve = code == C::Fmla_z_p_zz;
        self.copy_vec(acc, c, sve)?;
        if sve {
            let z = |n| O::vec(R::z(n).unwrap(), arr);
            self.predicate(R::P1, arr, self.config.bytes / sve_lane_bytes(arr))?;
            self.a
                .put(code, &[z(acc), O::pred(R::P1, PredQual::Merge), z(a), z(b)])?;
        } else {
            self.a.put(code, &[v(acc), v(a), v(b)])?;
        }
        self.copy_vec(dst, acc, sve)
    }

    pub(super) fn sign_predicate(&mut self, arr: A, mask: u8) -> Result {
        self.predicate(R::P2, arr, self.config.bytes / sve_lane_bytes(arr))?;
        self.a.put(
            C::Cmplt_p_p_zi,
            &[
                O::vec(R::P1, arr),
                O::pred(R::P2, PredQual::Zero),
                O::vec(R::z(mask).unwrap(), arr),
                O::imm(0),
            ],
        )
    }

    pub(super) fn select(&mut self, form: Select, mask: u8, yes: u8, no: u8, dst: u8) -> Result {
        let r = O::reg;
        let v = |n, a| O::vec(R::v(n).unwrap(), a);
        let z = |n, a| O::vec(R::z(n).unwrap(), a);
        match form {
            Select::Scalar(ty) => {
                self.a.put(
                    C::Subs_32_addsub_imm,
                    &[r(R::WZR), r(R::w(mask).unwrap()), O::imm(0)],
                )?;
                let cond = O::cond(Cond::Ne);
                let size = ty.bytes(self.config.bytes);
                match ty {
                    Type::F32 => self.a.put(
                        C::Fcsel_scalar,
                        &[v(dst, A::S), v(yes, A::S), v(no, A::S), cond],
                    ),
                    Type::F64 => self.a.put(
                        C::Fcsel_scalar,
                        &[v(dst, A::D), v(yes, A::D), v(no, A::D), cond],
                    ),
                    _ if size == 16 => {
                        let fallback = self.a.label();
                        let done = self.a.label();
                        self.a.skip_if(Cond::Ne)?;
                        self.a.jump(fallback);
                        self.copy_vec(dst, yes, false)?;
                        self.a.jump(done);
                        self.a.bind(fallback);
                        self.copy_vec(dst, no, false)?;
                        self.a.bind(done);
                        Ok(())
                    }
                    _ => {
                        let (code, reg): (_, fn(u8) -> Option<R>) = if size == 8 {
                            (C::Csel_64_condsel, R::x)
                        } else {
                            (C::Csel_32_condsel, R::w)
                        };
                        self.a.put(
                            code,
                            &[
                                r(reg(dst).unwrap()),
                                r(reg(yes).unwrap()),
                                r(reg(no).unwrap()),
                                cond,
                            ],
                        )
                    }
                }
            }
            Select::ScalarBits => {
                let x = |n| O::reg(R::x(n).unwrap());
                let tmp = x(9);
                self.a.put(C::Eor_64_log_shift, &[tmp, x(yes), x(no)])?;
                self.a.put(C::And_64_log_shift, &[tmp, tmp, x(mask)])?;
                self.a.put(C::Eor_64_log_shift, &[x(dst), x(no), tmp])
            }
            Select::Neon(arr) => {
                if let Some(arr) = arr {
                    self.a.sign_fill(v(16, arr), v(mask, arr))?;
                } else {
                    self.copy_vec(16, mask, false)?;
                }
                self.a.put(
                    C::Bsl_asimdsame,
                    &[v(16, A::B16), v(yes, A::B16), v(no, A::B16)],
                )?;
                self.copy_vec(dst, 16, false)
            }
            Select::Sve(Some(arr)) => {
                self.sign_predicate(arr, mask)?;
                self.a.put(
                    C::Sel_z_p_zz,
                    &[
                        z(dst, arr),
                        O::pred(R::P1, PredQual::None),
                        z(yes, arr),
                        z(no, arr),
                    ],
                )
            }
            Select::Sve(None) => {
                let z = |n| z(n, A::D);
                self.a.put(C::And_z_zz_, &[z(16), z(mask), z(yes)])?;
                self.a.put(C::Bic_z_zz_, &[z(dst), z(no), z(mask)])?;
                self.a.put(C::Orr_z_zz_, &[z(dst), z(dst), z(16)])
            }
            Select::Wide(scalable) => {
                let (code, value, form) = if scalable {
                    (C::Trn2_z_zz, z(16, A::D), Select::Sve(Some(A::D)))
                } else {
                    (C::Trn2_asimd, v(16, A::D2), Select::Neon(Some(A::D2)))
                };
                let mask = if scalable {
                    z(mask, A::D)
                } else {
                    v(mask, A::D2)
                };
                self.a.put(code, &[value, mask, mask])?;
                self.select(form, 16, yes, no, dst)
            }
        }
    }

    pub(super) fn sve_iota(&mut self, lane: Lane, dst: u8) -> Result {
        let size = lane.bytes();
        let arr = super::arrangements(size.min(8)).1;
        let d = O::vec(R::z(dst).unwrap(), arr);
        self.a.put(C::Index_z_ii, &[d, O::imm(0), O::imm(1)])?;
        if size == 16 {
            let zero = O::vec(R::Z16, A::D);
            self.a.put(C::Eor_z_zz_, &[zero, zero, zero])?;
            self.a.put(C::Zip1_z_zz, &[d, d, zero])?;
        } else if lane.float() {
            let code = if matches!(lane, Lane::F16 | Lane::BF16) {
                C::Ucvtf_half_z_p_z
            } else {
                C::Ucvtf_z_p_z
            };
            self.predicate(R::P1, arr, self.config.bytes / size)?;
            self.a.put(code, &[d, O::pred(R::P1, PredQual::Merge), d])?;
            if lane == Lane::BF16 {
                // Indices up to 31 are exact in both formats. Convert the
                // exponent bias after dropping the excess FP16 fraction bits.
                self.a.put(C::Lsr_z_zi, &[d, d, O::imm(3)])?;
                self.a.imm(R::W9, (127 - 15) << 7)?;
                let bias = z(16, A::H);
                self.a.put(C::Dup_z_r, &[bias, O::reg(R::W9)])?;
                self.a.put(C::Add_z_zz_, &[d, d, bias])?;
                self.predicate(R::P1, A::H, 1)?;
                self.a.put(
                    C::Cpy_z_p_i,
                    &[d, O::pred(R::P1, PredQual::Merge), O::imm(0)],
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn packed_cast(&mut self, code: C, arr: A, src: u8, dst: u8) -> Result {
        if matches!(arr, A::S | A::D) {
            self.predicate(R::P1, arr, self.config.bytes / sve_lane_bytes(arr))?;
            let z = |n| O::vec(R::z(n).unwrap(), arr);
            self.a
                .put(code, &[z(dst), O::pred(R::P1, PredQual::Merge), z(src)])
        } else {
            let v = |n| O::vec(R::v(n).unwrap(), arr);
            self.a.put(code, &[v(dst), v(src)])
        }
    }

    pub(super) fn float_cast(&mut self, code: C, from: Type, to: Type, src: u8, dst: u8) -> Result {
        let r = O::reg;
        let v = |n| R::v(n).unwrap();
        let gpr = |n, size| if size == 8 { R::x(n) } else { R::w(n) }.unwrap();
        let arr = |f| if f == Type::F32 { A::S } else { A::D };
        match code {
            C::Fcvt_s_d | C::Fcvt_d_s => self.a.put(code, &[r(v(dst)), r(v(src))]),
            C::Scvtf_scalar | C::Ucvtf_scalar => {
                let size = from.lane().unwrap().bytes();
                let mut input = gpr(src, size);
                if code == C::Scvtf_scalar && size < 4 {
                    self.a.put(
                        C::Sbfm_32m_bitfield,
                        &[r(R::W9), r(input), O::imm(0), O::imm((size * 8 - 1) as u64)],
                    )?;
                    input = R::W9;
                }
                self.a.put(code, &[O::vec(v(dst), arr(to)), r(input)])
            }
            C::Fcvtzs_scalar | C::Fcvtzu_scalar => {
                let size = to.lane().unwrap().bytes();
                let out = r(gpr(dst, size));
                self.a.put(code, &[out, O::vec(v(src), arr(from))])?;
                if size < 4 {
                    // FCVT saturates at the register width. Clamp again to
                    // the narrower destination before discarding upper bits.
                    let signed = code == C::Fcvtzs_scalar;
                    let bits = size * 8;
                    let limit = |this: &mut Self, value: u32, condition| {
                        this.a.imm(R::W9, value.into())?;
                        this.a
                            .put(C::Subs_32_addsub_shift, &[r(R::WZR), out, r(R::W9)])?;
                        this.a.put(
                            C::Csel_32_condsel,
                            &[out, r(R::W9), out, O::cond(condition)],
                        )
                    };
                    if signed {
                        limit(self, (-(1_i32 << (bits - 1))) as u32, Cond::Lt)?;
                        limit(self, (1_u32 << (bits - 1)) - 1, Cond::Gt)?;
                    } else {
                        limit(self, (1_u32 << bits) - 1, Cond::Hi)?;
                    }
                    self.a.put(
                        C::Ubfm_32m_bitfield,
                        &[out, out, O::imm(0), O::imm((bits - 1) as u64)],
                    )?;
                }
                Ok(())
            }
            _ => unreachable!("selected FP conversion"),
        }
    }

    pub(super) fn bool_mask(&mut self, condition: Cond, dst: u8) -> Result {
        self.a.put(
            C::Csinv_32_condsel,
            &[
                O::reg(R::w(dst).unwrap()),
                O::reg(R::WZR),
                O::reg(R::WZR),
                O::cond(invert(condition)),
            ],
        )
    }

    pub(super) fn mask_test(&mut self, all: bool, arr: A, src: u8, dst: u8) -> Result {
        let zero = if all { Cond::Eq } else { Cond::Ne };
        if matches!(arr, A::B | A::H | A::S | A::D) {
            self.predicate(R::P2, arr, self.config.bytes / sve_lane_bytes(arr))?;
            let code = if all {
                C::Cmpeq_p_p_zi
            } else {
                C::Cmpne_p_p_zi
            };
            self.a.put(
                code,
                &[
                    O::vec(R::P1, arr),
                    O::pred(R::P2, PredQual::Zero),
                    O::vec(R::z(src).unwrap(), arr),
                    O::imm(0),
                ],
            )?;
        } else {
            let mut v = R::v(src).unwrap();
            if all {
                self.a
                    .put(C::Cmeq0_asimdmisc, &[O::vec(R::V16, arr), O::vec(v, arr)])?;
                v = R::V16;
            }
            self.a
                .put(C::Umaxv_asimd, &[O::reg(R::V16), O::vec(v, A::B16)])?;
            self.a.put(
                C::Umov_lane,
                &[O::reg(R::W9), O::vec(R::V16, A::B16), O::imm(0)],
            )?;
            self.a.put(
                C::Subs_32_addsub_imm,
                &[O::reg(R::WZR), O::reg(R::W9), O::imm(0)],
            )?;
        }
        self.bool_mask(zero, dst)
    }

    pub(super) fn wide_mask_test(&mut self, all: bool, scalable: bool, src: u8, dst: u8) -> Result {
        if !scalable {
            return self.mask_test(false, A::B16, src, dst);
        }
        if !all {
            return self.mask_test(false, A::D, src, dst);
        }
        let input = z(src, A::D);
        let (low, high) = (z(16, A::D), z(self.temp(0), A::D));
        self.a.put(C::Trn1_z_zz, &[low, input, input])?;
        self.a.put(C::Trn2_z_zz, &[high, input, input])?;
        self.a.put(C::Orr_z_zz_, &[low, low, high])?;
        self.mask_test(true, A::D, 16, dst)
    }

    pub(super) fn lane_access(
        &mut self,
        ty: Type,
        src: u8,
        index: R,
        insert: Option<u8>,
        dst: u8,
    ) -> Result {
        let width = self.config.bytes;
        let size = ty.bytes(width);
        self.a.put(
            C::Subs_32_addsub_imm,
            &[O::reg(R::WZR), O::reg(index), O::imm((width / size) as u64)],
        )?;
        self.trap_if(Cond::Cs, Trap::Bounds)?;
        let temp = self.fallback_slot(0);
        self.access(false, width, src, temp)?;
        self.a.addr(R::X10, temp.0, temp.1)?;
        self.a.put(
            C::Add_64_addsub_sxtw,
            &[
                O::reg(R::X10),
                O::reg(R::X10),
                O::shifted(index, ShiftKind::Lsl, size.trailing_zeros() as u8),
            ],
        )?;
        if let Some(value) = insert {
            self.value_access(false, ty, value, (R::X10, 0))?;
            self.access(true, width, dst, temp)
        } else {
            self.value_access(true, ty, dst, (R::X10, 0))
        }
    }

    pub(super) fn float_move(
        &mut self,
        ty: Type,
        to_float: bool,
        vector: u8,
        integer: u8,
    ) -> Result {
        let v = O::reg(R::v(vector).unwrap());
        let w = O::reg(R::w(integer).unwrap());
        let x = O::reg(R::x(integer).unwrap());
        let (code, ops) = match (ty, to_float) {
            (Type::F32, true) => (C::Fmov_s_w, [v, w]),
            (Type::F64, true) => (C::Fmov_d_x, [v, x]),
            (Type::F32, false) => (C::Fmov_w_s, [w, v]),
            (Type::F64, false) => (C::Fmov_x_d, [x, v]),
            _ => unreachable!("floating-point register transfer"),
        };
        self.a.put(code, &ops)
    }

    pub(super) fn int_cast(&mut self, from: Type, to: Type, src: u8, dst: u8) -> Result {
        let from = from.lane().unwrap();
        let to = to.lane().unwrap();
        let (input, output) = (from.bytes(), to.bytes());
        let r = O::reg;
        let (ws, wd, xs, xd) = (
            r(R::w(src).unwrap()),
            r(R::w(dst).unwrap()),
            r(R::x(src).unwrap()),
            r(R::x(dst).unwrap()),
        );
        if input < output && from.signed() {
            if output == 8 {
                return self.a.put(
                    C::Sbfm_64m_bitfield,
                    &[xd, xs, O::imm(0), O::imm((input * 8 - 1) as u64)],
                );
            }
            self.a.put(
                C::Sbfm_32m_bitfield,
                &[wd, ws, O::imm(0), O::imm((input * 8 - 1) as u64)],
            )?;
            if output < 4 {
                self.a.put(
                    C::Ubfm_32m_bitfield,
                    &[wd, wd, O::imm(0), O::imm((output * 8 - 1) as u64)],
                )?;
            }
        } else if output < input && output < 4 {
            self.a.put(
                C::Ubfm_32m_bitfield,
                &[wd, ws, O::imm(0), O::imm((output * 8 - 1) as u64)],
            )?;
        } else if output == 4 && input == 8 {
            self.a.put(C::Orr_32_log_shift, &[wd, r(R::WZR), ws])?;
        } else if src != dst {
            self.a.put(C::Orr_64_log_shift, &[xd, r(R::XZR), xs])?;
        }
        Ok(())
    }

    pub(super) fn primitive(
        &mut self,
        action: Action,
        input: Type,
        output: Type,
        args: [Option<Type>; 3],
    ) -> Result {
        let site = fallback::Site::new(action, input, output, args, self.config.bytes);
        self.a.imm(R::X0, self.sites.add(site))?;
        self.a.addr(R::X1, R::X19, self.frame.temporary)?;
        self.a.imm(R::X16, fallback::entry())?;
        self.a.put(C::Blr_64_branch_reg, &[O::reg(R::X16)])?;
        self.byte_predicate = false;
        // W16 keeps the returned trap code for the side exit.
        self.a.put(
            C::Subs_32_addsub_imm,
            &[O::reg(R::W16), O::reg(R::W0), O::imm(0)],
        )?;
        self.report_if(Cond::Ne)?;
        Ok(())
    }

    pub(super) fn int_binary(&mut self, inst: IntBinary, lhs: u8, rhs: u8, dst: u8) -> Result {
        use IntOp::*;
        let size = inst.size;
        let wide = size == 8;
        let code = |word, double| if wide { double } else { word };
        let reg = |n| {
            if wide {
                O::reg(R::x(n).unwrap())
            } else {
                O::reg(R::w(n).unwrap())
            }
        };
        let (mut a, mut b, d, zero) = (reg(lhs), reg(rhs), reg(dst), reg(31));
        let (cmp, cmp_imm) = (
            code(C::Subs_32_addsub_shift, C::Subs_64_addsub_shift),
            code(C::Subs_32_addsub_imm, C::Subs_64_addsub_imm),
        );
        if size < 4 && inst.signed_inputs {
            let max = O::imm((size * 8 - 1) as u64);
            for (dst, src) in [(13, a), (14, b)] {
                self.a
                    .put(C::Sbfm_32m_bitfield, &[reg(dst), src, O::imm(0), max])?;
            }
            (a, b) = (reg(13), reg(14));
        }
        if inst.wrap_shift {
            self.a.put(
                C::Ubfm_32m_bitfield,
                &[reg(14), b, O::imm(0), O::imm(if size == 1 { 2 } else { 3 })],
            )?;
            b = reg(14);
        }
        match inst.op {
            Three(code) => self.a.put(code, &[d, a, b])?,
            Multiply(code) => self.a.put(code, &[d, a, b, zero])?,
            Compare(condition) => {
                self.a.put(cmp, &[zero, a, b])?;
                return self.bool_mask(condition, dst);
            }
            MinMax(cond) => {
                self.a.put(cmp, &[zero, a, b])?;
                self.a.put(
                    code(C::Csel_32_condsel, C::Csel_64_condsel),
                    &[d, a, b, O::cond(cond)],
                )?;
            }
            Divide { signed, remainder } => {
                self.a.put(cmp_imm, &[zero, b, O::imm(0)])?;
                self.trap_if(Cond::Eq, Trap::DivisionByZero)?;
                let div = if signed {
                    let safe = self.a.label();
                    let cmn = code(C::Adds_32_addsub_imm, C::Adds_64_addsub_imm);
                    self.a.put(cmn, &[zero, b, O::imm(1)])?;
                    self.a.skip_if(Cond::Eq)?;
                    self.a.jump(safe);
                    let min = i64::MIN >> (64 - size * 8);
                    let min_reg = if wide { R::X15 } else { R::W15 };
                    self.a.imm(min_reg, min as u64)?;
                    self.a.put(cmp, &[zero, a, reg(15)])?;
                    self.trap_if(Cond::Eq, Trap::DivisionOverflow)?;
                    self.a.bind(safe);
                    code(C::Sdiv_32_dp_2src, C::Sdiv_64_dp_2src)
                } else {
                    code(C::Udiv_32_dp_2src, C::Udiv_64_dp_2src)
                };
                if remainder {
                    self.a.put(div, &[reg(15), a, b])?;
                    self.a.put(
                        code(C::Msub_32a_dp_3src, C::Msub_64a_dp_3src),
                        &[d, reg(15), b, a],
                    )?;
                } else {
                    self.a.put(div, &[d, a, b])?;
                }
            }
        }
        if size < 4 {
            self.a.put(
                C::Ubfm_32m_bitfield,
                &[d, d, O::imm(0), O::imm((size * 8 - 1) as u64)],
            )?;
        }
        Ok(())
    }

    pub(super) fn packed(&mut self, op: Packed, a: u8, b: u8, d: u8) -> Result {
        let z = |n, a| O::vec(R::z(n).unwrap(), a);
        let v = |n, a| O::vec(R::v(n).unwrap(), a);
        match op {
            Packed::Neon(code, arr) => self.a.put(code, &[v(d, arr), v(a, arr), v(b, arr)]),
            Packed::Sve(code, arr) => self.a.put(code, &[z(d, arr), z(a, arr), z(b, arr)]),
            Packed::NeonMinMax64(cmp, max) => {
                let (lhs, rhs) = if max { (a, b) } else { (b, a) };
                self.a
                    .put(cmp, &[v(16, A::D2), v(lhs, A::D2), v(rhs, A::D2)])?;
                let v = |n| v(n, A::B16);
                if d == a {
                    self.a.put(C::Bif_asimd, &[v(d), v(b), v(16)])
                } else if d == b {
                    self.a.put(C::Bit_asimd, &[v(d), v(a), v(16)])
                } else {
                    self.a.put(C::Bsl_asimdsame, &[v(16), v(a), v(b)])?;
                    self.copy_vec(d, 16, false)
                }
            }
            Packed::NeonShift(code, arr, right) => {
                let (bits, mask_arr) = match arr {
                    A::B16 => (8, A::B16),
                    A::H8 => (16, A::H8),
                    A::S4 => (32, A::S4),
                    // SSHL/USHL only read the low byte of each shift count.
                    A::D2 => (64, A::S4),
                    _ => unreachable!("NEON shift lane arrangement"),
                };
                self.a
                    .put(C::Movi_asimd, &[v(16, mask_arr), O::imm(bits - 1)])?;
                let mask = v(16, A::B16);
                self.a
                    .put(C::And_asimdsame_only, &[mask, v(b, A::B16), mask])?;
                let count = v(16, arr);
                if right {
                    self.a.put(C::Neg_asimd, &[count, count])?;
                }
                self.a.put(code, &[v(d, arr), v(a, arr), count])
            }
            Packed::SveShift(code, arr) => {
                let bits = (sve_lane_bytes(arr) * 8) as u64;
                self.a.put(C::Dupm_z, &[z(16, arr), O::imm(bits - 1)])?;
                let mask = z(16, A::D);
                self.a.put(C::And_z_zz_, &[mask, z(b, A::D), mask])?;
                self.copy_vec(d, a, true)?;
                self.sve_binary(code, arr, d, 16)
            }
            Packed::FloatOrder(arr, unordered) => {
                if matches!(arr, A::S | A::D) {
                    self.predicate(R::P2, arr, self.config.bytes / sve_lane_bytes(arr))?;
                    self.a.put(
                        C::Fcmuo_p_p_zz,
                        &[
                            O::vec(R::P1, arr),
                            O::pred(R::P2, PredQual::Zero),
                            z(a, arr),
                            z(b, arr),
                        ],
                    )?;
                    if !unordered {
                        let q = |r| O::vec(r, A::B);
                        self.a.put(
                            C::Bic_p_p_pp,
                            &[q(R::P1), O::pred(R::P2, PredQual::Zero), q(R::P2), q(R::P1)],
                        )?;
                    }
                    return self.a.put(
                        C::Cpy_z_p_i,
                        &[z(d, arr), O::pred(R::P1, PredQual::Zero), O::simm(-1)],
                    );
                }
                self.a
                    .put(C::Fcmeq_asimdsame, &[v(16, arr), v(a, arr), v(a, arr)])?;
                self.a
                    .put(C::Fcmeq_asimdsame, &[v(d, arr), v(b, arr), v(b, arr)])?;
                let (d, temp) = (v(d, A::B16), v(16, A::B16));
                self.a.put(C::And_asimdsame_only, &[d, d, temp])?;
                if unordered {
                    self.a.put(C::Mvn_asimdmisc, &[d, d])?;
                }
                Ok(())
            }
            Packed::NeonCompare {
                code,
                arrangement: arr,
                invert,
            } => {
                self.a.put(code, &[v(d, arr), v(a, arr), v(b, arr)])?;
                if invert {
                    self.a
                        .put(C::Mvn_asimdmisc, &[v(d, A::B16), v(d, A::B16)])?;
                }
                Ok(())
            }
            Packed::SveCompare {
                code,
                arrangement: arr,
            } => {
                self.predicate(R::P2, arr, self.config.bytes / sve_lane_bytes(arr))?;
                self.a.put(
                    code,
                    &[
                        O::vec(R::P1, arr),
                        O::pred(R::P2, PredQual::Zero),
                        z(a, arr),
                        z(b, arr),
                    ],
                )?;
                self.a.put(
                    C::Cpy_z_p_i,
                    &[z(d, arr), O::pred(R::P1, PredQual::Zero), O::simm(-1)],
                )
            }
            Packed::SveDestructive(code, arr, commutative) => {
                let (src, rhs, write) = if d == b && a != b {
                    if commutative { (b, a, d) } else { (a, b, 16) }
                } else {
                    (a, b, d)
                };
                self.copy_vec(write, src, true)?;
                self.sve_binary(code, arr, write, rhs)?;
                self.copy_vec(d, write, true)
            }
        }
    }

    fn sve_binary(&mut self, code: C, arr: A, dst: u8, rhs: u8) -> Result {
        self.predicate(R::P1, arr, self.config.bytes / sve_lane_bytes(arr))?;
        let z = |n| O::vec(R::z(n).unwrap(), arr);
        self.a.put(
            code,
            &[z(dst), O::pred(R::P1, PredQual::Merge), z(dst), z(rhs)],
        )
    }
}

pub(super) fn sve_lane_bytes(a: A) -> usize {
    match a {
        A::B => 1,
        A::H => 2,
        A::S => 4,
        A::D => 8,
        _ => unreachable!("SVE lane arrangement"),
    }
}
