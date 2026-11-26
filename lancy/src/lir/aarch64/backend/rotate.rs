use super::{A, C, Emitter, O, PredQual, R, Result, ShiftKind, arrangements, groups::z};
use crate::hir::Type;

#[derive(Clone, Copy, Debug)]
pub(super) struct Form {
    pub size: usize,
    pub packed: bool,
    pub scalable: bool,
    pub right: bool,
}

impl Emitter<'_> {
    pub(super) fn rotate(&mut self, form: Form, src: u8, count: u8, dst: u8) -> Result {
        if !form.packed {
            return self.scalar_rotate(form, src, count, dst);
        }
        let bits = (8 * form.size) as u64;
        let (neon, scalable) = arrangements(form.size);
        if form.scalable {
            let lane = |n| z(n, scalable);
            let temp = self.temp(0);
            let (acc, data, counts) = (lane(dst), lane(16), lane(temp));
            self.copy_vec(16, src, true)?;
            self.copy_vec(dst, src, true)?;
            self.copy_vec(temp, count, true)?;
            self.a.put(C::And_z_zi, &[counts, O::imm(bits - 1)])?;
            self.predicate(R::P1, scalable, self.config.bytes / form.size)?;
            let active = O::pred(R::P1, PredQual::Merge);
            let (first, second) = if form.right {
                (C::Lsr_z_p_zz, C::Lsl_z_p_zz)
            } else {
                (C::Lsl_z_p_zz, C::Lsr_z_p_zz)
            };
            self.a.put(first, &[acc, active, acc, counts])?;
            self.a.put(C::Neg_z_p_z, &[counts, active, counts])?;
            self.a.put(C::And_z_zi, &[counts, O::imm(bits - 1)])?;
            self.a.put(second, &[data, active, data, counts])?;
            return self
                .a
                .put(C::Orr_z_zz_, &[z(dst, A::D), z(dst, A::D), z(16, A::D)]);
        }
        let v = |n, arr| O::vec(R::v(n).unwrap(), arr);
        let temp = self.temp(0);
        let (acc, data, counts, constant) =
            (v(dst, neon), v(src, neon), v(16, neon), v(temp, neon));
        let raw = |n| v(n, A::B16);
        self.neon_splat_int(neon, bits - 1, temp)?;
        self.a
            .put(C::And_asimdsame_only, &[raw(16), raw(count), raw(temp)])?;
        if form.right {
            self.a.put(C::Neg_asimd, &[counts, counts])?;
            self.a
                .put(C::And_asimdsame_only, &[raw(16), raw(16), raw(temp)])?;
        }
        self.a.put(C::Ushl_asimd, &[acc, data, counts])?;
        self.neon_splat_int(neon, bits, temp)?;
        self.a
            .put(C::Sub_asimdsame_only, &[counts, counts, constant])?;
        self.a.put(C::Ushl_asimd, &[counts, data, counts])?;
        self.a
            .put(C::Orr_asimdsame_only, &[raw(dst), raw(dst), raw(16)])
    }

    fn scalar_rotate(&mut self, form: Form, src: u8, count: u8, dst: u8) -> Result {
        let wide = form.size == 8;
        let reg = |n| if wide { R::x(n) } else { R::w(n) }.unwrap();
        let gpr = |n| O::reg(reg(n));
        let data = if form.size < 4 {
            let width = form.size * 8;
            self.a.put(
                C::And_32_log_imm,
                &[gpr(9), gpr(src), O::imm((1 << width) - 1)],
            )?;
            // RORV operates on a whole word. Repeat the byte or halfword so
            // every possible rotation wraps identically at the HIR lane width.
            for shift in [8, 16].into_iter().filter(|&shift| shift >= width) {
                self.a.put(
                    C::Orr_32_log_shift,
                    &[
                        gpr(9),
                        gpr(9),
                        O::shifted(R::W9, ShiftKind::Lsl, shift as u8),
                    ],
                )?;
            }
            gpr(9)
        } else {
            gpr(src)
        };
        let count = if form.right {
            gpr(count)
        } else {
            let neg = if wide {
                C::Sub_64_addsub_shift
            } else {
                C::Sub_32_addsub_shift
            };
            self.a.put(neg, &[gpr(10), gpr(31), gpr(count)])?;
            gpr(10)
        };
        let code = if wide {
            C::Rorv_64_dp_2src
        } else {
            C::Rorv_32_dp_2src
        };
        self.a.put(code, &[gpr(dst), data, count])?;
        if form.size < 4 {
            self.a.put(
                C::Ubfm_32m_bitfield,
                &[
                    gpr(dst),
                    gpr(dst),
                    O::imm(0),
                    O::imm((form.size * 8 - 1) as u64),
                ],
            )?;
        }
        Ok(())
    }

    pub(super) fn scalar_saturate(
        &mut self,
        code: C,
        size: usize,
        a: u8,
        b: u8,
        dst: u8,
    ) -> Result {
        let ty = if size == 8 { Type::F64 } else { Type::F32 };
        self.float_move(ty, true, 16, a)?;
        let rhs = self.temp(0);
        self.float_move(ty, true, rhs, b)?;
        let arr = arrangements(size).0;
        let d = O::vec(R::V16, arr);
        self.a.put(code, &[d, d, O::vec(R::v(rhs).unwrap(), arr)])?;
        self.neon_lane_to_int(16, dst, size, 0)
    }

    pub(super) fn neon_mul64(&mut self, a: u8, b: u8, dst: u8) -> Result {
        let (x9, x10) = (O::reg(R::X9), O::reg(R::X10));
        for lane in 0..2 {
            self.neon_lane_to_int(a, 9, 8, lane)?;
            self.neon_lane_to_int(b, 10, 8, lane)?;
            self.a
                .put(C::Madd_64a_dp_3src, &[x9, x9, x10, O::reg(R::XZR)])?;
            self.neon_lane_from_int(9, dst, 8, lane)?;
        }
        Ok(())
    }
}
