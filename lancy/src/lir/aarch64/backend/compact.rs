use super::{A, C, Cond, Emitter, O, PredQual, R, Result, arrangements, groups::z};

impl Emitter<'_> {
    pub(super) fn compact(
        &mut self,
        size: usize,
        scalable: bool,
        value: u8,
        mask: u8,
        dst: u8,
    ) -> Result {
        if !scalable {
            return self.neon_compact(size, value, mask, dst);
        }
        if size >= 4 {
            let (arr, mask) = if size == 16 {
                let mask = z(mask, A::D);
                self.a.put(C::Trn2_z_zz, &[z(16, A::D), mask, mask])?;
                (A::D, 16)
            } else {
                (arrangements(size).1, mask)
            };
            self.sign_predicate(arr, mask)?;
            return self.a.put(
                C::Compact_z_p_z,
                &[z(dst, arr), O::pred(R::P1, PredQual::None), z(value, arr)],
            );
        }

        let arr = arrangements(size).1;
        let lanes = self.config.bytes / 4;
        let active = |p| O::pred(p, PredQual::None);
        self.predicate(R::P4, A::S, lanes)?;
        self.a.imm(R::X13, 0)?;
        for offset in (0..self.config.bytes).step_by(lanes * size) {
            self.sve_widen_group(mask, 17, size, offset, true)?;
            self.a.put(
                C::Cmplt_p_p_zi,
                &[
                    O::vec(R::P1, A::S),
                    O::pred(R::P4, PredQual::Zero),
                    z(17, A::S),
                    O::imm(0),
                ],
            )?;
            self.sve_widen_group(value, 16, size, offset, false)?;
            self.a
                .put(C::Compact_z_p_z, &[z(16, A::S), active(R::P1), z(16, A::S)])?;
            self.sve_narrow_group(16, size)?;
            if offset == 0 {
                self.copy_vec(dst, 16, true)?;
            } else {
                self.sve_prefix(R::P5, arr, R::X13)?;
                self.sve_splice(arr, dst, R::P5, 16)?;
            }
            // Append at the actual number of active lanes, which varies per group.
            self.a.put(
                C::Cntp_x_p_p,
                &[O::reg(R::X14), active(R::P4), O::vec(R::P1, A::S)],
            )?;
            self.a.put(
                C::Add_64_addsub_shift,
                &[O::reg(R::X13), O::reg(R::X13), O::reg(R::X14)],
            )?;
        }
        let zero = z(16, A::D);
        self.a.put(C::Eor_z_zz_, &[zero, zero, zero])?;
        self.sve_prefix(R::P5, arr, R::X13)?;
        self.a.put(
            C::Sel_z_p_zz,
            &[z(dst, arr), active(R::P5), z(dst, arr), z(16, arr)],
        )
    }

    fn neon_compact(&mut self, size: usize, value: u8, mask: u8, dst: u8) -> Result {
        let v = |n| O::vec(R::v(n).unwrap(), A::B16);
        let (d, src, scratch) = (v(dst), v(value), v(16));
        self.a.put(C::Eor_asimdsame_only, &[d, d, d])?;
        // Prepending active lanes in reverse order leaves their original order in the result.
        for lane in (0..16 / size).rev() {
            let end = (lane + 1) * size;
            self.neon_lane_to_int(mask, 9, 1, end - 1)?;
            self.a.put(
                C::Subs_32_addsub_imm,
                &[O::reg(R::WZR), O::reg(R::W9), O::imm(128)],
            )?;
            let skip = self.a.label();
            self.a.skip_if(Cond::Cs)?;
            self.a.jump(skip);
            let source = if end == 16 {
                src
            } else {
                self.a
                    .put(C::Ext_asimd, &[scratch, src, src, O::imm(end as u64)])?;
                scratch
            };
            self.a
                .put(C::Ext_asimd, &[d, source, d, O::imm((16 - size) as u64)])?;
            self.a.bind(skip);
        }
        Ok(())
    }
}
