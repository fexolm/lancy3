use super::{A, C, Cond, Emitter, O, PredQual, R, Result, ShiftKind, arrangements, groups::z};

impl Emitter<'_> {
    pub(super) fn sve_masked(
        &mut self,
        form: (C, A),
        base: R,
        mask: u8,
        value: u8,
        indices: Option<u8>,
        dst: Option<u8>,
    ) -> Result {
        self.sign_predicate(form.1, mask)?;
        self.sve_masked_data(form, base, value, indices, dst)
    }

    fn sve_masked_data(
        &mut self,
        (code, arr): (C, A),
        base: R,
        value: u8,
        indices: Option<u8>,
        dst: Option<u8>,
    ) -> Result {
        let loaded = dst.map_or(value, |dst| if dst == value { 16 } else { dst });
        let qualifier = if dst.is_some() {
            PredQual::Zero
        } else {
            PredQual::None
        };
        let operands = [
            z(loaded, arr),
            O::pred(R::P1, qualifier),
            O::mem(base),
            z(indices.unwrap_or(16), arr),
        ];
        self.a
            .put(code, &operands[..3 + usize::from(indices.is_some())])?;
        if let Some(dst) = dst {
            self.merge_gather(arr, dst, loaded, value)?;
        }
        Ok(())
    }

    pub(super) fn sve_wide_masked(
        &mut self,
        code: C,
        base: R,
        mask: u8,
        value: u8,
        indices: Option<u8>,
        dst: Option<u8>,
    ) -> Result {
        let mask = z(mask, A::D);
        let d16 = z(16, A::D);
        self.a.put(C::Trn2_z_zz, &[d16, mask, mask])?;
        self.sign_predicate(A::D, 16)?;
        let indices = if let Some(indices) = indices {
            // The low word of the original index suffices for wrapped addresses.
            // Each 128-bit lane addresses the two consecutive doublewords 2*i and 2*i+1.
            let i = z(indices, A::D);
            let parity = z(self.temp(0), A::D);
            self.a.put(C::Trn1_z_zz, &[d16, i, i])?;
            self.a.put(C::Index_z_ii, &[parity, O::imm(0), O::imm(1)])?;
            self.a.put(C::And_z_zi, &[parity, O::imm(1)])?;
            self.a.put(C::Add_z_zz_, &[d16, d16, d16])?;
            self.a.put(C::Add_z_zz_, &[d16, d16, parity])?;
            Some(16)
        } else {
            None
        };
        self.sve_masked_data((code, A::D), base, value, indices, dst)
    }

    pub(super) fn sve_narrow_masked(
        &mut self,
        (code, size): (C, usize),
        base: R,
        [mask, value, indices]: [u8; 3],
        dst: Option<u8>,
    ) -> Result {
        let arr = arrangements(size).1;
        let lanes = self.config.bytes / 4;
        self.predicate(R::P4, A::S, lanes)?;
        // RCJHWV orders same-address stores within a single SVE instruction.
        // Emitting groups in lane order extends this guarantee across the logical vector.
        let group = self.temp(0);
        for offset in (0..self.config.bytes).step_by(lanes * size) {
            self.sve_widen_group(mask, group, size, offset, true)?;
            self.a.put(
                C::Cmplt_p_p_zi,
                &[
                    O::vec(R::P1, A::S),
                    O::pred(R::P4, PredQual::Zero),
                    z(group, A::S),
                    O::imm(0),
                ],
            )?;
            self.sve_widen_group(indices, 16, size, offset, true)?;
            let ix = z(16, A::S);
            if let Some(dst) = dst {
                self.a.put(
                    code,
                    &[ix, O::pred(R::P1, PredQual::Zero), O::mem(base), ix],
                )?;
                self.sve_narrow_group(16, size)?;
                if offset == 0 {
                    self.copy_vec(dst, 16, true)?;
                } else {
                    self.a.imm(R::X13, (offset / size) as u64)?;
                    self.sve_prefix(R::P5, arr, R::X13)?;
                    self.sve_splice(arr, dst, R::P5, 16)?;
                }
            } else {
                self.sve_widen_group(value, group, size, offset, false)?;
                self.a.put(
                    code,
                    &[
                        z(group, A::S),
                        O::pred(R::P1, PredQual::None),
                        O::mem(base),
                        ix,
                    ],
                )?;
            }
        }
        if let Some(dst) = dst {
            self.sign_predicate(arr, mask)?;
            self.merge_gather(arr, dst, dst, value)?;
        }
        Ok(())
    }

    fn merge_gather(&mut self, arr: A, dst: u8, loaded: u8, value: u8) -> Result {
        self.a.put(
            C::Sel_z_p_zz,
            &[
                z(dst, arr),
                O::pred(R::P1, PredQual::None),
                z(loaded, arr),
                z(value, arr),
            ],
        )
    }

    pub(super) fn neon_masked(
        &mut self,
        size: usize,
        base: R,
        mask: u8,
        value: u8,
        indices: Option<u8>,
        dst: Option<u8>,
    ) -> Result {
        if let Some(dst) = dst {
            self.copy_vec(dst, value, false)?;
        }
        for lane in 0..16 / size {
            let skip = self.a.label();
            self.neon_lane_to_int(mask, 9, 1, (lane + 1) * size - 1)?;
            self.a.put(
                C::Subs_32_addsub_imm,
                &[O::reg(R::WZR), O::reg(R::W9), O::imm(128)],
            )?;
            self.a.skip_if(Cond::Cs)?;
            self.a.jump(skip);

            let address = if let Some(indices) = indices {
                self.neon_lane_to_int(indices, 9, size.min(8), lane)?;
                if size < 8 {
                    self.a.put(
                        C::Sbfm_64m_bitfield,
                        &[
                            O::reg(R::X9),
                            O::reg(R::X9),
                            O::imm(0),
                            O::imm((8 * size - 1) as u64),
                        ],
                    )?;
                }
                self.a.put(
                    C::Add_64_addsub_shift,
                    &[
                        O::reg(R::X10),
                        O::reg(base),
                        O::shifted(R::X9, ShiftKind::Lsl, size.trailing_zeros() as u8),
                    ],
                )?;
                (R::X10, 0)
            } else {
                (base, lane * size)
            };
            match (dst, size) {
                (Some(dst), 16) => self.a.access(true, size, dst, address)?,
                (None, 16) => self.a.access(false, size, value, address)?,
                (Some(dst), _) => {
                    self.a.access(true, size, 9, address)?;
                    self.neon_lane_from_int(9, dst, size, lane)?;
                }
                (None, _) => {
                    self.neon_lane_to_int(value, 9, size, lane)?;
                    self.a.access(false, size, 9, address)?;
                }
            }
            self.a.bind(skip);
        }
        Ok(())
    }
}
