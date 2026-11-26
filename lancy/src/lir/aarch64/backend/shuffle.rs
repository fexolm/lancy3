use super::{
    A, C, Cond, Emitter, O, PredQual, R, Result,
    groups::z,
    inst::{Expand, Shuffle},
    operations::sve_lane_bytes,
    sve,
};
use crate::hir::Type;
use splice_target::Profile;

impl Emitter<'_> {
    pub(super) fn shuffle(&mut self, form: Shuffle, args: [u8; 3], dst: u8) -> Result {
        match form {
            Shuffle::Sve(arr) => self.sve_shuffle(arr, args, dst),
            Shuffle::Neon(arr, size) => self.neon_shuffle(arr, size, args, dst),
            Shuffle::Wide(scalable) => self.wide_shuffle(scalable, args, dst),
        }
    }

    fn wide_shuffle(&mut self, scalable: bool, [a, b, indices]: [u8; 3], dst: u8) -> Result {
        if !scalable {
            self.pair_to_gpr(indices, 9, 10)?;
            let d = O::vec(R::v(dst).unwrap(), A::B16);
            self.a.put(C::Eor_asimdsame_only, &[d, d, d])?;
            let done = self.a.label();
            let second = self.a.label();
            let zero = O::reg(R::XZR);
            self.a
                .put(C::Subs_64_addsub_imm, &[zero, O::reg(R::X10), O::imm(0)])?;
            self.a.jump_if(Cond::Ne, done)?;
            self.a
                .put(C::Subs_64_addsub_imm, &[zero, O::reg(R::X9), O::imm(1)])?;
            self.a.jump_if(Cond::Hi, done)?;
            self.a.jump_if(Cond::Eq, second)?;
            self.copy_vec(dst, a, false)?;
            self.a.jump(done);
            self.a.bind(second);
            self.copy_vec(dst, b, false)?;
            self.a.bind(done);
            return Ok(());
        }
        let indices = z(indices, A::D);
        let index = z(17, A::D);
        let scratch = z(dst, A::D);
        let p = |r| O::vec(r, A::D);
        let lanes = self.config.bytes / 8;
        self.predicate(R::P4, A::D, lanes)?;
        let active = O::pred(R::P4, PredQual::Zero);
        // Check the entire index before doubling it to a pair of table offsets.
        self.a.put(C::Trn2_z_zz, &[scratch, indices, indices])?;
        self.a
            .put(C::Cmpne_p_p_zi, &[p(R::P5), active, scratch, O::imm(0)])?;
        self.a.put(C::Trn1_z_zz, &[index, indices, indices])?;
        self.a.put(
            C::Cmphi_p_p_zi,
            &[p(R::P6), active, index, O::imm((lanes - 1) as u64)],
        )?;
        let bytes = |r| O::vec(r, A::B);
        self.a.put(
            C::Orr_p_p_pp,
            &[bytes(R::P5), active, bytes(R::P5), bytes(R::P6)],
        )?;
        self.a
            .put(C::Index_z_ii, &[scratch, O::imm(0), O::imm(1)])?;
        self.a.put(C::And_z_zi, &[scratch, O::imm(1)])?;
        self.a.put(C::Add_z_zz_, &[index, index, index])?;
        self.a.put(C::Add_z_zz_, &[index, index, scratch])?;
        self.sve_shuffle(A::D, [a, b, 17], dst)?;
        self.a.put(
            C::Cpy_z_p_i,
            &[scratch, O::pred(R::P5, PredQual::Merge), O::imm(0)],
        )
    }

    fn sve_shuffle(&mut self, arr: A, [a, b, indices]: [u8; 3], dst: u8) -> Result {
        let lanes = self.config.bytes / sve_lane_bytes(arr);
        let z = |n| O::vec(R::z(n).unwrap(), arr);
        let p = |r| O::pred(r, PredQual::Zero);
        let (idx, scratch, out) = (z(indices), z(16), z(dst));
        self.predicate(R::P3, arr, lanes)?;
        // Physical TBL may read beyond the logical width. Choose the source using
        // logical lane numbers and independently zero anything past both tables.
        self.a.put(
            C::Cmpls_p_p_zi,
            &[
                O::vec(R::P1, arr),
                p(R::P3),
                idx,
                O::imm((lanes - 1) as u64),
            ],
        )?;
        self.a.put(
            C::Cmphi_p_p_zi,
            &[
                O::vec(R::P2, arr),
                p(R::P3),
                idx,
                O::imm((2 * lanes - 1) as u64),
            ],
        )?;
        self.copy_vec(16, indices, true)?;
        self.a.put(C::Sub_z_zi, &[scratch, O::imm(lanes as u64)])?;
        self.a.put(C::Tbl_z_zz, &[scratch, z(b), scratch])?;
        self.a.put(C::Tbl_z_zz, &[out, z(a), idx])?;
        self.a.put(
            C::Sel_z_p_zz,
            &[out, O::pred(R::P1, PredQual::None), out, scratch],
        )?;
        self.a.put(
            C::Cpy_z_p_i,
            &[out, O::pred(R::P2, PredQual::Merge), O::imm(0)],
        )
    }

    fn neon_shuffle(&mut self, arr: A, size: usize, [a, b, indices]: [u8; 3], dst: u8) -> Result {
        let v = |n, a| O::vec(R::v(n).unwrap(), a);
        let bytes = |n| v(n, A::B16);
        let (index, scratch, out) = (bytes(indices), bytes(16), bytes(dst));
        if size == 1 {
            self.a.put(C::Movi_asimd, &[out, O::imm(16)])?;
            self.a.put(C::Sub_asimdsame_only, &[scratch, index, out])?;
            self.a.put(C::Tbl_asimd, &[out, bytes(a), index])?;
            return self.a.put(C::Tbx_asimd, &[out, bytes(b), scratch]);
        }

        self.neon_byte_indices(arr, size, indices, dst)?;
        self.neon_splat_int(arr, (32 / size) as u64, 16)?;
        // Check the entire original index before TBL sees the truncated byte offsets.
        self.a.put(
            C::Cmhs_asimdsame,
            &[v(16, arr), v(indices, arr), v(16, arr)],
        )?;
        self.a.put(C::Orr_asimdsame_only, &[out, out, scratch])?;
        self.copy_vec(16, a, false)?;
        self.copy_vec(17, b, false)?;
        // TBL2 also reads the implicit register immediately following V16.
        self.a.put(C::Tbl2_asimd, &[out, scratch, out])
    }

    pub(super) fn neon_splat_int(&mut self, arr: A, bits: u64, dst: u8) -> Result {
        let d = O::vec(R::v(dst).unwrap(), arr);
        if bits <= 255 && arr != A::D2 {
            return self.a.put(C::Movi_asimd, &[d, O::imm(bits)]);
        }
        let reg = if arr == A::D2 { R::X9 } else { R::W9 };
        self.a.imm(reg, bits)?;
        self.a.put(C::Dup_asimdgen, &[d, O::reg(reg)])
    }

    fn neon_byte_indices(&mut self, arr: A, size: usize, indices: u8, dst: u8) -> Result {
        let v = |n, a| O::vec(R::v(n).unwrap(), a);
        let (index, scratch, out) = (v(indices, A::B16), v(16, A::B16), v(dst, A::B16));
        if size == 1 {
            return self.copy_vec(dst, indices, false);
        }
        if size == 8 {
            // Repeat byte 0 for the first index and byte 8 for the second.
            self.literal(Type::U128, 16, 0x0808080808080808_0000000000000000)?;
            self.a.put(C::Tbl_asimd, &[out, index, scratch])?;
            self.a.put(C::Shl_asimd_imm, &[out, out, O::imm(3)])?;
        } else {
            // For valid indices, multiplication broadcasts the scaled byte
            // offset without a carry into the next byte.
            self.a.put(C::Movi_asimd, &[scratch, O::imm(size as u64)])?;
            self.a.put(
                C::Mul_asimdsame_only,
                &[v(dst, arr), v(indices, arr), v(16, arr)],
            )?;
        }
        let offsets = u64::from_le_bytes(std::array::from_fn(|i| (i % size) as u8));
        self.neon_splat_int(arr, offsets, 16)?;
        self.a.put(C::Add_asimdsame_only, &[out, out, scratch])
    }

    pub(super) fn expand(&mut self, form: Expand, data: u8, mask: u8, dst: u8) -> Result {
        match form {
            Expand::Sve2(arr) => {
                self.sign_predicate(arr, mask)?;
                self.sve2_expand(arr, data, dst)
            }
            Expand::Sve(arr) => {
                self.sign_predicate(arr, mask)?;
                self.sve_expand(arr, data, dst)
            }
            Expand::Neon(arr, size) => self.neon_expand(arr, size, data, mask, dst),
            Expand::Wide(profile) => self.wide_expand(profile, data, mask, dst),
        }
    }

    fn wide_expand(&mut self, profile: Profile, data: u8, mask: u8, dst: u8) -> Result {
        if sve(profile) {
            let mask = z(mask, A::D);
            self.a.put(C::Trn2_z_zz, &[z(16, A::D), mask, mask])?;
            self.sign_predicate(A::D, 16)?;
            return if profile == Profile::Sve2 {
                self.sve2_expand(A::D, data, dst)
            } else {
                self.sve_expand(A::D, data, dst)
            };
        }
        let v = |n, arr| O::vec(R::v(n).unwrap(), arr);
        let scratch = v(16, A::D2);
        let mask = v(mask, A::D2);
        self.a.put(C::Trn2_asimd, &[scratch, mask, mask])?;
        self.a.sign_fill(scratch, scratch)?;
        self.a.put(
            C::And_asimdsame_only,
            &[v(dst, A::B16), v(data, A::B16), v(16, A::B16)],
        )
    }

    fn sve2_expand(&mut self, arr: A, data: u8, dst: u8) -> Result {
        let z = |n| O::vec(R::z(n).unwrap(), arr);
        let zero = O::vec(R::Z16, A::D);
        let rank = z(16);
        self.a.put(C::Eor_z_zz_, &[zero, zero, zero])?;
        // HISTCNT counts active lanes up to the current lane, with zero in an
        // inactive lane. Subtracting one gives it an out-of-range TBL index.
        self.a.put(
            C::Histcnt_z_p_zz,
            &[rank, O::pred(R::P1, PredQual::Zero), rank, rank],
        )?;
        self.a.put(C::Sub_z_zi, &[rank, O::imm(1)])?;
        self.a.put(C::Tbl_z_zz, &[z(dst), z(data), rank])
    }

    fn sve_expand(&mut self, arr: A, data: u8, dst: u8) -> Result {
        let z = |n| O::vec(R::z(n).unwrap(), arr);
        let zero = O::vec(R::Z16, A::D);
        let (d, scratch) = (z(dst), z(16));
        self.a.put(
            C::Cpy_z_p_i,
            &[d, O::pred(R::P1, PredQual::Zero), O::imm(1)],
        )?;
        let lanes = self.config.bytes / sve_lane_bytes(arr);
        for exponent in 0..lanes.trailing_zeros() {
            let distance = 1 << exponent;
            self.predicate(R::P2, arr, distance)?;
            self.a.put(C::Eor_z_zz_, &[zero, zero, zero])?;
            self.a.put(
                C::Splice_z_p_zz,
                &[scratch, O::pred(R::P2, PredQual::None), scratch, d],
            )?;
            self.a.put(C::Add_z_zz_, &[d, d, scratch])?;
        }
        self.a.put(C::Sub_z_zi, &[d, O::imm(1)])?;
        self.a.put(C::Tbl_z_zz, &[d, z(data), d])?;
        self.a.put(C::Eor_z_zz_, &[zero, zero, zero])?;
        self.a.put(
            C::Sel_z_p_zz,
            &[d, O::pred(R::P1, PredQual::None), d, scratch],
        )
    }

    fn neon_expand(&mut self, arr: A, size: usize, data: u8, mask: u8, dst: u8) -> Result {
        let v = |n, a| O::vec(R::v(n).unwrap(), a);
        let (d, scratch) = (v(dst, arr), v(16, arr));
        let (d_bytes, scratch_bytes) = (v(dst, A::B16), v(16, A::B16));
        self.a.sign_fill(d, v(mask, arr))?;
        self.a.put(C::Neg_asimd, &[d, d])?;
        for exponent in 0..(16 / size).trailing_zeros() {
            let bytes = size << exponent;
            self.a.put(
                C::Eor_asimdsame_only,
                &[scratch_bytes, scratch_bytes, scratch_bytes],
            )?;
            self.a.put(
                C::Ext_asimd,
                &[
                    scratch_bytes,
                    scratch_bytes,
                    d_bytes,
                    O::imm((16 - bytes) as u64),
                ],
            )?;
            self.a.put(C::Add_asimdsame_only, &[d, d, scratch])?;
        }
        self.neon_splat_int(arr, 1, 16)?;
        self.a.put(C::Sub_asimdsame_only, &[d, d, scratch])?;
        self.neon_byte_indices(arr, size, dst, dst)?;
        self.a
            .put(C::Tbl_asimd, &[d_bytes, v(data, A::B16), d_bytes])?;
        self.a.sign_fill(scratch, v(mask, arr))?;
        self.a
            .put(C::And_asimdsame_only, &[d_bytes, d_bytes, scratch_bytes])
    }
}
