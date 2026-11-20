use super::{
    A, C, Cond, Emitter, O, PredQual, R, Result, arrangements, operations::sve_lane_bytes,
};

pub(super) fn z(n: u8, arr: A) -> O {
    O::vec(R::z(n).unwrap(), arr)
}

impl Emitter<'_> {
    pub(super) fn sve_each_lane(
        &mut self,
        arr: A,
        emit_body: impl FnOnce(&mut Self) -> Result,
    ) -> Result {
        self.sve_lanes_after(arr, 0, emit_body)
    }

    pub(super) fn sve_lanes_after(
        &mut self,
        arr: A,
        first: usize,
        emit_body: impl FnOnce(&mut Self) -> Result,
    ) -> Result {
        self.sve_lane_range(
            arr,
            first,
            self.config.bytes / sve_lane_bytes(arr),
            emit_body,
        )
    }

    pub(super) fn sve_lane_range(
        &mut self,
        arr: A,
        first: usize,
        end: usize,
        emit_body: impl FnOnce(&mut Self) -> Result,
    ) -> Result {
        self.predicate(R::P4, arr, end)?;
        if first == 0 {
            self.a.put(C::Pfalse_p, &[O::vec(R::P1, A::B)])?;
        } else {
            self.predicate(R::P1, arr, first)?;
        }
        let next = self.a.label();
        let done = self.a.label();
        self.a.bind(next);
        self.sve_next_lane(arr)?;
        self.a.skip_if(Cond::Ne)?;
        self.a.jump(done);
        emit_body(self)?;
        self.a.jump(next);
        self.a.bind(done);
        Ok(())
    }

    pub(super) fn sve_next_lane(&mut self, arr: A) -> Result {
        self.sve_next_predicated_lane(arr, R::P1, R::P4)
    }

    pub(super) fn sve_next_predicated_lane(&mut self, arr: A, cursor: R, active: R) -> Result {
        let cursor = O::vec(cursor, arr);
        self.a.put(
            C::Pnext_p_p_p,
            &[cursor, O::pred(active, PredQual::None), cursor],
        )
    }

    pub(super) fn sve_lane_to_int(&mut self, arr: A, src: u8, dst: u8) -> Result {
        self.sve_predicated_lane_to_int(arr, src, dst, R::P1)
    }

    pub(super) fn sve_predicated_lane_to_int(
        &mut self,
        arr: A,
        src: u8,
        dst: u8,
        pred: R,
    ) -> Result {
        let dst = if arr == A::D { R::x(dst) } else { R::w(dst) }.unwrap();
        self.a.put(
            C::Lastb_r_p_z,
            &[O::reg(dst), O::pred(pred, PredQual::None), z(src, arr)],
        )
    }

    pub(super) fn sve_lane_from_int(&mut self, arr: A, src: u8, dst: u8, pred: R) -> Result {
        let src = if arr == A::D { R::x(src) } else { R::w(src) }.unwrap();
        self.a.put(
            C::Cpy_z_p_r,
            &[z(dst, arr), O::pred(pred, PredQual::Merge), O::reg(src)],
        )
    }

    /// Widen the group of 32-bit destinations starting at a logical byte offset.
    pub(super) fn sve_widen_group(
        &mut self,
        src: u8,
        dst: u8,
        size: usize,
        offset: usize,
        signed: bool,
    ) -> Result {
        let src = if offset != 0 {
            self.copy_vec(dst, src, true)?;
            let bytes = z(dst, A::B);
            self.a
                .put(C::Ext_z_zzi, &[bytes, bytes, bytes, O::imm(offset as u64)])?;
            dst
        } else {
            src
        };
        let code = if signed { C::Sunpklo_z } else { C::Uunpklo_z };
        let half = if size == 1 {
            self.a.put(code, &[z(dst, A::H), z(src, A::B)])?;
            dst
        } else {
            src
        };
        self.a.put(code, &[z(dst, A::S), z(half, A::H)])
    }

    pub(super) fn sve_narrow_group(&mut self, n: u8, size: usize) -> Result {
        let h = z(n, A::H);
        self.a.put(C::Uzp1_z_zz, &[h, h, h])?;
        if size == 1 {
            let b = z(n, A::B);
            self.a.put(C::Uzp1_z_zz, &[b, b, b])?;
        }
        Ok(())
    }

    pub(super) fn sve_prefix(&mut self, p: R, arr: A, lanes: R) -> Result {
        self.a.put(
            C::Whilelo_p_x_x,
            &[O::vec(p, arr), O::reg(R::XZR), O::reg(lanes)],
        )
    }

    pub(super) fn sve_splice(&mut self, arr: A, dst: u8, p: R, rhs: u8) -> Result {
        let d = z(dst, arr);
        self.a.put(
            C::Splice_z_p_zz,
            &[d, O::pred(p, PredQual::None), d, z(rhs, arr)],
        )
    }

    pub(super) fn neon_lane_to_int(
        &mut self,
        src: u8,
        dst: u8,
        size: usize,
        lane: usize,
    ) -> Result {
        let gpr = if size == 8 { R::x(dst) } else { R::w(dst) }.unwrap();
        let v = O::vec(R::v(src).unwrap(), arrangements(size).0);
        self.a
            .put(C::Umov_lane, &[O::reg(gpr), v, O::imm(lane as u64)])
    }

    pub(super) fn neon_lane_from_int(
        &mut self,
        src: u8,
        dst: u8,
        size: usize,
        lane: usize,
    ) -> Result {
        let gpr = if size == 8 { R::x(src) } else { R::w(src) }.unwrap();
        let v = O::vec(R::v(dst).unwrap(), arrangements(size).0);
        self.a
            .put(C::Ins_lane, &[v, O::reg(gpr), O::imm(lane as u64)])
    }
}
