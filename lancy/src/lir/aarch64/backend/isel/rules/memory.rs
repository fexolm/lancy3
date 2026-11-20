use super::{Rules, reg};
use crate::{
    hir::{Type, Value, VectorAddress},
    lir::aarch64::backend::{
        C, arrangements,
        inst::{Inst, Masked},
        sve,
    },
};
use splice_target::Profile;

impl Rules<'_> {
    pub(super) fn masked_memory(
        &mut self,
        address: VectorAddress,
        mask: Value,
        value: Value,
    ) -> Inst {
        let ty = self.ty(value);
        let dst = self.result.map(|v| reg(self.f, v));
        Inst::Masked {
            kind: masked(
                ty,
                self.config.profile,
                address.indices.is_some(),
                dst.is_some(),
            ),
            ty,
            base: self.input(address.base),
            indices: address.indices.map(|v| self.input(v)),
            mask: self.input(mask),
            value: self.input(value),
            dst,
        }
    }
}

fn masked(ty: Type, profile: Profile, indexed: bool, load: bool) -> Masked {
    let size = ty.lane().unwrap().bytes();
    if !sve(profile) {
        return Masked::Neon;
    }
    if size == 16 {
        let code = match (load, indexed) {
            (true, true) => C::Ld1d_z_p_bz_lsl,
            (true, false) => C::Ld1d_z_p_bi_u64,
            (false, true) => C::St1d_z_p_bz_lsl,
            (false, false) => C::St1d_z_p_bi_,
        };
        return Masked::SveWide(code);
    }
    let code = match (load, indexed, size) {
        (true, true, 1) => return Masked::SveNarrow(C::Ld1b_z_p_bz_sxtw),
        (true, true, 2) => return Masked::SveNarrow(C::Ld1h_z_p_bz_sxtw),
        (true, true, 4) => C::Ld1w_z_p_bz_sxtw,
        (true, true, 8) => C::Ld1d_z_p_bz_lsl,
        (false, true, 1) => return Masked::SveNarrow(C::St1b_z_p_bz_sxtw),
        (false, true, 2) => return Masked::SveNarrow(C::St1h_z_p_bz_sxtw),
        (false, true, 4) => C::St1w_z_p_bz_sxtw,
        (false, true, 8) => C::St1d_z_p_bz_lsl,
        (true, false, 1) => C::Ld1b_z_p_bi_u8,
        (true, false, 2) => C::Ld1h_z_p_bi_u16,
        (true, false, 4) => C::Ld1w_z_p_bi_u32,
        (true, false, 8) => C::Ld1d_z_p_bi_u64,
        (false, false, 1) => C::St1b_z_p_bi_,
        (false, false, 2) => C::St1h_z_p_bi_,
        (false, false, 4) => C::St1w_z_p_bi_,
        (false, false, 8) => C::St1d_z_p_bi_,
        _ => unreachable!("verified lane width"),
    };
    Masked::Sve {
        code,
        arrangement: arrangements(size).1,
    }
}
