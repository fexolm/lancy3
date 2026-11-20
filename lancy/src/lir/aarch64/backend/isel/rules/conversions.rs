use super::{Rules, fp, scalar_int};
use crate::{
    hir::{Lane, Type, Value},
    lir::aarch64::backend::{
        A, C, arrangements,
        inst::{Inst, class},
        integer_float::Form,
        sve,
        width::{Conversion, Narrow, Widen},
    },
};
use splice_target::Profile;

impl Rules<'_> {
    pub(super) fn bitcast(&mut self, to: Type, v: Value) -> Inst {
        let from = self.ty(v);
        if class(to) != class(from) {
            Inst::BitMove(from, to, self.dst(), self.input(v))
        } else {
            Inst::Copy(self.dst(), self.input(v))
        }
    }

    pub(super) fn scalar_cast(&mut self, to: Type, v: Value) -> Option<Inst> {
        let from = self.ty(v);
        let dst = self.dst();
        if from == to {
            return Some(Inst::Copy(dst, self.input(v)));
        }
        if let Some(half) = self.half_cast(from.lane()?, to.lane()?, v) {
            return Some(half);
        }
        if let Some(inst) = self.integer_float_cast(from.lane()?, to.lane()?, v) {
            return Some(inst);
        }
        if from.lane().is_some_and(scalar_int) && to.lane().is_some_and(scalar_int) {
            return Some(Inst::IntCast(from, to, dst, self.input(v)));
        }
        if [from, to]
            .into_iter()
            .all(|t| t.lane().is_some_and(|lane| !lane.float()))
        {
            return Some(Inst::WideCast(from, to, dst, self.input(v)));
        }
        let code = float_cast(from, to)?;
        Some(Inst::FloatCast(code, from, to, dst, self.input(v)))
    }

    pub(super) fn vector_cast(&mut self, to: Lane, v: Value) -> Option<Inst> {
        let from = self.ty(v).lane().unwrap();
        if from == to || (!from.float() && !to.float()) {
            return Some(Inst::Copy(self.dst(), self.input(v)));
        }
        if let Some(half) = self.half_cast(from, to, v) {
            return Some(half);
        }
        if let Some(inst) = self.integer_float_cast(from, to, v) {
            return Some(inst);
        }
        let (code, arr) = packed_cast(from, to, self.config.profile)?;
        Some(Inst::PackedCast(code, arr, self.dst(), self.input(v)))
    }

    fn integer_float_cast(&mut self, from: Lane, to: Lane, v: Value) -> Option<Inst> {
        if from.float() == to.float() {
            return None;
        }
        let (int, float) = if from.float() { (to, from) } else { (from, to) };
        if int.bytes() != 16 && float.bytes() != 2 {
            return None;
        }
        let packed = matches!(self.ty(v), Type::Vector(_));
        let form = Form {
            from,
            to,
            packed,
            scalable: packed && sve(self.config.profile),
        };
        Some(Inst::IntegerFloatCast(form, self.dst(), self.input(v)))
    }

    pub(super) fn widen(&mut self, to: Lane, high: bool, v: Value) -> Inst {
        let from = self.ty(v).lane().unwrap();
        let profile = self.config.profile;
        let form = select_widen(from, to, high, profile).unwrap_or(Widen::Convert(
            Conversion {
                from,
                to,
                scalable: sve(profile),
            },
            high,
        ));
        Inst::Widen(form, self.dst(), self.input(v))
    }

    pub(super) fn narrow(&mut self, to: Lane, a: Value, b: Value) -> Inst {
        let from = self.ty(a).lane().unwrap();
        let profile = self.config.profile;
        let form = select_narrow(from, to, profile).unwrap_or(Narrow::Convert(Conversion {
            from,
            to,
            scalable: sve(profile),
        }));
        Inst::Narrow(form, self.dst(), self.input(a), self.input(b))
    }
}

fn float_cast(from: Type, to: Type) -> Option<C> {
    match (from, to) {
        (Type::F32, Type::F64) => Some(C::Fcvt_d_s),
        (Type::F64, Type::F32) => Some(C::Fcvt_s_d),
        (from, to) if fp(to) && from.lane().is_some_and(scalar_int) => {
            Some(if from.lane().unwrap().signed() {
                C::Scvtf_scalar
            } else {
                C::Ucvtf_scalar
            })
        }
        (from, to) if fp(from) && to.lane().is_some_and(scalar_int) => {
            Some(if to.lane().unwrap().signed() {
                C::Fcvtzs_scalar
            } else {
                C::Fcvtzu_scalar
            })
        }
        _ => None,
    }
}

fn packed_cast(from: Lane, to: Lane, profile: Profile) -> Option<(C, A)> {
    let size = from.bytes();
    if size != to.bytes() || !matches!(size, 4 | 8) || from.float() == to.float() {
        return None;
    }
    let (neon, scalable) = arrangements(size);
    let code = match (sve(profile), from.float(), from.signed() || to.signed()) {
        (true, true, true) => C::Fcvtzs_z_p_z,
        (true, true, false) => C::Fcvtzu_z_p_z,
        (true, false, true) => C::Scvtf_z_p_z,
        (true, false, false) => C::Ucvtf_z_p_z,
        (false, true, true) => C::Fcvtzs_asimd,
        (false, true, false) => C::Fcvtzu_asimd,
        (false, false, true) => C::Scvtf_asimd,
        (false, false, false) => C::Ucvtf_asimd,
    };
    Some((code, if sve(profile) { scalable } else { neon }))
}

fn converts_width(from: Lane, to: Lane) -> bool {
    matches!((from, to), (Lane::F32, Lane::F64) | (Lane::F64, Lane::F32))
        || (!from.float() && !to.float() && from.bytes().max(to.bytes()) <= 8)
}

fn half_neon(size: usize) -> A {
    [A::B8, A::H4, A::S2][size.trailing_zeros() as usize]
}

fn select_widen(from: Lane, to: Lane, high: bool, profile: Profile) -> Option<Widen> {
    if to.bytes() == 16 && !from.float() {
        return Some(Widen::Wide {
            signed: from.signed(),
            scalable: sve(profile),
            high,
        });
    }
    if !converts_width(from, to) {
        return None;
    }
    let (small, small_sve) = arrangements(from.bytes());
    let (large, large_sve) = arrangements(to.bytes());
    Some(if sve(profile) {
        let code = if from.float() {
            C::Fcvt_z_d_p_s
        } else if from.signed() {
            C::Sunpklo_z
        } else {
            C::Uunpklo_z
        };
        Widen::Sve {
            code,
            from: small_sve,
            to: large_sve,
            high,
        }
    } else {
        let code = match (from.float(), from.signed(), high) {
            (true, _, false) => C::Fcvtl_asimd,
            (true, _, true) => C::Fcvtl2_asimd,
            (false, true, false) => C::Sxtl_asimd,
            (false, true, true) => C::Sxtl2_asimd,
            (false, false, false) => C::Uxtl_asimd,
            (false, false, true) => C::Uxtl2_asimd,
        };
        let src = if high { small } else { half_neon(from.bytes()) };
        Widen::Neon(code, large, src)
    })
}

fn select_narrow(from: Lane, to: Lane, profile: Profile) -> Option<Narrow> {
    if from.bytes() == 16 && !to.float() {
        return Some(if sve(profile) {
            Narrow::Sve {
                to: A::D,
                float: false,
            }
        } else {
            Narrow::NeonWide
        });
    }
    if !converts_width(from, to) {
        return None;
    }
    let (small, small_sve) = arrangements(to.bytes());
    let float = from.float();
    Some(if sve(profile) {
        Narrow::Sve {
            to: small_sve,
            float,
        }
    } else {
        let codes = if float {
            [C::Fcvtn_asimd, C::Fcvtn2_asimd]
        } else {
            [C::Xtn_asimd, C::Xtn2_asimd]
        };
        Narrow::Neon {
            codes,
            from: arrangements(from.bytes()).0,
            to: [half_neon(to.bytes()), small],
        }
    })
}
