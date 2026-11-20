use super::{Rules, fp};
use crate::{
    hir::{Lane, Ternary, Type, Value},
    lir::aarch64::backend::{
        A, C, arrangements,
        inst::{Inst, Select, class},
        sve,
    },
};
use regalloc2::RegClass;
use splice_target::Profile;

impl Rules<'_> {
    pub(super) fn scalar_fma(&mut self, a: Value, b: Value, c: Value) -> Option<Inst> {
        let ty = self.ty(a);
        if !fp(ty) {
            return self.half_fma(a, b, c);
        }
        let arr = if ty == Type::F32 { A::S } else { A::D };
        Some(Inst::Fma(
            C::Fmadd_scalar,
            arr,
            self.dst(),
            [self.input(a), self.input(b), self.input(c)],
        ))
    }

    pub(super) fn vector_fma(&mut self, a: Value, b: Value, c: Value) -> Option<Inst> {
        let Type::Vector(lane @ (Lane::F32 | Lane::F64)) = self.ty(a) else {
            return self.half_fma(a, b, c);
        };
        let (neon, scalable) = arrangements(lane.bytes());
        let (code, arr) = if sve(self.config.profile) {
            (C::Fmla_z_p_zz, scalable)
        } else {
            (C::Fmla_asimd, neon)
        };
        Some(Inst::Fma(
            code,
            arr,
            self.dst(),
            [self.input(a), self.input(b), self.input(c)],
        ))
    }

    pub(super) fn scalar_select(&mut self, mask: Value, a: Value, b: Value) -> Inst {
        Inst::Select(
            Select::Scalar(self.ty(a)),
            self.dst(),
            self.input(mask),
            self.input(a),
            self.input(b),
        )
    }

    pub(super) fn scalar_bit_select(&mut self, mask: Value, a: Value, b: Value) -> Inst {
        let form = if class(self.ty(a)) == RegClass::Int {
            Select::ScalarBits
        } else {
            Select::Neon(None)
        };
        Inst::Select(
            form,
            self.dst(),
            self.input(mask),
            self.input(a),
            self.input(b),
        )
    }

    pub(super) fn vector_select(&mut self, kind: Ternary, mask: Value, a: Value, b: Value) -> Inst {
        let profile = self.config.profile;
        let dst = self.dst();
        if kind == Ternary::BitSelect && profile == Profile::Sve2 {
            return Inst::Sve2Ternary(
                C::Bsl_z_zzz_,
                dst,
                [self.input(a), self.input(b), self.input(mask)],
            );
        }
        let scalable = sve(profile);
        let size = self.ty(a).lane().unwrap().bytes();
        let form = if kind == Ternary::Select && size == 16 {
            Select::Wide(scalable)
        } else {
            let arr = (kind == Ternary::Select).then(|| {
                let (neon, sve) = arrangements(size);
                if scalable { sve } else { neon }
            });
            if scalable {
                Select::Sve(arr)
            } else {
                Select::Neon(arr)
            }
        };
        Inst::Select(form, dst, self.input(mask), self.input(a), self.input(b))
    }
}
