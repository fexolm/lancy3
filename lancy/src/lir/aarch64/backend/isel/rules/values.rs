use super::Rules;
use crate::{
    hir::{Lane, Type, Value},
    lir::aarch64::backend::{C, arrangements, inst::Inst, sve},
    lir::constants::iota_bytes,
};

impl Rules<'_> {
    pub(super) fn iota(&self, lane: Lane) -> Inst {
        if sve(self.config.profile) {
            return Inst::SveIota(lane, self.dst());
        }
        let bytes = iota_bytes(lane, self.config.bytes);
        let bits = u128::from_le_bytes(*bytes.first_chunk().unwrap());
        Inst::Literal(Type::Vector(lane), bits, self.dst())
    }

    pub(super) fn splat(&mut self, v: Value) -> Inst {
        let ty = self.ty(v);
        let size = ty.lane().unwrap().bytes();
        let scalable = sve(self.config.profile);
        let dst = self.dst();
        let src = self.input(v);
        if size > 8 {
            return if scalable {
                Inst::SveSplat128(dst, src)
            } else {
                Inst::Copy(dst, src)
            };
        }
        let (neon, sve) = arrangements(size);
        let (code, arr) = if scalable {
            (C::Dup_z_r, sve)
        } else {
            (C::Dup_asimdgen, neon)
        };
        Inst::Splat(code, arr, ty, dst, src)
    }

    pub(super) fn lane_access(&mut self, src: Value, index: Value, insert: Option<Value>) -> Inst {
        let lane = self.ty(src).lane().unwrap().scalar();
        let insert = insert.map(|v| self.input(v));
        Inst::LaneAccess {
            lane,
            src: self.input(src),
            index: self.input(index),
            insert,
            dst: self.dst(),
        }
    }
}
