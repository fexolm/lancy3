use super::Rules;
use crate::{
    hir::{Binary, Lane, Value},
    lir::aarch64::backend::{A, C, Cond, arrangements, inst::Inst, reduce::Form, sve},
};
use splice_target::Profile;

impl Rules<'_> {
    pub(super) fn reduce(&mut self, op: Binary, v: Value) -> Inst {
        let lane = self.ty(v).lane().unwrap();
        let profile = self.config.profile;
        let form =
            select_reduce(op, lane, profile).unwrap_or(Form::Ordered(op, lane, sve(profile)));
        Inst::Reduce(form, self.dst(), self.input(v))
    }

    pub(super) fn mask_test(&mut self, v: Value, all: bool) -> Inst {
        let size = self.ty(v).lane().unwrap().bytes();
        if size > 8 {
            return Inst::WideMaskTest(all, sve(self.config.profile), self.dst(), self.input(v));
        }
        let (neon, scalable) = arrangements(size);
        let arr = if sve(self.config.profile) {
            scalable
        } else {
            neon
        };
        Inst::MaskTest(all, arr, self.dst(), self.input(v))
    }
}

fn select_reduce(op: Binary, lane: Lane, profile: Profile) -> Option<Form> {
    use Binary::*;
    let size = lane.bytes();
    let scalable = sve(profile);
    if lane.float() {
        let arr = match (op, lane) {
            (Add, Lane::F32) => A::S,
            (Add, Lane::F64) => A::D,
            _ => return None,
        };
        return Some(Form::FloatAdd(arr, scalable));
    }
    if size > 8 {
        return None;
    }
    let (neon, sve) = arrangements(size);
    let signed = lane.signed();
    let (code, arr) = if scalable {
        let code = match op {
            Add => C::Uaddv_d_p_z,
            And => C::Andv_v_p_z,
            Or => C::Orv_v_p_z,
            Xor => C::Eorv_v_p_z,
            Min if signed => C::Sminv_v_p_z,
            Min => C::Uminv_v_p_z,
            Max if signed => C::Smaxv_v_p_z,
            Max => C::Umaxv_v_p_z,
            _ => return None,
        };
        (code, sve)
    } else {
        let code = match op {
            Add if size == 8 => C::Addp_d_asimd,
            Add => C::Addv_asimd,
            And => return Some(Form::NeonBits(C::And_64_log_shift, size)),
            Or => return Some(Form::NeonBits(C::Orr_64_log_shift, size)),
            Xor => return Some(Form::NeonBits(C::Eor_64_log_shift, size)),
            Min | Max if size == 8 => {
                let cond = match (op, signed) {
                    (Min, true) => Cond::Le,
                    (Min, false) => Cond::Ls,
                    (Max, true) => Cond::Ge,
                    _ => Cond::Cs,
                };
                return Some(Form::NeonMinMax64(cond));
            }
            Min if signed => C::Sminv_asimd,
            Min => C::Uminv_asimd,
            Max if signed => C::Smaxv_asimd,
            Max => C::Umaxv_asimd,
            _ => return None,
        };
        (code, neon)
    };
    Some(Form::Integer(code, arr, size))
}
