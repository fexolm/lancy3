use super::{Rules, scalar_binary::float_compare};
use crate::{
    hir::{Binary, Lane, Type, Unary, Value},
    lir::aarch64::backend::{
        C,
        half::{Action, Form},
        inst::Inst,
        sve,
    },
};

fn half(lane: Lane) -> bool {
    matches!(lane, Lane::F16 | Lane::BF16)
}

impl Rules<'_> {
    fn half_form(&self, v: Value, action: Action) -> Option<Form> {
        let ty = self.ty(v);
        let lane = ty.lane().filter(|&lane| half(lane))?;
        let packed = matches!(ty, Type::Vector(_));
        Some(Form {
            lane,
            packed,
            scalable: packed && sve(self.config.profile),
            action,
        })
    }

    pub(super) fn half_binary(&mut self, op: Binary, a: Value, b: Value) -> Option<Inst> {
        use Binary::*;
        let action = match op {
            Add => Action::Binary(C::Fadd_scalar),
            Sub => Action::Binary(C::Fsub_scalar),
            Mul => Action::Binary(C::Fmul_scalar),
            Div => Action::Binary(C::Fdiv_scalar),
            Min => Action::Binary(C::Fminnm_scalar),
            Max => Action::Binary(C::Fmaxnm_scalar),
            _ => Action::Compare(float_compare(op, Type::F64)?.0),
        };
        let form = self.half_form(a, action)?;
        Some(Inst::Half(
            form,
            self.dst(),
            self.input(a),
            [Some(self.input(b)), None],
        ))
    }

    pub(super) fn half_unary(&mut self, op: Unary, v: Value) -> Option<Inst> {
        use Unary::*;
        if !matches!(
            op,
            Sqrt | Floor | Ceil | Trunc | Round | Nearest | Fract | Recip
        ) {
            return None;
        }
        let form = self.half_form(v, Action::Unary(op))?;
        Some(Inst::Half(form, self.dst(), self.input(v), [None; 2]))
    }

    pub(super) fn half_cast(&mut self, from: Lane, to: Lane, v: Value) -> Option<Inst> {
        if !(from.float() && to.float() && (half(from) || half(to))) {
            return None;
        }
        let dst = self.dst();
        if from == to {
            return Some(Inst::Copy(dst, self.input(v)));
        }
        if matches!(self.ty(v), Type::Vector(_)) {
            let form = self.half_form(v, Action::Cast(to))?;
            Some(Inst::Half(form, dst, self.input(v), [None; 2]))
        } else {
            Some(Inst::HalfCast(from, to, dst, self.input(v)))
        }
    }

    pub(super) fn half_fma(&mut self, a: Value, b: Value, c: Value) -> Option<Inst> {
        let form = self.half_form(a, Action::MulAdd)?;
        Some(Inst::Half(
            form,
            self.dst(),
            self.input(a),
            [Some(self.input(b)), Some(self.input(c))],
        ))
    }
}
