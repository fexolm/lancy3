use super::Rules;
use crate::{
    hir::{Binary, Type, Value},
    lir::aarch64::backend::{inst::Inst, sve, wide::MathForm},
};

impl Rules<'_> {
    pub(super) fn wide_binary(&mut self, op: Binary, a: Value, b: Value) -> Option<Inst> {
        use Binary::*;
        let ty = self.ty(a);
        let lane = ty.lane().filter(|lane| lane.bytes() == 16)?;
        if !matches!(
            op,
            Mul | Div
                | Rem
                | Eq
                | Ne
                | Lt
                | Le
                | Gt
                | Ge
                | Min
                | Max
                | Shl
                | Shr
                | RotateLeft
                | RotateRight
                | SaturatingAdd
                | SaturatingSub
        ) {
            return None;
        }
        let packed = matches!(ty, Type::Vector(_));
        let form = MathForm {
            op,
            signed: lane.signed(),
            packed,
            scalable: packed && sve(self.config.profile),
        };
        Some(Inst::WideMath(
            form,
            self.dst(),
            self.input(a),
            self.input(b),
        ))
    }
}
