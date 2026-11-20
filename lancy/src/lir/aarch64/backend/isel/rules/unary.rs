use super::Rules;
use crate::{
    hir::{Type, Unary, Value},
    lir::aarch64::backend::{
        A, C, arrangements,
        inst::Inst,
        sve,
        unary::{FloatOp, FloatRegs, Form},
    },
};
use splice_target::Profile;

impl Rules<'_> {
    pub(super) fn unary(&mut self, op: Unary, v: Value) -> Option<Inst> {
        let ty = self.ty(v);
        if let Some(half) = self.half_unary(op, v) {
            return Some(half);
        }
        if let Some(special) = self.float_unary_special(op, v) {
            return Some(special);
        }
        if op == Unary::Abs
            && ty
                .lane()
                .is_some_and(|lane| !lane.signed() && !lane.float())
        {
            return Some(Inst::Copy(self.dst(), self.input(v)));
        }
        if ty.lane().is_some_and(|lane| lane.bytes() == 16)
            && matches!(
                op,
                Unary::Neg
                    | Unary::Abs
                    | Unary::Popcnt
                    | Unary::LeadingZeros
                    | Unary::TrailingZeros
            )
        {
            let scalable = matches!(ty, Type::Vector(_)) && sve(self.config.profile);
            return Some(Inst::WideUnary(op, scalable, self.dst(), self.input(v)));
        }
        let form = select_unary(op, ty, self.config.profile)?;
        Some(Inst::Unary(form, self.dst(), self.input(v)))
    }
}

fn select_unary(op: Unary, ty: Type, profile: Profile) -> Option<Form> {
    use Unary::*;
    let lane = ty.lane()?;
    let size = lane.bytes();
    let packed = matches!(ty, Type::Vector(_));
    let scalable = packed && sve(profile);
    if op == Not {
        return Some(if scalable {
            Form::Sve(C::Not_z_p_z, A::D)
        } else if packed || size >= 16 || (lane.float() && size >= 4) {
            Form::Neon(C::Mvn_asimdmisc, A::B16)
        } else {
            Form::Integer(op, size)
        });
    }
    if lane.float() {
        if size == 2 {
            return matches!(op, Neg | Abs).then_some(Form::HalfSign(op, packed, scalable));
        }
        let (neon, sve) = arrangements(size);
        let (regs, encoding) = if scalable {
            (FloatRegs::Sve(sve), 2)
        } else if packed {
            (FloatRegs::Neon(neon), 1)
        } else {
            (FloatRegs::Scalar(sve), 0)
        };
        let codes = match op {
            Neg => [C::Fneg_scalar, C::Fneg_asimd, C::Fneg_z_p_z],
            Abs => [C::Fabs_scalar, C::Fabs_asimd, C::Fabs_z_p_z],
            Sqrt => [C::Fsqrt_scalar, C::Fsqrt_asimd, C::Fsqrt_z_p_z],
            Floor => [C::Frintm_scalar, C::Frintm_asimd, C::Frintm_z_p_z],
            Ceil => [C::Frintp_scalar, C::Frintp_asimd, C::Frintp_z_p_z],
            Trunc | Fract => [C::Frintz_scalar, C::Frintz_asimd, C::Frintz_z_p_z],
            Round => [C::Frinta_scalar, C::Frinta_asimd, C::Frinta_z_p_z],
            Nearest => [C::Frintn_scalar, C::Frintn_asimd, C::Frintn_z_p_z],
            Recip => return Some(Form::Float(FloatOp::Recip, regs)),
            _ => return None,
        };
        let code = codes[encoding];
        let form = if op == Fract {
            FloatOp::Fract(code)
        } else {
            FloatOp::Native(code)
        };
        return Some(Form::Float(form, regs));
    }
    if size > 8 {
        return None;
    }
    if !packed {
        return Some(Form::Integer(op, size));
    }
    let (neon, sve) = arrangements(size);
    Some(if scalable {
        let code = match op {
            Neg => C::Neg_z_p_z,
            Abs => C::Abs_z_p_z,
            Popcnt => C::Cnt_z_p_z,
            LeadingZeros => C::Clz_z_p_z,
            TrailingZeros => return Some(Form::SveCtz(sve)),
            _ => unreachable!("verified integer unary"),
        };
        Form::Sve(code, sve)
    } else {
        let code = match op {
            Neg => C::Neg_asimd,
            Abs => C::Abs_asimd,
            Popcnt => return Some(Form::NeonPopcnt(size)),
            LeadingZeros | TrailingZeros if size == 8 => {
                return Some(Form::NeonCountZeros64(op));
            }
            LeadingZeros => C::Clz_asimd,
            TrailingZeros => return Some(Form::NeonCtz(neon)),
            _ => unreachable!("verified integer unary"),
        };
        Form::Neon(code, neon)
    })
}
