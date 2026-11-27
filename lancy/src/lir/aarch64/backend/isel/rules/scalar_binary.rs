use super::Rules;
use crate::{
    hir::{Binary, Type, Value},
    lir::aarch64::backend::{
        A, C, Cond,
        inst::{Inst, IntBinary, IntOp, Packed},
    },
};

impl Rules<'_> {
    pub(super) fn scalar_binary(&mut self, op: Binary, a: Value, b: Value) -> Option<Inst> {
        let ty = self.ty(a);
        let dst = self.dst();
        if let Some(special) = self.float_binary_special(op, a, b) {
            return Some(special);
        }
        if let Some(half) = self.half_binary(op, a, b) {
            return Some(half);
        }
        if let Some(wide) = self.wide_binary(op, a, b) {
            return Some(wide);
        }
        if let Some(rotate) = self.rotate(op, a, b) {
            return Some(rotate);
        }
        if matches!(op, Binary::SaturatingAdd | Binary::SaturatingSub) {
            let lane = ty.lane().unwrap();
            let code = match (op, lane.signed()) {
                (Binary::SaturatingAdd, true) => C::Sqadd_asimd,
                (Binary::SaturatingAdd, false) => C::Uqadd_asimd,
                (_, true) => C::Sqsub_asimd,
                (_, false) => C::Uqsub_asimd,
            };
            return Some(Inst::ScalarSaturate(
                code,
                lane.bytes(),
                dst,
                self.input(a),
                self.input(b),
            ));
        }
        if ty == Type::I32
            && let Some((code, src, imm, shift)) = self.add_sub_immediate(op, a, b)
        {
            return Some(Inst::Imm(code, imm, shift, dst, self.input(src)));
        }
        if matches!(op, Binary::Add | Binary::Sub)
            && ty.lane().is_some_and(|lane| lane.bytes() == 16)
        {
            return Some(Inst::WideAdd {
                subtract: op == Binary::Sub,
                scalable: false,
                dst,
                a: self.input(a),
                b: self.input(b),
            });
        }
        if let Some(form) = integer_binary(op, ty) {
            return Some(Inst::Int(form, dst, self.input(a), self.input(b)));
        }
        if let Some((code, arr)) = float_binary(op, ty) {
            return Some(Inst::Float(code, arr, dst, self.input(a), self.input(b)));
        }
        if let Some((cond, arr)) = float_compare(op, ty) {
            return Some(Inst::FloatCompare(
                cond,
                arr,
                dst,
                self.input(a),
                self.input(b),
            ));
        }
        if matches!(ty, Type::F32 | Type::F64 | Type::I128 | Type::U128) {
            let code = neon_bits(op)?;
            return Some(Inst::Packed(
                Packed::Neon(code, A::B16),
                dst,
                self.input(a),
                self.input(b),
            ));
        }
        None
    }

    fn add_sub_immediate(&self, op: Binary, a: Value, b: Value) -> Option<(C, Value, u16, u8)> {
        let code = match op {
            Binary::Add => C::Add_32_addsub_imm,
            Binary::Sub => C::Sub_32_addsub_imm,
            _ => return None,
        };
        let form = |v| self.ctx.constant(v).and_then(add_immediate);
        if let Some((imm, shift)) = form(b) {
            return Some((code, a, imm, shift));
        }
        if op == Binary::Add {
            let (imm, shift) = form(a)?;
            return Some((code, b, imm, shift));
        }
        None
    }
}

fn add_immediate(n: i32) -> Option<(u16, u8)> {
    if (0..4096).contains(&n) {
        Some((n as u16, 0))
    } else if (0..1 << 24).contains(&n) && n & 4095 == 0 {
        Some(((n >> 12) as u16, 12))
    } else {
        None
    }
}

fn integer_binary(op: Binary, ty: Type) -> Option<IntBinary> {
    use Binary::*;
    use IntOp::*;
    let lane = ty.lane()?;
    let size = lane.bytes();
    if size > 8 || (lane.float() && (size > 2 || !matches!(op, And | Or | Xor))) {
        return None;
    }
    let choose = |word, double| if size == 8 { double } else { word };
    let signed = lane.signed();
    let native = match op {
        Add => Three(choose(C::Add_32_addsub_shift, C::Add_64_addsub_shift)),
        Sub => Three(choose(C::Sub_32_addsub_shift, C::Sub_64_addsub_shift)),
        And => Three(choose(C::And_32_log_shift, C::And_64_log_shift)),
        Or => Three(choose(C::Orr_32_log_shift, C::Orr_64_log_shift)),
        Xor => Three(choose(C::Eor_32_log_shift, C::Eor_64_log_shift)),
        Mul => Multiply(choose(C::Madd_32a_dp_3src, C::Madd_64a_dp_3src)),
        Div | Rem => Divide {
            signed,
            remainder: op == Rem,
        },
        Shl => Three(choose(C::Lslv_32_dp_2src, C::Lslv_64_dp_2src)),
        Shr if signed => Three(choose(C::Asrv_32_dp_2src, C::Asrv_64_dp_2src)),
        Shr => Three(choose(C::Lsrv_32_dp_2src, C::Lsrv_64_dp_2src)),
        Eq => Compare(Cond::Eq),
        Ne => Compare(Cond::Ne),
        Lt => Compare(if signed { Cond::Lt } else { Cond::Cc }),
        Le => Compare(if signed { Cond::Le } else { Cond::Ls }),
        Gt => Compare(if signed { Cond::Gt } else { Cond::Hi }),
        Ge => Compare(if signed { Cond::Ge } else { Cond::Cs }),
        Min => MinMax(if signed { Cond::Le } else { Cond::Ls }),
        Max => MinMax(if signed { Cond::Ge } else { Cond::Cs }),
        _ => return None,
    };
    Some(IntBinary {
        op: native,
        size,
        signed_inputs: signed && matches!(op, Lt | Le | Gt | Ge | Min | Max | Div | Rem | Shr),
        wrap_shift: size < 4 && matches!(op, Shl | Shr),
    })
}

fn float_binary(op: Binary, ty: Type) -> Option<(C, A)> {
    let arr = match ty {
        Type::F32 => A::S,
        Type::F64 => A::D,
        _ => return None,
    };
    let code = match op {
        Binary::Add => C::Fadd_scalar,
        Binary::Sub => C::Fsub_scalar,
        Binary::Mul => C::Fmul_scalar,
        Binary::Div => C::Fdiv_scalar,
        _ => return None,
    };
    Some((code, arr))
}

pub(super) fn float_compare(op: Binary, ty: Type) -> Option<(Cond, A)> {
    let arr = match ty {
        Type::F32 => A::S,
        Type::F64 => A::D,
        _ => return None,
    };
    let cond = match op {
        Binary::Eq => Cond::Eq,
        Binary::Ne => Cond::Ne,
        Binary::Lt => Cond::Mi,
        Binary::Le => Cond::Ls,
        Binary::Gt => Cond::Gt,
        Binary::Ge => Cond::Ge,
        Binary::Ordered => Cond::Vc,
        Binary::Unordered => Cond::Vs,
        _ => return None,
    };
    Some((cond, arr))
}

fn neon_bits(op: Binary) -> Option<C> {
    match op {
        Binary::And => Some(C::And_asimdsame_only),
        Binary::Or => Some(C::Orr_asimdsame_only),
        Binary::Xor => Some(C::Eor_asimdsame_only),
        _ => None,
    }
}
