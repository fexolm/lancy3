use super::{Rules, scalar_int};
use crate::{
    hir::{Binary, Lane, Op, Simd, Type, Value},
    lir::aarch64::backend::{
        A, C, arrangements,
        inst::{Inst, Packed, VectorDivide},
        sve,
    },
};
use splice_target::Profile;

impl Rules<'_> {
    pub(super) fn vector_binary(&mut self, op: Binary, a: Value, b: Value) -> Option<Inst> {
        let ty = self.ty(a);
        let lane = ty.lane().unwrap();
        let profile = self.config.profile;
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
        if !sve(profile) && op == Binary::Mul && scalar_int(lane) && lane.bytes() == 8 {
            return Some(Inst::NeonMul64(dst, self.input(a), self.input(b)));
        }

        if op == Binary::Xor
            && profile == Profile::Sve2
            && let Some(args) = self.xor3_args(a, b)
        {
            return Some(Inst::Sve2Ternary(
                C::Eor3_z_zzz_,
                dst,
                args.map(|v| self.input(v)),
            ));
        }
        if matches!(op, Binary::Shl | Binary::Shr)
            && scalar_int(lane)
            && let Some(count) = self.shift_count(b, lane)
        {
            let src = self.input(a);
            return Some(if count == 0 {
                Inst::Copy(dst, src)
            } else {
                let (code, arr) = packed_shift_immediate(op, lane, profile);
                Inst::PackedShiftImm(code, arr, count, dst, src)
            });
        }
        if matches!(op, Binary::Div | Binary::Rem) && scalar_int(lane) {
            let form = VectorDivide {
                size: lane.bytes(),
                signed: lane.signed(),
                remainder: op == Binary::Rem,
                scalable: sve(profile),
            };
            return Some(Inst::VectorDivide(form, dst, self.input(a), self.input(b)));
        }
        if matches!(op, Binary::Add | Binary::Sub) && lane.bytes() == 16 {
            return Some(Inst::WideAdd {
                subtract: op == Binary::Sub,
                scalable: sve(profile),
                dst,
                a: self.input(a),
                b: self.input(b),
            });
        }
        if let Some(form) = packed_binary(op, ty, profile) {
            return Some(Inst::Packed(form, dst, self.input(a), self.input(b)));
        }
        if matches!(op, Binary::Ordered | Binary::Unordered) {
            let (n, z) = arrangements(lane.bytes());
            let form =
                Packed::FloatOrder(if sve(profile) { z } else { n }, op == Binary::Unordered);
            return Some(Inst::Packed(form, dst, self.input(a), self.input(b)));
        }
        let (form, swap) = packed_compare(op, ty, profile)?;
        let (a, b) = if swap { (b, a) } else { (a, b) };
        Some(Inst::Packed(form, dst, self.input(a), self.input(b)))
    }

    fn xor3_args(&self, a: Value, b: Value) -> Option<[Value; 3]> {
        [(a, b), (b, a)].into_iter().find_map(|(inner, other)| {
            match self.ctx.single_use(inner)? {
                Op::Bin(Binary::Xor, x, y) => Some([x, y, other]),
                _ => None,
            }
        })
    }

    fn shift_count(&self, v: Value, lane: Lane) -> Option<u8> {
        let Op::Simd(Simd::Splat(amount)) = self.ctx.defining_op(v)? else {
            return None;
        };
        let amount = match self.ctx.defining_op(amount)? {
            Op::Const(n) => n as u8,
            Op::Literal(lane, bits) if !lane.float() => bits as u8,
            _ => return None,
        };
        Some(amount & (lane.bytes() as u8 * 8 - 1))
    }
}

fn packed_binary(op: Binary, ty: Type, profile: Profile) -> Option<Packed> {
    use Binary::*;
    let Type::Vector(lane) = ty else { return None };
    let size = lane.bytes();
    let bitwise = matches!(op, And | Or | Xor);
    if !bitwise {
        if matches!(lane, Lane::F32 | Lane::F64) {
            let (neon, scalable) = arrangements(size);
            if sve(profile) {
                let code = match op {
                    Add => C::Fadd_z_zz_,
                    Sub => C::Fsub_z_zz_,
                    Mul => C::Fmul_z_zz_,
                    Div => return Some(Packed::SveDestructive(C::Fdiv_z_p_zz_, scalable, false)),
                    _ => return None,
                };
                return Some(Packed::Sve(code, scalable));
            }
            let code = match op {
                Add => C::Fadd_asimdsame,
                Sub => C::Fsub_asimdsame,
                Mul => C::Fmul_asimdsame,
                Div => C::Fdiv_asimdsame,
                _ => return None,
            };
            return Some(Packed::Neon(code, neon));
        }
        if lane.float() || size > 8 {
            return None;
        }
    }
    let signed = lane.signed();
    if sve(profile) {
        let a = if bitwise { A::D } else { arrangements(size).1 };
        let code = match op {
            Add => C::Add_z_zz_,
            Sub => C::Sub_z_zz_,
            Mul if profile == Profile::Sve2 => C::Mul_z_zz_,
            Mul => return Some(Packed::SveDestructive(C::Mul_z_p_zz_, a, true)),
            Min | Max => {
                let code = match (op, signed) {
                    (Min, true) => C::Smin_z_p_zz,
                    (Min, false) => C::Umin_z_p_zz,
                    (Max, true) => C::Smax_z_p_zz,
                    _ => C::Umax_z_p_zz,
                };
                return Some(Packed::SveDestructive(code, a, true));
            }
            SaturatingAdd if signed => C::Sqadd_z_zz,
            SaturatingAdd => C::Uqadd_z_zz,
            SaturatingSub if signed => C::Sqsub_z_zz,
            SaturatingSub => C::Uqsub_z_zz,
            Shl | Shr => {
                let code = if op == Shl {
                    C::Lsl_z_p_zz
                } else if signed {
                    C::Asr_z_p_zz
                } else {
                    C::Lsr_z_p_zz
                };
                return Some(Packed::SveShift(code, a));
            }
            And => C::And_z_zz_,
            Or => C::Orr_z_zz_,
            Xor => C::Eor_z_zz_,
            _ => return None,
        };
        Some(Packed::Sve(code, a))
    } else {
        let a = if bitwise {
            A::B16
        } else {
            arrangements(size).0
        };
        let code = match op {
            Add => C::Add_asimdsame_only,
            Sub => C::Sub_asimdsame_only,
            Mul if size <= 4 => C::Mul_asimdsame_only,
            Min | Max if size == 8 => {
                let cmp = if signed {
                    C::Cmgt_asimdsame
                } else {
                    C::Cmhi_asimdsame
                };
                return Some(Packed::NeonMinMax64(cmp, op == Max));
            }
            Min if signed => C::Smin_asimd,
            Min => C::Umin_asimd,
            Max if signed => C::Smax_asimd,
            Max => C::Umax_asimd,
            SaturatingAdd if signed => C::Sqadd_asimd,
            SaturatingAdd => C::Uqadd_asimd,
            SaturatingSub if signed => C::Sqsub_asimd,
            SaturatingSub => C::Uqsub_asimd,
            Shl | Shr => {
                let code = if signed && op == Shr {
                    C::Sshl_asimd
                } else {
                    C::Ushl_asimd
                };
                return Some(Packed::NeonShift(code, a, op == Shr));
            }
            And => C::And_asimdsame_only,
            Or => C::Orr_asimdsame_only,
            Xor => C::Eor_asimdsame_only,
            _ => return None,
        };
        Some(Packed::Neon(code, a))
    }
}

fn packed_compare(op: Binary, ty: Type, profile: Profile) -> Option<(Packed, bool)> {
    use Binary::*;
    let Type::Vector(lane) = ty else { return None };
    if lane.bytes() > 8 || matches!(lane, Lane::F16 | Lane::BF16) {
        return None;
    }
    let (neon, sve_arr) = arrangements(lane.bytes());
    let swap = matches!(op, Lt | Le);
    let floating = lane.float();
    let signed = lane.signed();
    let scalable = sve(profile);
    let code = match (scalable, floating, signed, op) {
        (true, true, _, Eq) => C::Fcmeq_p_p_zz,
        (true, true, _, Ne) => C::Fcmne_p_p_zz,
        (true, true, _, Gt | Lt) => C::Fcmgt_p_p_zz,
        (true, true, _, Ge | Le) => C::Fcmge_p_p_zz,
        (true, false, _, Eq) => C::Cmpeq_p_p_zz,
        (true, false, _, Ne) => C::Cmpne_p_p_zz,
        (true, false, true, Gt | Lt) => C::Cmpgt_p_p_zz,
        (true, false, true, Ge | Le) => C::Cmpge_p_p_zz,
        (true, false, false, Gt | Lt) => C::Cmphi_p_p_zz,
        (true, false, false, Ge | Le) => C::Cmphs_p_p_zz,
        (false, true, _, Eq | Ne) => C::Fcmeq_asimdsame,
        (false, true, _, Gt | Lt) => C::Fcmgt_asimdsame,
        (false, true, _, Ge | Le) => C::Fcmge_asimdsame,
        (false, false, _, Eq | Ne) => C::Cmeq_asimdsame,
        (false, false, true, Gt | Lt) => C::Cmgt_asimdsame,
        (false, false, true, Ge | Le) => C::Cmge_asimdsame,
        (false, false, false, Gt | Lt) => C::Cmhi_asimdsame,
        (false, false, false, Ge | Le) => C::Cmhs_asimdsame,
        _ => return None,
    };
    let packed = if scalable {
        Packed::SveCompare {
            code,
            arrangement: sve_arr,
        }
    } else {
        Packed::NeonCompare {
            code,
            arrangement: neon,
            invert: op == Ne,
        }
    };
    Some((packed, swap))
}

fn packed_shift_immediate(op: Binary, lane: Lane, profile: Profile) -> (C, A) {
    let (neon, scalable) = arrangements(lane.bytes());
    match (sve(profile), op, lane.signed()) {
        (true, Binary::Shl, _) => (C::Lsl_z_zi, scalable),
        (true, _, true) => (C::Asr_z_zi, scalable),
        (true, _, false) => (C::Lsr_z_zi, scalable),
        (false, Binary::Shl, _) => (C::Shl_asimd_imm, neon),
        (false, _, true) => (C::Sshr_asimd_imm, neon),
        (false, _, false) => (C::Ushr_asimd_imm, neon),
    }
}
