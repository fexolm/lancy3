use super::{
    A, C, Cond, Emitter, O, PredQual, R, Result, ShiftKind, arrangements, half,
    inst::{IntBinary, IntOp},
    operations::sve_lane_bytes,
    wide::MathForm,
};
use crate::hir::{Binary, Lane, Type};

#[derive(Clone, Copy, Debug)]
pub(super) enum Form {
    Integer(C, A, usize),
    NeonBits(C, usize),
    NeonMinMax64(Cond),
    FloatAdd(A, bool),
    Ordered(Binary, Lane, bool),
}

impl Emitter<'_> {
    pub(super) fn reduce(&mut self, form: Form, src: u8, dst: u8) -> Result {
        match form {
            Form::Integer(code, arr, size) => self.reduce_int(code, arr, size, src, dst),
            Form::NeonBits(code, size) => {
                self.pair_to_gpr(src, 9, 10)?;
                let r = O::reg;
                let x9 = r(R::X9);
                let out = r(R::x(dst).unwrap());
                self.a
                    .put(code, &[if size == 8 { out } else { x9 }, x9, r(R::X10)])?;
                for shift in [32, 16, 8]
                    .into_iter()
                    .take(3 - size.trailing_zeros() as usize)
                {
                    self.a.put(
                        code,
                        &[
                            if shift as usize == size * 8 { out } else { x9 },
                            x9,
                            O::shifted(R::X9, ShiftKind::Lsr, shift),
                        ],
                    )?;
                }
                self.truncate_reduction(size, dst)
            }
            Form::NeonMinMax64(cond) => {
                self.pair_to_gpr(src, 9, 10)?;
                let int = IntBinary {
                    op: IntOp::MinMax(cond),
                    size: 8,
                    signed_inputs: false,
                    wrap_shift: false,
                };
                self.int_binary(int, 9, 10, dst)
            }
            Form::FloatAdd(arr, scalable) => self.reduce_float_add(arr, scalable, src, dst),
            Form::Ordered(op, lane, scalable) => self.reduce_ordered(op, lane, scalable, src, dst),
        }
    }

    fn reduce_ordered(
        &mut self,
        op: Binary,
        lane: Lane,
        scalable: bool,
        src: u8,
        dst: u8,
    ) -> Result {
        let size = lane.bytes();
        if size == 16 {
            return self.wide_reduce(
                MathForm {
                    op,
                    signed: lane.signed(),
                    packed: false,
                    scalable,
                },
                src,
                dst,
            );
        }
        use Binary::*;
        let bits = match op {
            And => Some(C::And_64_log_shift),
            Or => Some(C::Orr_64_log_shift),
            Xor => Some(C::Eor_64_log_shift),
            _ => None,
        };
        let arr = arrangements(size).1;
        if lane.float() && size >= 4 && bits.is_none() {
            let v = |n| O::vec(R::v(n).unwrap(), arr);
            let d = v(dst);
            self.a.put(C::Fmov_scalar, &[d, v(src)])?;
            return self.fold_lanes(scalable, size, src, |e| {
                e.float_move(lane.scalar(), true, 16, 10)?;
                match op {
                    Mul => e.a.put(C::Fmul_scalar, &[d, d, v(16)]),
                    Min | Max => e.ordered_minmax(op == Min, d, v(16)),
                    _ => unreachable!("selected floating reduction"),
                }
            });
        }
        self.neon_lane_to_int(src, 9, size, 0)?;
        self.fold_lanes(scalable, size, src, |e| {
            let (acc, rhs) = (O::reg(R::X9), O::reg(R::X10));
            if let Some(code) = bits {
                return e.a.put(code, &[acc, acc, rhs]);
            }
            if !lane.float() {
                debug_assert_eq!(op, Mul);
                return e
                    .a
                    .put(C::Madd_64a_dp_3src, &[acc, acc, rhs, O::reg(R::XZR)]);
            }
            let code = match op {
                Add => C::Fadd_scalar,
                Mul => C::Fmul_scalar,
                Min => C::Fminnm_scalar,
                Max => C::Fmaxnm_scalar,
                _ => unreachable!("selected half reduction"),
            };
            let form = half::Form {
                lane,
                packed: false,
                scalable: false,
                action: half::Action::Binary(code),
            };
            e.half_lane(form, 9, [Some(10), None], 9)
        })?;
        if lane.float() && size >= 4 {
            return self.float_move(lane.scalar(), true, dst, 9);
        }
        self.int_cast(Type::U64, lane.scalar(), 9, dst)
    }

    fn fold_lanes(
        &mut self,
        scalable: bool,
        size: usize,
        src: u8,
        mut step: impl FnMut(&mut Self) -> Result,
    ) -> Result {
        if scalable {
            let arr = arrangements(size).1;
            return self.sve_lanes_after(arr, 1, |e| {
                e.sve_lane_to_int(arr, src, 10)?;
                step(e)
            });
        }
        for lane in 1..16 / size {
            self.neon_lane_to_int(src, 10, size, lane)?;
            step(self)?;
        }
        Ok(())
    }

    pub(super) fn ordered_minmax(&mut self, min: bool, acc: O, rhs: O) -> Result {
        // FMINNM/FMAXNM propagate signaling NaNs even when one operand is numeric.
        let done = self.a.label();
        let replace = self.a.label();
        self.a.put(C::Fcmp_scalar, &[rhs, rhs])?;
        self.a.jump_if(Cond::Vs, done)?;
        self.a.put(C::Fcmp_scalar, &[acc, acc])?;
        self.a.jump_if(Cond::Vs, replace)?;
        let code = if min {
            C::Fminnm_scalar
        } else {
            C::Fmaxnm_scalar
        };
        self.a.put(code, &[acc, acc, rhs])?;
        self.a.jump(done);
        self.a.bind(replace);
        self.a.put(C::Fmov_scalar, &[acc, rhs])?;
        self.a.bind(done);
        Ok(())
    }

    fn reduce_int(&mut self, code: C, arr: A, size: usize, src: u8, dst: u8) -> Result {
        let out = O::reg(R::V16);
        if matches!(arr, A::B | A::H | A::S | A::D) {
            self.predicate(R::P1, arr, self.config.bytes / size)?;
            self.a.put(
                code,
                &[
                    out,
                    O::pred(R::P1, PredQual::None),
                    O::vec(R::z(src).unwrap(), arr),
                ],
            )?;
        } else {
            self.a.put(code, &[out, O::vec(R::v(src).unwrap(), arr)])?;
        }
        let ty = if size == 8 { Type::F64 } else { Type::F32 };
        self.float_move(ty, false, 16, dst)?;
        self.truncate_reduction(size, dst)
    }

    fn truncate_reduction(&mut self, size: usize, dst: u8) -> Result {
        if size < 4 {
            let to = O::reg(R::w(dst).unwrap());
            self.a.put(
                C::Ubfm_32m_bitfield,
                &[to, to, O::imm(0), O::imm((size * 8 - 1) as u64)],
            )?;
        }
        Ok(())
    }

    fn reduce_float_add(&mut self, arr: A, scalable: bool, src: u8, dst: u8) -> Result {
        let v = |n, a| O::vec(R::v(n).unwrap(), a);
        let d = v(dst, arr);
        if scalable {
            self.a.put(C::Fmov_scalar, &[d, v(src, arr)])?;
            self.predicate(R::P1, arr, self.config.bytes / sve_lane_bytes(arr))?;
            self.predicate(R::P2, arr, 1)?;
            let p1 = O::vec(R::P1, A::B);
            self.a.put(
                C::Bic_p_p_pp,
                &[p1, O::pred(R::P1, PredQual::Zero), p1, O::vec(R::P2, A::B)],
            )?;
            self.a.put(
                C::Fadda_scalar_p_z,
                &[
                    d,
                    O::pred(R::P1, PredQual::None),
                    d,
                    O::vec(R::z(src).unwrap(), arr),
                ],
            )
        } else if arr == A::D {
            self.a.put(C::Faddp_d_asimd, &[d, v(src, A::D2)])
        } else {
            self.a.put(C::Fmov_scalar, &[d, v(src, A::S)])?;
            let lane = v(16, A::S);
            for index in 1..4 {
                self.a
                    .put(C::Dup_s_lane_asimd, &[lane, v(src, A::S4), O::imm(index)])?;
                self.a.put(C::Fadd_scalar, &[d, d, lane])?;
            }
            Ok(())
        }
    }
}
