use super::{
    A, C, Cond, Emitter, O, PredQual, R, Result, arrangements,
    groups::z,
    inst::{IntBinary, IntOp, VectorDivide},
};
use crate::Trap;

impl Emitter<'_> {
    pub(super) fn vector_divide(&mut self, form: VectorDivide, a: u8, b: u8, dst: u8) -> Result {
        if !form.scalable {
            let int = IntBinary {
                op: IntOp::Divide {
                    signed: form.signed,
                    remainder: form.remainder,
                },
                size: form.size,
                signed_inputs: form.signed,
                wrap_shift: false,
            };
            for lane in 0..16 / form.size {
                self.neon_lane_to_int(a, 9, form.size, lane)?;
                self.neon_lane_to_int(b, 10, form.size, lane)?;
                self.int_binary(int, 9, 10, 11)?;
                self.neon_lane_from_int(11, dst, form.size, lane)?;
            }
            return Ok(());
        }
        if form.size >= 4 {
            let arr = arrangements(form.size).1;
            self.predicate(R::P4, arr, self.config.bytes / form.size)?;
            return self.sve_divide_group(form, arr, [a, b], dst, 16);
        }

        let arr = arrangements(form.size).1;
        let lanes = self.config.bytes / 4;
        self.predicate(R::P4, A::S, lanes)?;
        // The groups execute in increasing lane order, including their trap checks.
        for offset in (0..self.config.bytes).step_by(lanes * form.size) {
            self.sve_widen_group(a, 16, form.size, offset, form.signed)?;
            self.sve_widen_group(b, 17, form.size, offset, form.signed)?;
            self.sve_divide_group(form, A::S, [16, 17], 16, 18)?;
            self.sve_narrow_group(16, form.size)?;
            if offset == 0 {
                self.copy_vec(dst, 16, true)?;
            } else {
                self.a.imm(R::X13, (offset / form.size) as u64)?;
                self.sve_prefix(R::P5, arr, R::X13)?;
                self.sve_splice(arr, dst, R::P5, 16)?;
            }
        }
        Ok(())
    }

    fn sve_divide_group(
        &mut self,
        form: VectorDivide,
        arr: A,
        [a, b]: [u8; 2],
        dst: u8,
        scratch: u8,
    ) -> Result {
        let p = |n| O::vec(n, arr);
        let zero = |n| O::pred(n, PredQual::Zero);
        let active = |n| O::pred(n, PredQual::None);
        self.a.put(
            C::Cmpeq_p_p_zi,
            &[p(R::P1), zero(R::P4), z(b, arr), O::imm(0)],
        )?;
        let zero_divisor_prefix = if form.signed {
            let min = i64::MIN >> (64 - 8 * form.size);
            let bits = if arr == A::S {
                min as u32 as u64
            } else {
                min as u64
            };
            self.a.put(C::Dupm_z, &[z(scratch, arr), O::imm(bits)])?;
            self.a.put(
                C::Cmpeq_p_p_zi,
                &[p(R::P2), zero(R::P4), z(b, arr), O::simm(-1)],
            )?;
            self.a.put(
                C::Cmpeq_p_p_zz,
                &[p(R::P2), zero(R::P2), z(a, arr), z(scratch, arr)],
            )?;
            // Only a zero divisor strictly before the first overflow wins.
            self.a.put(
                C::Brkb_p_p_p,
                &[O::vec(R::P3, A::B), zero(R::P4), O::vec(R::P2, A::B)],
            )?;
            R::P3
        } else {
            R::P4
        };
        self.a.put(
            C::Ptest_p_p,
            &[active(zero_divisor_prefix), O::vec(R::P1, A::B)],
        )?;
        self.trap_if(Cond::Ne, Trap::DivisionByZero)?;
        if form.signed {
            self.a
                .put(C::Ptest_p_p, &[active(R::P4), O::vec(R::P2, A::B)])?;
            self.trap_if(Cond::Ne, Trap::DivisionOverflow)?;
        }

        let div = if form.signed {
            C::Sdiv_z_p_zz
        } else {
            C::Udiv_z_p_zz
        };
        let pg = O::pred(R::P4, PredQual::Merge);
        let quotient = if form.remainder { scratch } else { dst };
        self.copy_vec(quotient, a, true)?;
        let q = z(quotient, arr);
        self.a.put(div, &[q, pg, q, z(b, arr)])?;
        if form.remainder {
            self.copy_vec(dst, a, true)?;
            self.a
                .put(C::Mls_z_p_zz, &[z(dst, arr), pg, q, z(b, arr)])?;
        }
        Ok(())
    }
}
