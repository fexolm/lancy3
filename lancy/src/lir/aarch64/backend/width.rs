use super::{
    A, C, Emitter, O, PredQual, R, Result, arrangements, groups::z, operations::sve_lane_bytes,
};
use crate::hir::Lane;

#[derive(Clone, Copy, Debug)]
pub(super) enum Widen {
    Neon(C, A, A),
    Sve {
        code: C,
        from: A,
        to: A,
        high: bool,
    },
    Wide {
        signed: bool,
        scalable: bool,
        high: bool,
    },
    Convert(Conversion, bool),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Narrow {
    Neon { codes: [C; 2], from: A, to: [A; 2] },
    Sve { to: A, float: bool },
    NeonWide,
    Convert(Conversion),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Conversion {
    pub from: Lane,
    pub to: Lane,
    pub scalable: bool,
}

impl Emitter<'_> {
    pub(super) fn widen(&mut self, form: Widen, src: u8, dst: u8) -> Result {
        match form {
            Widen::Convert(form, high) => {
                let count = self.config.bytes / form.to.bytes();
                let start = if high { count } else { 0 };
                self.convert_width(form, src, dst, start, 0, count)
            }
            Widen::Wide {
                signed,
                scalable,
                high,
            } => self.widen_to_pair(signed, scalable, high, src, dst),
            Widen::Neon(code, to, from) => self.a.put(
                code,
                &[
                    O::vec(R::v(dst).unwrap(), to),
                    O::vec(R::v(src).unwrap(), from),
                ],
            ),
            Widen::Sve {
                code,
                from,
                to,
                high,
            } => {
                let z = |n, arr| O::vec(R::z(n).unwrap(), arr);
                let src = self.sve_widen_source(src, high)?;
                if code == C::Fcvt_z_d_p_s {
                    let packed = z(16, from);
                    self.a
                        .put(C::Zip1_z_zz, &[packed, z(src, from), z(src, from)])?;
                    self.predicate(R::P1, to, self.config.bytes / sve_lane_bytes(to))?;
                    self.a
                        .put(code, &[z(dst, to), O::pred(R::P1, PredQual::Merge), packed])
                } else {
                    self.a.put(code, &[z(dst, to), z(src, from)])
                }
            }
        }
    }

    pub(super) fn narrow(&mut self, form: Narrow, a: u8, b: u8, dst: u8) -> Result {
        match form {
            Narrow::Convert(form) => {
                let half = self.config.bytes / form.from.bytes();
                self.convert_width(form, a, dst, 0, 0, half)?;
                self.convert_width(form, b, dst, 0, half, half)
            }
            Narrow::NeonWide => {
                let v = |n| O::vec(R::v(n).unwrap(), A::D2);
                self.a.put(C::Uzp1_asimd, &[v(dst), v(a), v(b)])
            }
            Narrow::Neon { codes, from, to } => {
                let v = |n, arr| O::vec(R::v(n).unwrap(), arr);
                let acc = if dst == b { 16 } else { dst };
                self.a.put(codes[0], &[v(acc, to[0]), v(a, from)])?;
                self.a.put(codes[1], &[v(acc, to[1]), v(b, from)])?;
                self.copy_vec(dst, acc, false)
            }
            Narrow::Sve { to, float } => {
                let z = |n, arr| O::vec(R::z(n).unwrap(), arr);
                let (d, scratch) = (z(dst, to), z(16, to));
                let half = self.config.bytes / (2 * sve_lane_bytes(to));
                let (a, b) = if float {
                    self.predicate(R::P1, A::D, half)?;
                    let p = O::pred(R::P1, PredQual::Merge);
                    self.a.put(C::Fcvt_z_s_p_d, &[scratch, p, z(b, A::D)])?;
                    self.a.put(C::Fcvt_z_s_p_d, &[d, p, z(a, A::D)])?;
                    (d, scratch)
                } else {
                    (z(a, to), z(b, to))
                };
                self.a.put(C::Uzp1_z_zz, &[scratch, b, b])?;
                self.a.put(C::Uzp1_z_zz, &[d, a, a])?;
                self.predicate(R::P1, to, half)?;
                self.a.put(
                    C::Splice_z_p_zz,
                    &[d, O::pred(R::P1, PredQual::None), d, scratch],
                )
            }
        }
    }

    fn convert_width(
        &mut self,
        form: Conversion,
        src: u8,
        dst: u8,
        start: usize,
        output: usize,
        count: usize,
    ) -> Result {
        let source = form.from.bytes();
        let target = form.to.bytes();
        if form.scalable {
            let source_arr = arrangements(source.min(8)).1;
            let target_arr = arrangements(target.min(8)).1;
            let input_step = if source == 16 { 2 } else { 1 };
            let output_step = if target == 16 { 2 } else { 1 };
            // The two cursors follow different element sizes and stay within
            // the requested halves of the logical vector.
            self.predicate(R::P5, source_arr, (start + count) * input_step)?;
            if start == 0 {
                self.a.put(C::Pfalse_p, &[O::vec(R::P2, A::B)])?;
            } else {
                self.predicate(R::P2, source_arr, start * input_step)?;
            }
            return self.sve_lane_range(
                target_arr,
                output * output_step,
                (output + count) * output_step,
                |e| {
                    e.sve_next_predicated_lane(source_arr, R::P2, R::P5)?;
                    e.sve_predicated_lane_to_int(source_arr, src, 9, R::P2)?;
                    if source == 16 {
                        e.sve_next_predicated_lane(source_arr, R::P2, R::P5)?;
                        e.sve_predicated_lane_to_int(source_arr, src, 10, R::P2)?;
                    }
                    e.cast_width_lane(form)?;
                    e.sve_lane_from_int(target_arr, 9, dst, R::P1)?;
                    if target == 16 {
                        e.sve_next_lane(A::D)?;
                        e.sve_lane_from_int(A::D, 10, dst, R::P1)?;
                    }
                    Ok(())
                },
            );
        }
        for index in 0..count {
            if source == 16 {
                self.pair_to_gpr(src, 9, 10)?;
            } else {
                self.neon_lane_to_int(src, 9, source, start + index)?;
            }
            self.cast_width_lane(form)?;
            if target == 16 {
                self.pair_from_gpr(dst, 9, 10)?;
            } else {
                self.neon_lane_from_int(9, dst, target, output + index)?;
            }
        }
        Ok(())
    }

    fn cast_width_lane(&mut self, form: Conversion) -> Result {
        let Conversion { from, to, .. } = form;
        if from.float() && to.float() {
            if from.bytes() == 2 {
                self.half_cast(from, to, 9, 16)?;
                return self.float_move(to.scalar(), false, 16, 9);
            }
            self.float_move(from.scalar(), true, 16, 9)?;
            return self.half_cast(from, to, 16, 9);
        }
        if from.float() {
            let code = if to.signed() {
                C::Fcvtzs_scalar
            } else {
                C::Fcvtzu_scalar
            };
            self.float_move(from.scalar(), true, 16, 9)?;
            self.float_cast(code, from.scalar(), to.scalar(), 16, 12)?;
            self.copy_converted_lane()
        } else {
            let code = if from.signed() {
                C::Scvtf_scalar
            } else {
                C::Ucvtf_scalar
            };
            self.float_cast(code, from.scalar(), to.scalar(), 9, 16)?;
            self.float_move(to.scalar(), false, 16, 9)
        }
    }

    fn copy_converted_lane(&mut self) -> Result {
        self.a.put(
            C::Orr_64_log_shift,
            &[O::reg(R::X9), O::reg(R::XZR), O::reg(R::X12)],
        )
    }

    fn sve_widen_source(&mut self, src: u8, high: bool) -> Result<u8> {
        if !high {
            return Ok(src);
        }
        // Select the upper half of the logical vector before interleaving or unpacking.
        self.copy_vec(16, src, true)?;
        let scratch = z(16, A::B);
        self.a.put(
            C::Ext_z_zzi,
            &[
                scratch,
                scratch,
                scratch,
                O::imm((self.config.bytes / 2) as u64),
            ],
        )?;
        Ok(16)
    }

    fn widen_to_pair(
        &mut self,
        signed: bool,
        scalable: bool,
        high: bool,
        src: u8,
        dst: u8,
    ) -> Result {
        if scalable {
            let src = self.sve_widen_source(src, high)?;
            let low = z(src, A::D);
            let upper = if high { z(dst, A::D) } else { z(16, A::D) };
            if signed {
                self.a.put(C::Asr_z_zi, &[upper, low, O::imm(63)])?;
            } else {
                self.a.put(C::Eor_z_zz_, &[upper, upper, upper])?;
            }
            return self.a.put(C::Zip1_z_zz, &[z(dst, A::D), low, upper]);
        }
        self.neon_lane_to_int(src, 9, 8, usize::from(high))?;
        if signed {
            self.a.put(
                C::Sbfm_64m_bitfield,
                &[O::reg(R::X10), O::reg(R::X9), O::imm(63), O::imm(63)],
            )?;
        }
        self.pair_from_gpr(dst, 9, if signed { 10 } else { 31 })
    }
}
