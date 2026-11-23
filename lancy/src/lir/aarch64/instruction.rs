use super::operand::{Arrangement as A, Bank, Operand as O, PredQual, ShiftKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {
    Neon,
    Sve,
    Sve2,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FeatureSet(u8);

impl FeatureSet {
    pub const NONE: Self = Self(0);
    pub const ALL: Self = Self(7);
    pub const fn with(self, feature: Feature) -> Self {
        Self(
            self.0
                | match feature {
                    Feature::Neon => 1,
                    Feature::Sve => 2,
                    Feature::Sve2 => 6,
                },
        )
    }
    pub const fn contains(self, feature: Feature) -> bool {
        self.0 & Self::NONE.with(feature).0 == Self::NONE.with(feature).0
    }
}

/// Exact instruction forms supported by the Lancy encoder.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Code {
    Pnext_p_p_p,
    Orr_p_p_pp,
    Lastb_r_p_z,
    Cpy_z_p_r,
    Fcvt_d_h,
    Fcvt_h_d,
    Fcvtxn_s_d,
    Fminnm_scalar,
    Fmaxnm_scalar,
    And_32_log_imm,
    And_64_log_imm,
    Adcs_64,
    Sbcs_64,
    Umulh_64,
    And_z_zi,
    Trn1_z_zz,
    Sdiv_z_p_zz,
    Udiv_z_p_zz,
    Mls_z_p_zz,
    Cmpeq_minus1_p_p_z,
    Brkb_p_p_p,
    Ptest_p_p,
    Whilelo_p_x_x,
    Cntp_x_p_p,
    Umov_lane,
    Ins_lane,
    Ld1b_z_p_bz_sxtw,
    Ld1h_z_p_bz_sxtw,
    St1b_z_p_bz_sxtw,
    St1h_z_p_bz_sxtw,
    St1w_z_p_bz_sxtw,
    St1d_z_p_bz_lsl,
    Adc_64,
    Sbc_64,
    Pfalse_p,
    Trn1_p_pp,
    Trn2_z_zz,
    Trn2_asimd,
    Dup_q0_z,
    Ucvtf_half_z_p_z,
    Sminv_asimd,
    Smaxv_asimd,
    Uminv_asimd,
    Umaxv_asimd,
    Sminv_v_p_z,
    Smaxv_v_p_z,
    Uminv_v_p_z,
    Umaxv_v_p_z,
    Andv_v_p_z,
    Orv_v_p_z,
    Eorv_v_p_z,
    Fadda_scalar_p_z,
    Bic_p_p_pp,
    Fmov_scalar,
    Dup_s_lane_asimd,
    Faddp_d_asimd,
    Sxtl_asimd,
    Sxtl2_asimd,
    Uxtl_asimd,
    Uxtl2_asimd,
    Xtn_asimd,
    Xtn2_asimd,
    Fcvtl_asimd,
    Fcvtl2_asimd,
    Fcvtn_asimd,
    Fcvtn2_asimd,
    Sunpklo_z,
    Uunpklo_z,
    Zip1_z_zz,
    Uzp1_z_zz,
    Uzp1_asimd,
    Fcvt_z_d_p_s,
    Fcvt_z_s_p_d,
    Shl_asimd_imm,
    Sshr_asimd_imm,
    Ushr_asimd_imm,
    Lsl_z_zi,
    Lsr_z_zi,
    Asr_z_zi,
    Tbl_z_zz,
    Sub_z_zi,
    Cmphi_p_p_zi,
    Cmpls_p_p_zi,
    Cpy_zero_z_p,
    Histcnt_z_p_zz,
    Tbl_asimd,
    Tbl2_asimd,
    Tbx_asimd,
    Ext_asimd,
    Ext_z_zzi,
    Splice_z_p_zz,
    Cpy_one_z_p,
    Smin_asimd,
    Smax_asimd,
    Umin_asimd,
    Umax_asimd,
    Sqadd_asimd,
    Uqadd_asimd,
    Sqsub_asimd,
    Uqsub_asimd,
    Sshl_asimd,
    Ushl_asimd,
    Bit_asimd,
    Bif_asimd,
    Movi_asimd,
    Dupm_z,
    Smin_z_p_zz,
    Smax_z_p_zz,
    Umin_z_p_zz,
    Umax_z_p_zz,
    Sqadd_z_zz,
    Uqadd_z_zz,
    Sqsub_z_zz,
    Uqsub_z_zz,
    Lsl_z_p_zz,
    Lsr_z_p_zz,
    Asr_z_p_zz,
    Fneg_scalar,
    Fabs_scalar,
    Fsqrt_scalar,
    Frintm_scalar,
    Frintp_scalar,
    Frintz_scalar,
    Frinta_scalar,
    Frintn_scalar,
    Fneg_asimd,
    Fabs_asimd,
    Fsqrt_asimd,
    Frintm_asimd,
    Frintp_asimd,
    Frintz_asimd,
    Frinta_asimd,
    Frintn_asimd,
    Fneg_z_p_z,
    Fabs_z_p_z,
    Fsqrt_z_p_z,
    Frintm_z_p_z,
    Frintp_z_p_z,
    Frintz_z_p_z,
    Frinta_z_p_z,
    Frintn_z_p_z,
    Not_z_p_z,
    Neg_z_p_z,
    Abs_z_p_z,
    Clz_z_p_z,
    Rbit_z_p_z,
    Cnt_z_p_z,
    Orn_32_log_shift,
    Orn_64_log_shift,
    Orr_32_log_imm,
    Orr_64_log_imm,
    Eor_32_log_imm,
    Clz_32,
    Clz_64,
    Rbit_32,
    Rbit_64,
    Csneg_32_condsel,
    Csneg_64_condsel,
    Cnt_asimd,
    Neg_asimd,
    Abs_asimd,
    Clz_asimd,
    Rbit_asimd,
    Rev16_asimd,
    Rev32_asimd,
    Uaddlp_asimd,
    Mov_x_v_d1,
    Fmov_imm_scalar,
    Fmov_imm_asimd,
    Fmov_imm_z,
    Fmadd_scalar,
    Fmla_asimd,
    Fmla_z_p_zz,
    Compact_z_p_z,
    Fcvt_s_d,
    Fcvt_d_s,
    Scvtf_scalar,
    Ucvtf_scalar,
    Fcvtzs_scalar,
    Fcvtzu_scalar,
    Scvtf_asimd,
    Ucvtf_asimd,
    Fcvtzs_asimd,
    Fcvtzu_asimd,
    Scvtf_z_p_z,
    Ucvtf_z_p_z,
    Fcvtzs_z_p_z,
    Fcvtzu_z_p_z,
    Bsl_asimdsame,
    Bic_asimdsame,
    Sshr_sign_asimdshf,
    Bic_z_zz_,
    Fcsel_scalar,
    Addv_asimd,
    Addp_d_asimd,
    Uaddv_d_p_z,
    Fcmp_scalar,
    Cmeq_asimdsame,
    Cmgt_asimdsame,
    Cmge_asimdsame,
    Cmhi_asimdsame,
    Cmhs_asimdsame,
    Fcmeq_asimdsame,
    Fcmgt_asimdsame,
    Fcmge_asimdsame,
    Mvn_asimdmisc,
    Cmeq0_asimdmisc,
    Umaxv_b,
    Umov_w_b0,
    Cmpeq_p_p_zi0,
    Cmpne_p_p_zi0,
    Cmpeq_p_p_zz,
    Cmpne_p_p_zz,
    Cmpgt_p_p_zz,
    Cmpge_p_p_zz,
    Cmphi_p_p_zz,
    Cmphs_p_p_zz,
    Fcmeq_p_p_zz,
    Fcmne_p_p_zz,
    Fcmuo_p_p_zz,
    Fcmgt_p_p_zz,
    Fcmge_p_p_zz,
    Cpy_minus1_z_p,
    Index_0_1_z,
    Add_32_addsub_imm,
    Add_64_addsub_imm,
    Sub_32_addsub_imm,
    Sub_64_addsub_imm,
    Adds_32_addsub_imm,
    Adds_64_addsub_imm,
    Subs_32_addsub_imm,
    Subs_64_addsub_imm,
    Add_32_addsub_shift,
    Add_64_addsub_shift,
    Adds_64_addsub_shift,
    Sub_32_addsub_shift,
    Sub_64_addsub_shift,
    Subs_32_addsub_shift,
    Subs_64_addsub_shift,
    And_32_log_shift,
    And_64_log_shift,
    Orr_32_log_shift,
    Orr_64_log_shift,
    Eor_32_log_shift,
    Eor_64_log_shift,
    Madd_32a_dp_3src,
    Madd_64a_dp_3src,
    Msub_32a_dp_3src,
    Msub_64a_dp_3src,
    Sdiv_32_dp_2src,
    Sdiv_64_dp_2src,
    Udiv_32_dp_2src,
    Udiv_64_dp_2src,
    Movz_32_movewide,
    Movz_64_movewide,
    Movk_32_movewide,
    Movk_64_movewide,
    Fmov_s_w,
    Fmov_w_s,
    Fmov_d_x,
    Fmov_x_d,
    Fadd_scalar,
    Fsub_scalar,
    Fmul_scalar,
    Fdiv_scalar,
    Mov_v_d1_x,
    Sbfm_32m_bitfield,
    Sbfm_64m_bitfield,
    Ubfm_64m_bitfield,
    Add_64_addsub_sxtw,
    Ubfm_32m_bitfield,
    Lslv_32_dp_2src,
    Lslv_64_dp_2src,
    Rorv_32_dp_2src,
    Rorv_64_dp_2src,
    Asrv_32_dp_2src,
    Asrv_64_dp_2src,
    Lsrv_32_dp_2src,
    Lsrv_64_dp_2src,
    Csel_32_condsel,
    Csel_64_condsel,
    Csinv_32_condsel,
    Ldr_32_ldst_pos,
    Ldr_64_ldst_pos,
    Str_32_ldst_pos,
    Str_64_ldst_pos,
    Ldrb_32_ldst_pos,
    Strb_32_ldst_pos,
    Ldrh_32_ldst_pos,
    Strh_32_ldst_pos,
    Ldr_q_ldst_pos,
    Str_q_ldst_pos,
    Ldr_s_ldst_pos,
    Str_s_ldst_pos,
    Ldr_d_ldst_pos,
    Str_d_ldst_pos,
    Nop_HI_hints,
    B_only_branch_imm,
    Bl_only_branch_imm,
    B_only_condbranch,
    Cbz_32_compbranch,
    Cbz_64_compbranch,
    Cbnz_32_compbranch,
    Cbnz_64_compbranch,
    Br_64_branch_reg,
    Blr_64_branch_reg,
    Ret_64r_branch_reg,
    Add_asimdsame_only,
    Sub_asimdsame_only,
    Mul_asimdsame_only,
    And_asimdsame_only,
    Orr_asimdsame_only,
    Eor_asimdsame_only,
    Fadd_asimdsame,
    Fsub_asimdsame,
    Fmul_asimdsame,
    Fdiv_asimdsame,
    Dup_asimdgen,
    Saddl_asimddiff_L,
    Ptrue_p_s_,
    Add_z_zz_,
    Sub_z_zz_,
    Mul_z_zz_,
    Mul_z_p_zz_,
    Fadd_z_zz_,
    Fsub_z_zz_,
    Fmul_z_zz_,
    Fdiv_z_p_zz_,
    Fdivr_z_p_zz_,
    Dup_z_r,
    Cmplt_p_p_zi0,
    Sel_z_p_zz,
    Ld1w_z_p_bz_sxtw,
    Ld1d_z_p_bz_lsl,
    And_z_zz_,
    Orr_z_zz_,
    Eor_z_zz_,
    Ld1b_z_p_bi_u8,
    Ld1h_z_p_bi_u16,
    Ld1w_z_p_bi_u32,
    Ld1d_z_p_bi_u64,
    St1b_z_p_bi_,
    St1h_z_p_bi_,
    St1w_z_p_bi_,
    St1d_z_p_bi_,
    Eor3_z_zzz_,
    Bsl_z_zzz_,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EncodeError {
    #[error("expected {expected} operands, got {got}")]
    OperandCount { expected: usize, got: usize },
    #[error("invalid operand at index {0}")]
    Operand(usize),
    #[error("invalid instruction suffix {0}")]
    Suffix(u8),
    #[error("unaligned AArch64 instruction address {0:#x}")]
    Address(u64),
    #[error("branch from {pc:#x} to {target:#x} is out of range or unaligned")]
    Branch { pc: u64, target: u64 },
}

#[derive(Clone, Copy, Debug)]
struct Branch {
    target: u64,
    bits: u8,
    shift: u8,
}

impl Branch {
    fn field(self, pc: u64) -> Result<u32, EncodeError> {
        let delta = i128::from(self.target) - i128::from(pc);
        let limit = 1i128 << (self.bits + 1);
        if delta % 4 != 0 || !(-limit..limit).contains(&delta) {
            return Err(EncodeError::Branch {
                pc,
                target: self.target,
            });
        }
        Ok((((delta / 4) as u32) & ((1 << self.bits) - 1)) << self.shift)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Instruction {
    code: Code,
    pc: u64,
    word: u32,
    branch: Option<Branch>,
}

impl Instruction {
    pub fn with(code: Code, operands: &[O], pc: u64) -> Result<Self, EncodeError> {
        Self::with_suffix(code, 0, operands, pc)
    }

    pub fn with_suffix(
        code: Code,
        suffix: u8,
        operands: &[O],
        pc: u64,
    ) -> Result<Self, EncodeError> {
        aligned(pc)?;
        let (word, branch) = code.lower(suffix, operands)?;
        let instruction = Self {
            code,
            pc,
            word,
            branch,
        };
        instruction.encode()?;
        Ok(instruction)
    }

    pub fn code(&self) -> Code {
        self.code
    }
    pub fn pc(&self) -> u64 {
        self.pc
    }
    pub fn label_target(&self) -> Option<u64> {
        self.branch.map(|branch| branch.target)
    }
    pub fn encode(&self) -> Result<u32, EncodeError> {
        self.encode_at(self.pc)
    }
    pub fn encode_at(&self, pc: u64) -> Result<u32, EncodeError> {
        aligned(pc)?;
        Ok(self.word | self.branch.map_or(Ok(0), |branch| branch.field(pc))?)
    }
}

fn aligned(pc: u64) -> Result<(), EncodeError> {
    if pc.is_multiple_of(4) {
        Ok(())
    } else {
        Err(EncodeError::Address(pc))
    }
}

type Encoded = (u32, Option<Branch>);

impl Code {
    pub(super) fn encode(self, suffix: u8, operands: &[O], pc: u64) -> Result<u32, EncodeError> {
        aligned(pc)?;
        let (word, branch) = self.lower(suffix, operands)?;
        Ok(word | branch.map_or(Ok(0), |branch| branch.field(pc))?)
    }

    pub const fn required_feature(self) -> Option<Feature> {
        use Code::*;
        match self {
            Pnext_p_p_p | Orr_p_p_pp | Lastb_r_p_z | Cpy_z_p_r => Some(Feature::Sve),
            Fcvtxn_s_d => Some(Feature::Neon),
            And_z_zi | Trn1_z_zz => Some(Feature::Sve),
            Umov_lane | Ins_lane => Some(Feature::Neon),
            Sdiv_z_p_zz | Udiv_z_p_zz | Mls_z_p_zz | Cmpeq_minus1_p_p_z | Brkb_p_p_p
            | Ptest_p_p | Whilelo_p_x_x | Cntp_x_p_p | Ld1b_z_p_bz_sxtw | Ld1h_z_p_bz_sxtw
            | St1b_z_p_bz_sxtw | St1h_z_p_bz_sxtw | St1w_z_p_bz_sxtw | St1d_z_p_bz_lsl => {
                Some(Feature::Sve)
            }
            Trn2_asimd | Uzp1_asimd => Some(Feature::Neon),
            Pfalse_p | Trn1_p_pp | Trn2_z_zz | Dup_q0_z | Ucvtf_half_z_p_z => Some(Feature::Sve),
            Sminv_asimd | Smaxv_asimd | Uminv_asimd | Umaxv_asimd | Dup_s_lane_asimd
            | Faddp_d_asimd => Some(Feature::Neon),
            Sminv_v_p_z | Smaxv_v_p_z | Uminv_v_p_z | Umaxv_v_p_z | Andv_v_p_z | Orv_v_p_z
            | Eorv_v_p_z | Fadda_scalar_p_z | Bic_p_p_pp => Some(Feature::Sve),
            Sxtl_asimd | Sxtl2_asimd | Uxtl_asimd | Uxtl2_asimd | Xtn_asimd | Xtn2_asimd
            | Fcvtl_asimd | Fcvtl2_asimd | Fcvtn_asimd | Fcvtn2_asimd => Some(Feature::Neon),
            Sunpklo_z | Uunpklo_z | Zip1_z_zz | Uzp1_z_zz | Fcvt_z_d_p_s | Fcvt_z_s_p_d => {
                Some(Feature::Sve)
            }
            Shl_asimd_imm | Sshr_asimd_imm | Ushr_asimd_imm | Tbl_asimd | Tbl2_asimd
            | Tbx_asimd | Ext_asimd => Some(Feature::Neon),
            Ext_z_zzi | Splice_z_p_zz | Cpy_one_z_p => Some(Feature::Sve),
            Lsl_z_zi | Lsr_z_zi | Asr_z_zi | Tbl_z_zz | Sub_z_zi | Cmphi_p_p_zi | Cmpls_p_p_zi
            | Cpy_zero_z_p => Some(Feature::Sve),
            Histcnt_z_p_zz => Some(Feature::Sve2),
            Smin_asimd | Smax_asimd | Umin_asimd | Umax_asimd | Sqadd_asimd | Uqadd_asimd
            | Sqsub_asimd | Uqsub_asimd | Sshl_asimd | Ushl_asimd | Bit_asimd | Bif_asimd
            | Movi_asimd => Some(Feature::Neon),
            Dupm_z | Smin_z_p_zz | Smax_z_p_zz | Umin_z_p_zz | Umax_z_p_zz | Sqadd_z_zz
            | Uqadd_z_zz | Sqsub_z_zz | Uqsub_z_zz | Lsl_z_p_zz | Lsr_z_p_zz | Asr_z_p_zz => {
                Some(Feature::Sve)
            }
            Fneg_asimd | Fabs_asimd | Fsqrt_asimd | Frintm_asimd | Frintp_asimd | Frintz_asimd
            | Frinta_asimd | Frintn_asimd | Cnt_asimd | Neg_asimd | Abs_asimd | Clz_asimd
            | Rbit_asimd | Rev16_asimd | Rev32_asimd | Uaddlp_asimd | Mov_x_v_d1
            | Fmov_imm_asimd => Some(Feature::Neon),
            Fneg_z_p_z | Fabs_z_p_z | Fsqrt_z_p_z | Frintm_z_p_z | Frintp_z_p_z | Frintz_z_p_z
            | Frinta_z_p_z | Frintn_z_p_z | Not_z_p_z | Neg_z_p_z | Abs_z_p_z | Clz_z_p_z
            | Rbit_z_p_z | Cnt_z_p_z | Fmov_imm_z => Some(Feature::Sve),
            Fmla_asimd => Some(Feature::Neon),
            Fmla_z_p_zz | Compact_z_p_z => Some(Feature::Sve),
            Scvtf_asimd | Ucvtf_asimd | Fcvtzs_asimd | Fcvtzu_asimd => Some(Feature::Neon),
            Scvtf_z_p_z | Ucvtf_z_p_z | Fcvtzs_z_p_z | Fcvtzu_z_p_z => Some(Feature::Sve),
            Bsl_asimdsame | Bic_asimdsame | Sshr_sign_asimdshf | Addv_asimd | Addp_d_asimd => {
                Some(Feature::Neon)
            }
            Bic_z_zz_ | Uaddv_d_p_z => Some(Feature::Sve),
            Cmeq_asimdsame | Cmgt_asimdsame | Cmge_asimdsame | Cmhi_asimdsame | Cmhs_asimdsame
            | Fcmeq_asimdsame | Fcmgt_asimdsame | Fcmge_asimdsame | Mvn_asimdmisc
            | Cmeq0_asimdmisc | Umaxv_b | Umov_w_b0 => Some(Feature::Neon),
            Cmpeq_p_p_zi0 | Cmpne_p_p_zi0 | Cmpeq_p_p_zz | Cmpne_p_p_zz | Cmpgt_p_p_zz
            | Cmpge_p_p_zz | Cmphi_p_p_zz | Cmphs_p_p_zz | Fcmeq_p_p_zz | Fcmne_p_p_zz
            | Fcmgt_p_p_zz | Fcmge_p_p_zz | Fcmuo_p_p_zz | Cpy_minus1_z_p | Index_0_1_z => {
                Some(Feature::Sve)
            }
            Ptrue_p_s_ | Add_z_zz_ | Sub_z_zz_ | Mul_z_p_zz_ | And_z_zz_ | Orr_z_zz_
            | Eor_z_zz_ | Ld1b_z_p_bi_u8 | Ld1h_z_p_bi_u16 | Ld1w_z_p_bi_u32 | Ld1d_z_p_bi_u64
            | St1b_z_p_bi_ | St1h_z_p_bi_ | St1w_z_p_bi_ | St1d_z_p_bi_ | Fadd_z_zz_
            | Fsub_z_zz_ | Fmul_z_zz_ | Fdiv_z_p_zz_ | Fdivr_z_p_zz_ | Dup_z_r | Cmplt_p_p_zi0
            | Sel_z_p_zz | Ld1w_z_p_bz_sxtw | Ld1d_z_p_bz_lsl => Some(Feature::Sve),
            Eor3_z_zzz_ | Bsl_z_zzz_ | Mul_z_zz_ => Some(Feature::Sve2),
            Add_asimdsame_only | Sub_asimdsame_only | Mul_asimdsame_only | And_asimdsame_only
            | Orr_asimdsame_only | Eor_asimdsame_only | Saddl_asimddiff_L | Ldr_q_ldst_pos
            | Str_q_ldst_pos | Mov_v_d1_x | Fadd_asimdsame | Fsub_asimdsame | Fmul_asimdsame
            | Fdiv_asimdsame | Dup_asimdgen => Some(Feature::Neon),
            _ => None,
        }
    }

    pub fn is_available(self, features: FeatureSet) -> bool {
        self.required_feature().is_none_or(|f| features.contains(f))
    }

    fn lower(self, suffix: u8, operands: &[O]) -> Result<Encoded, EncodeError> {
        use Code::*;
        if suffix > u8::from(self == Saddl_asimddiff_L) {
            return Err(EncodeError::Suffix(suffix));
        }
        let (base, wide, form) = match self {
            Pnext_p_p_p => (0x2519c400, false, Form::PredNext),
            Orr_p_p_pp => (0x25804000, false, Form::SvePredBits),
            Lastb_r_p_z => (0x0521a000, false, Form::SveGprLane(false)),
            Cpy_z_p_r => (0x0528a000, false, Form::SveGprLane(true)),
            Fcvt_d_h => (0x1ee2c000, false, Form::ArrangedPair(A::D, A::H)),
            Fcvt_h_d => (0x1e63c000, false, Form::ArrangedPair(A::H, A::D)),
            Fcvtxn_s_d => (0x7e616800, false, Form::ArrangedPair(A::S, A::D)),
            Fminnm_scalar => (0x1e207800, false, Form::FloatScalar),
            Fmaxnm_scalar => (0x1e206800, false, Form::FloatScalar),
            And_32_log_imm => (0x12000000, false, Form::LogicalImmediate),
            And_64_log_imm => (0x12000000, true, Form::LogicalImmediate),
            Adcs_64 => (0xba000000, true, Form::Three),
            Sbcs_64 => (0xfa000000, true, Form::Three),
            Umulh_64 => (0x9bc07c00, true, Form::Three),
            And_z_zi => (0x05800000, false, Form::SveDupMask),
            Trn1_z_zz => (0x05207000, false, Form::SveArithmetic),
            Sdiv_z_p_zz => (0x04140000, false, Form::SveDestructiveWide),
            Udiv_z_p_zz => (0x04150000, false, Form::SveDestructiveWide),
            Mls_z_p_zz => (0x04006000, false, Form::SveMultiplyAccumulate),
            Cmpeq_minus1_p_p_z => (0x251f8000, false, Form::SveCompareZero),
            Brkb_p_p_p => (0x25904000, false, Form::PredBreak),
            Ptest_p_p => (0x2550c000, false, Form::PredTest),
            Whilelo_p_x_x => (0x25201c00, false, Form::PredWhile),
            Cntp_x_p_p => (0x25208000, false, Form::PredCount),
            Umov_lane => (0x0e003c00, false, Form::NeonGprLane(false)),
            Ins_lane => (0x4e001c00, false, Form::NeonGprLane(true)),
            Ld1b_z_p_bz_sxtw => (0x84404000, false, Form::SveIndexed(A::S, PredQual::Zero)),
            Ld1h_z_p_bz_sxtw => (0x84e04000, false, Form::SveIndexed(A::S, PredQual::Zero)),
            St1b_z_p_bz_sxtw => (0xe440c000, false, Form::SveIndexed(A::S, PredQual::None)),
            St1h_z_p_bz_sxtw => (0xe4e0c000, false, Form::SveIndexed(A::S, PredQual::None)),
            St1w_z_p_bz_sxtw => (0xe560c000, false, Form::SveIndexed(A::S, PredQual::None)),
            St1d_z_p_bz_lsl => (0xe5a0a000, false, Form::SveIndexed(A::D, PredQual::None)),
            Adc_64 => (0x1a000000, true, Form::Three),
            Sbc_64 => (0x5a000000, true, Form::Three),
            Pfalse_p => (0x2518e400, false, Form::PredFalse),
            Trn1_p_pp => (0x05205000, false, Form::PredPermute),
            Trn2_z_zz => (0x05207400, false, Form::SveArithmetic),
            Trn2_asimd => (0x0e006800, false, Form::Neon(false)),
            Dup_q0_z => (0x05302000, false, Form::RegPair(Bank::Z, Bank::Z)),
            Ucvtf_half_z_p_z => (0x6553a000, false, Form::SveFloatWidth(A::H, A::H)),
            Sminv_asimd => (0x4e31a800, false, Form::NeonSum(false)),
            Smaxv_asimd => (0x4e30a800, false, Form::NeonSum(false)),
            Uminv_asimd => (0x6e31a800, false, Form::NeonSum(false)),
            Umaxv_asimd => (0x6e30a800, false, Form::NeonSum(false)),
            Sminv_v_p_z => (0x040a2000, false, Form::SveSum),
            Smaxv_v_p_z => (0x04082000, false, Form::SveSum),
            Uminv_v_p_z => (0x040b2000, false, Form::SveSum),
            Umaxv_v_p_z => (0x04092000, false, Form::SveSum),
            Andv_v_p_z => (0x041a2000, false, Form::SveSum),
            Orv_v_p_z => (0x04182000, false, Form::SveSum),
            Eorv_v_p_z => (0x04192000, false, Form::SveSum),
            Fadda_scalar_p_z => (0x65182000, false, Form::SveFadda),
            Bic_p_p_pp => (0x25004010, false, Form::SvePredBits),
            Fmov_scalar => (0x1e204000, false, Form::FloatScalarUnary),
            Dup_s_lane_asimd => (0x5e040400, false, Form::NeonSingle),
            Faddp_d_asimd => (0x7e70d800, false, Form::ArrangedPair(A::D, A::D2)),
            Sxtl_asimd => (0x0f00a400, false, Form::NeonExtend(false)),
            Sxtl2_asimd => (0x4f00a400, false, Form::NeonExtend(true)),
            Uxtl_asimd => (0x2f00a400, false, Form::NeonExtend(false)),
            Uxtl2_asimd => (0x6f00a400, false, Form::NeonExtend(true)),
            Xtn_asimd => (0x0e212800, false, Form::NeonNarrow(false)),
            Xtn2_asimd => (0x4e212800, false, Form::NeonNarrow(true)),
            Fcvtl_asimd => (0x0e617800, false, Form::ArrangedPair(A::D2, A::S2)),
            Fcvtl2_asimd => (0x4e617800, false, Form::ArrangedPair(A::D2, A::S4)),
            Fcvtn_asimd => (0x0e616800, false, Form::ArrangedPair(A::S2, A::D2)),
            Fcvtn2_asimd => (0x4e616800, false, Form::ArrangedPair(A::S4, A::D2)),
            Sunpklo_z => (0x05303800, false, Form::SveExtend),
            Uunpklo_z => (0x05323800, false, Form::SveExtend),
            Zip1_z_zz => (0x05206000, false, Form::SveArithmetic),
            Uzp1_z_zz => (0x05206800, false, Form::SveArithmetic),
            Uzp1_asimd => (0x0e001800, false, Form::Neon(false)),
            Fcvt_z_d_p_s => (0x65cba000, false, Form::SveFloatWidth(A::D, A::S)),
            Fcvt_z_s_p_d => (0x65caa000, false, Form::SveFloatWidth(A::S, A::D)),
            Shl_asimd_imm => (
                0x4f005400,
                false,
                Form::VectorShiftImmediate {
                    sve: false,
                    left: true,
                },
            ),
            Sshr_asimd_imm => (
                0x4f000400,
                false,
                Form::VectorShiftImmediate {
                    sve: false,
                    left: false,
                },
            ),
            Ushr_asimd_imm => (
                0x6f000400,
                false,
                Form::VectorShiftImmediate {
                    sve: false,
                    left: false,
                },
            ),
            Lsl_z_zi => (
                0x04209c00,
                false,
                Form::VectorShiftImmediate {
                    sve: true,
                    left: true,
                },
            ),
            Lsr_z_zi => (
                0x04209400,
                false,
                Form::VectorShiftImmediate {
                    sve: true,
                    left: false,
                },
            ),
            Asr_z_zi => (
                0x04209000,
                false,
                Form::VectorShiftImmediate {
                    sve: true,
                    left: false,
                },
            ),
            Tbl_z_zz => (0x05203000, false, Form::SveArithmetic),
            Sub_z_zi => (0x2521c000, false, Form::SveSubImmediate),
            Cmphi_p_p_zi => (0x24200010, false, Form::SveUnsignedCompareImmediate),
            Cmpls_p_p_zi => (0x24202010, false, Form::SveUnsignedCompareImmediate),
            Cpy_zero_z_p => (0x05104000, false, Form::SveMaskZero),
            Histcnt_z_p_zz => (0x45a0c000, false, Form::SveHistcnt),
            Tbl_asimd => (0x0e000000, false, Form::NeonTable),
            Tbl2_asimd => (0x0e002000, false, Form::NeonTable),
            Tbx_asimd => (0x0e001000, false, Form::NeonTable),
            Ext_asimd => (0x6e000000, false, Form::NeonExt),
            Ext_z_zzi => (0x05200000, false, Form::SveExt),
            Splice_z_p_zz => (0x052c8000, false, Form::SveSplice),
            Cpy_one_z_p => (0x05100020, false, Form::SveMaskValue),
            Smin_asimd => (0x0e206c00, false, Form::Neon(true)),
            Smax_asimd => (0x0e206400, false, Form::Neon(true)),
            Umin_asimd => (0x2e206c00, false, Form::Neon(true)),
            Umax_asimd => (0x2e206400, false, Form::Neon(true)),
            Sqadd_asimd => (0x0e200c00, false, Form::Neon(false)),
            Uqadd_asimd => (0x2e200c00, false, Form::Neon(false)),
            Sqsub_asimd => (0x0e202c00, false, Form::Neon(false)),
            Uqsub_asimd => (0x2e202c00, false, Form::Neon(false)),
            Sshl_asimd => (0x0e204400, false, Form::Neon(false)),
            Ushl_asimd => (0x2e204400, false, Form::Neon(false)),
            Bit_asimd => (0x2ea01c00, false, Form::NeonBits),
            Bif_asimd => (0x2ee01c00, false, Form::NeonBits),
            Movi_asimd => (0x4f000400, false, Form::NeonIntImmediate),
            Dupm_z => (0x05c00000, false, Form::SveDupMask),
            Smin_z_p_zz => (0x040a0000, false, Form::SveDestructive),
            Smax_z_p_zz => (0x04080000, false, Form::SveDestructive),
            Umin_z_p_zz => (0x040b0000, false, Form::SveDestructive),
            Umax_z_p_zz => (0x04090000, false, Form::SveDestructive),
            Sqadd_z_zz => (0x04201000, false, Form::SveArithmetic),
            Uqadd_z_zz => (0x04201400, false, Form::SveArithmetic),
            Sqsub_z_zz => (0x04201800, false, Form::SveArithmetic),
            Uqsub_z_zz => (0x04201c00, false, Form::SveArithmetic),
            Lsl_z_p_zz => (0x04138000, false, Form::SveDestructive),
            Lsr_z_p_zz => (0x04118000, false, Form::SveDestructive),
            Asr_z_p_zz => (0x04108000, false, Form::SveDestructive),
            Fneg_scalar => (0x1e214000, false, Form::FloatScalarUnary),
            Fabs_scalar => (0x1e20c000, false, Form::FloatScalarUnary),
            Fsqrt_scalar => (0x1e21c000, false, Form::FloatScalarUnary),
            Frintm_scalar => (0x1e254000, false, Form::FloatScalarUnary),
            Frintp_scalar => (0x1e24c000, false, Form::FloatScalarUnary),
            Frintz_scalar => (0x1e25c000, false, Form::FloatScalarUnary),
            Frinta_scalar => (0x1e264000, false, Form::FloatScalarUnary),
            Frintn_scalar => (0x1e244000, false, Form::FloatScalarUnary),
            Fneg_asimd => (0x2ea0f800, false, Form::FloatNeonUnary),
            Fabs_asimd => (0x0ea0f800, false, Form::FloatNeonUnary),
            Fsqrt_asimd => (0x2ea1f800, false, Form::FloatNeonUnary),
            Frintm_asimd => (0x0e219800, false, Form::FloatNeonUnary),
            Frintp_asimd => (0x0ea18800, false, Form::FloatNeonUnary),
            Frintz_asimd => (0x0ea19800, false, Form::FloatNeonUnary),
            Frinta_asimd => (0x2e218800, false, Form::FloatNeonUnary),
            Frintn_asimd => (0x0e218800, false, Form::FloatNeonUnary),
            Fneg_z_p_z => (0x041da000, false, Form::SveUnary(1)),
            Fabs_z_p_z => (0x041ca000, false, Form::SveUnary(1)),
            Fsqrt_z_p_z => (0x650da000, false, Form::SveUnary(1)),
            Frintm_z_p_z => (0x6502a000, false, Form::SveUnary(1)),
            Frintp_z_p_z => (0x6501a000, false, Form::SveUnary(1)),
            Frintz_z_p_z => (0x6503a000, false, Form::SveUnary(1)),
            Frinta_z_p_z => (0x6504a000, false, Form::SveUnary(1)),
            Frintn_z_p_z => (0x6500a000, false, Form::SveUnary(1)),
            Not_z_p_z => (0x041ea000, false, Form::SveUnary(0)),
            Neg_z_p_z => (0x0417a000, false, Form::SveUnary(0)),
            Abs_z_p_z => (0x0416a000, false, Form::SveUnary(0)),
            Clz_z_p_z => (0x0419a000, false, Form::SveUnary(0)),
            Rbit_z_p_z => (0x05278000, false, Form::SveUnary(0)),
            Cnt_z_p_z => (0x041aa000, false, Form::SveUnary(0)),
            Orn_32_log_shift => (0x2a200000, false, Form::Shift(true)),
            Orn_64_log_shift => (0x2a200000, true, Form::Shift(true)),
            Orr_32_log_imm => (0x32000000, false, Form::LogicalImmediate),
            Orr_64_log_imm => (0x32000000, true, Form::LogicalImmediate),
            Eor_32_log_imm => (0x52000000, false, Form::LogicalImmediate),
            Clz_32 => (0x5ac01000, false, Form::RegPair(Bank::W, Bank::W)),
            Clz_64 => (0xdac01000, false, Form::RegPair(Bank::X, Bank::X)),
            Rbit_32 => (0x5ac00000, false, Form::RegPair(Bank::W, Bank::W)),
            Rbit_64 => (0xdac00000, false, Form::RegPair(Bank::X, Bank::X)),
            Csneg_32_condsel => (0x5a800400, false, Form::Select),
            Csneg_64_condsel => (0x5a800400, true, Form::Select),
            Cnt_asimd => (0x4e205800, false, Form::NeonUnary(0)),
            Neg_asimd => (0x6e20b800, false, Form::NeonUnary(3)),
            Abs_asimd => (0x4e20b800, false, Form::NeonUnary(3)),
            Clz_asimd => (0x6e204800, false, Form::NeonUnary(2)),
            Rbit_asimd => (0x6e605800, false, Form::NeonUnary(0)),
            Rev16_asimd => (0x4e201800, false, Form::NeonUnary(0)),
            Rev32_asimd => (0x6e200800, false, Form::NeonUnary(0)),
            Uaddlp_asimd => (0x6e202800, false, Form::NeonPairwiseLong),
            Mov_x_v_d1 => (0x4e183c00, false, Form::RegPair(Bank::X, Bank::V)),
            Fmov_imm_scalar => (0x1e201000, false, Form::FloatImmediate),
            Fmov_imm_asimd => (0x4f00f400, false, Form::FloatNeonImmediate),
            Fmov_imm_z => (0x2539c000, false, Form::SveFloatImmediate),
            Fmadd_scalar => (0x1f000000, false, Form::FloatFma),
            Fmla_asimd => (0x0e20cc00, false, Form::FloatNeon),
            Fmla_z_p_zz => (0x65200000, false, Form::SveFma),
            Compact_z_p_z => (0x05218000, false, Form::SveCompact),
            Fcvt_s_d => (0x1e624000, false, Form::RegPair(Bank::V, Bank::V)),
            Fcvt_d_s => (0x1e22c000, false, Form::RegPair(Bank::V, Bank::V)),
            Scvtf_scalar => (0x1e220000, false, Form::FloatInt(true)),
            Ucvtf_scalar => (0x1e230000, false, Form::FloatInt(true)),
            Fcvtzs_scalar => (0x1e380000, false, Form::FloatInt(false)),
            Fcvtzu_scalar => (0x1e390000, false, Form::FloatInt(false)),
            Scvtf_asimd => (0x0e21d800, false, Form::FloatNeonUnary),
            Ucvtf_asimd => (0x2e21d800, false, Form::FloatNeonUnary),
            Fcvtzs_asimd => (0x0ea1b800, false, Form::FloatNeonUnary),
            Fcvtzu_asimd => (0x2ea1b800, false, Form::FloatNeonUnary),
            Scvtf_z_p_z => (0x6594a000, false, Form::SveConvert),
            Ucvtf_z_p_z => (0x6595a000, false, Form::SveConvert),
            Fcvtzs_z_p_z => (0x659ca000, false, Form::SveConvert),
            Fcvtzu_z_p_z => (0x659da000, false, Form::SveConvert),
            Bsl_asimdsame => (0x2e601c00, false, Form::NeonBits),
            Bic_asimdsame => (0x0e601c00, false, Form::NeonBits),
            Sshr_sign_asimdshf => (0x4f000400, false, Form::NeonSign),
            Bic_z_zz_ => (0x04e03000, false, Form::SveBits),
            Fcsel_scalar => (0x1e200c00, false, Form::FloatSelect),
            Addv_asimd => (0x4e31b800, false, Form::NeonSum(false)),
            Addp_d_asimd => (0x5ef1b800, false, Form::NeonSum(true)),
            Uaddv_d_p_z => (0x04012000, false, Form::SveSum),
            Fcmp_scalar => (0x1e202000, false, Form::FloatCompare),
            Cmeq_asimdsame => (0x2e208c00, false, Form::Neon(false)),
            Cmgt_asimdsame => (0x0e203400, false, Form::Neon(false)),
            Cmge_asimdsame => (0x0e203c00, false, Form::Neon(false)),
            Cmhi_asimdsame => (0x2e203400, false, Form::Neon(false)),
            Cmhs_asimdsame => (0x2e203c00, false, Form::Neon(false)),
            Fcmeq_asimdsame => (0x0e20e400, false, Form::FloatNeon),
            Fcmgt_asimdsame => (0x2ea0e400, false, Form::FloatNeon),
            Fcmge_asimdsame => (0x2e20e400, false, Form::FloatNeon),
            Mvn_asimdmisc => (0x6e205800, false, Form::NeonUnary(0)),
            Cmeq0_asimdmisc => (0x4e209800, false, Form::NeonUnary(3)),
            Umaxv_b => (0x6e30a800, false, Form::RegPair(Bank::V, Bank::V)),
            Umov_w_b0 => (0x0e013c00, false, Form::RegPair(Bank::W, Bank::V)),
            Cmpeq_p_p_zi0 => (0x25008000, false, Form::SveCompareZero),
            Cmpne_p_p_zi0 => (0x25008010, false, Form::SveCompareZero),
            Cmpeq_p_p_zz => (0x2400a000, false, Form::SveCompare(false)),
            Cmpne_p_p_zz => (0x2400a010, false, Form::SveCompare(false)),
            Cmpgt_p_p_zz => (0x24008010, false, Form::SveCompare(false)),
            Cmpge_p_p_zz => (0x24008000, false, Form::SveCompare(false)),
            Cmphi_p_p_zz => (0x24000010, false, Form::SveCompare(false)),
            Cmphs_p_p_zz => (0x24000000, false, Form::SveCompare(false)),
            Fcmeq_p_p_zz => (0x65006000, false, Form::SveCompare(true)),
            Fcmne_p_p_zz => (0x65006010, false, Form::SveCompare(true)),
            Fcmuo_p_p_zz => (0x6500c000, false, Form::SveCompare(true)),
            Fcmgt_p_p_zz => (0x65004010, false, Form::SveCompare(true)),
            Fcmge_p_p_zz => (0x65004000, false, Form::SveCompare(true)),
            Cpy_minus1_z_p => (0x05101fe0, false, Form::SveMaskValue),
            Index_0_1_z => (0x04214000, false, Form::SveIndex),
            Add_32_addsub_imm => (0x11000000, false, Form::AddImm(false)),
            Add_64_addsub_imm => (0x11000000, true, Form::AddImm(false)),
            Sub_32_addsub_imm => (0x51000000, false, Form::AddImm(false)),
            Sub_64_addsub_imm => (0x51000000, true, Form::AddImm(false)),
            Adds_32_addsub_imm => (0x31000000, false, Form::AddImm(true)),
            Adds_64_addsub_imm => (0x31000000, true, Form::AddImm(true)),
            Subs_32_addsub_imm => (0x71000000, false, Form::AddImm(true)),
            Subs_64_addsub_imm => (0x71000000, true, Form::AddImm(true)),
            Add_32_addsub_shift => (0x0b000000, false, Form::Shift(false)),
            Add_64_addsub_shift => (0x0b000000, true, Form::Shift(false)),
            Adds_64_addsub_shift => (0x2b000000, true, Form::Shift(false)),
            Sub_32_addsub_shift => (0x4b000000, false, Form::Shift(false)),
            Sub_64_addsub_shift => (0x4b000000, true, Form::Shift(false)),
            Subs_32_addsub_shift => (0x6b000000, false, Form::Shift(false)),
            Subs_64_addsub_shift => (0x6b000000, true, Form::Shift(false)),
            And_32_log_shift => (0x0a000000, false, Form::Shift(true)),
            And_64_log_shift => (0x0a000000, true, Form::Shift(true)),
            Orr_32_log_shift => (0x2a000000, false, Form::Shift(true)),
            Orr_64_log_shift => (0x2a000000, true, Form::Shift(true)),
            Eor_32_log_shift => (0x4a000000, false, Form::Shift(true)),
            Eor_64_log_shift => (0x4a000000, true, Form::Shift(true)),
            Madd_32a_dp_3src => (0x1b000000, false, Form::Multiply),
            Madd_64a_dp_3src => (0x1b000000, true, Form::Multiply),
            Msub_32a_dp_3src => (0x1b008000, false, Form::Multiply),
            Msub_64a_dp_3src => (0x1b008000, true, Form::Multiply),
            Sdiv_32_dp_2src => (0x1ac00c00, false, Form::Three),
            Sdiv_64_dp_2src => (0x1ac00c00, true, Form::Three),
            Udiv_32_dp_2src => (0x1ac00800, false, Form::Three),
            Udiv_64_dp_2src => (0x1ac00800, true, Form::Three),
            Movz_32_movewide => (0x52800000, false, Form::Move),
            Movz_64_movewide => (0x52800000, true, Form::Move),
            Movk_32_movewide => (0x72800000, false, Form::Move),
            Movk_64_movewide => (0x72800000, true, Form::Move),
            Fmov_s_w => (0x1e270000, false, Form::RegPair(Bank::V, Bank::W)),
            Fmov_w_s => (0x1e260000, false, Form::RegPair(Bank::W, Bank::V)),
            Fmov_d_x => (0x9e670000, false, Form::RegPair(Bank::V, Bank::X)),
            Fmov_x_d => (0x9e660000, false, Form::RegPair(Bank::X, Bank::V)),
            Fadd_scalar => (0x1e202800, false, Form::FloatScalar),
            Fsub_scalar => (0x1e203800, false, Form::FloatScalar),
            Fmul_scalar => (0x1e200800, false, Form::FloatScalar),
            Fdiv_scalar => (0x1e201800, false, Form::FloatScalar),
            Mov_v_d1_x => (0x4e181c00, false, Form::RegPair(Bank::V, Bank::X)),
            Sbfm_32m_bitfield => (0x13000000, false, Form::Bitfield),
            Sbfm_64m_bitfield => (0x13400000, true, Form::Bitfield),
            Ubfm_32m_bitfield => (0x53000000, false, Form::Bitfield),
            Ubfm_64m_bitfield => (0x53400000, true, Form::Bitfield),
            Add_64_addsub_sxtw => (0x8b20c000, false, Form::ExtendedAdd),
            Lslv_32_dp_2src => (0x1ac02000, false, Form::Three),
            Lslv_64_dp_2src => (0x1ac02000, true, Form::Three),
            Rorv_32_dp_2src => (0x1ac02c00, false, Form::Three),
            Rorv_64_dp_2src => (0x1ac02c00, true, Form::Three),
            Asrv_32_dp_2src => (0x1ac02800, false, Form::Three),
            Asrv_64_dp_2src => (0x1ac02800, true, Form::Three),
            Lsrv_32_dp_2src => (0x1ac02400, false, Form::Three),
            Lsrv_64_dp_2src => (0x1ac02400, true, Form::Three),
            Csel_32_condsel => (0x1a800000, false, Form::Select),
            Csel_64_condsel => (0x1a800000, true, Form::Select),
            Csinv_32_condsel => (0x5a800000, false, Form::Select),
            Ldr_32_ldst_pos => (0xb9400000, false, Form::Memory(4)),
            Ldr_64_ldst_pos => (0x79400000, true, Form::Memory(8)),
            Str_32_ldst_pos => (0xb9000000, false, Form::Memory(4)),
            Str_64_ldst_pos => (0x79000000, true, Form::Memory(8)),
            Ldrb_32_ldst_pos => (0x39400000, false, Form::Memory(1)),
            Strb_32_ldst_pos => (0x39000000, false, Form::Memory(1)),
            Ldrh_32_ldst_pos => (0x79400000, false, Form::Memory(2)),
            Strh_32_ldst_pos => (0x79000000, false, Form::Memory(2)),
            Ldr_q_ldst_pos => (0x3dc00000, false, Form::Memory(16)),
            Str_q_ldst_pos => (0x3d800000, false, Form::Memory(16)),
            Ldr_s_ldst_pos => (0xbd400000, false, Form::FloatMemory(4)),
            Str_s_ldst_pos => (0xbd000000, false, Form::FloatMemory(4)),
            Ldr_d_ldst_pos => (0xfd400000, false, Form::FloatMemory(8)),
            Str_d_ldst_pos => (0xfd000000, false, Form::FloatMemory(8)),
            Nop_HI_hints => (0xd503201f, false, Form::None),
            B_only_branch_imm => (0x14000000, false, Form::Branch(BranchKind::Direct)),
            Bl_only_branch_imm => (0x94000000, false, Form::Branch(BranchKind::Direct)),
            B_only_condbranch => (0x54000000, false, Form::Branch(BranchKind::Condition)),
            Cbz_32_compbranch => (0x34000000, false, Form::Branch(BranchKind::Compare)),
            Cbz_64_compbranch => (0x34000000, true, Form::Branch(BranchKind::Compare)),
            Cbnz_32_compbranch => (0x35000000, false, Form::Branch(BranchKind::Compare)),
            Cbnz_64_compbranch => (0x35000000, true, Form::Branch(BranchKind::Compare)),
            Br_64_branch_reg => (0xd61f0000, false, Form::Indirect),
            Blr_64_branch_reg => (0xd63f0000, false, Form::Indirect),
            Ret_64r_branch_reg => (0xd65f0000, false, Form::Indirect),
            Add_asimdsame_only => (0x0e208400, false, Form::Neon(false)),
            Sub_asimdsame_only => (0x2e208400, false, Form::Neon(false)),
            Mul_asimdsame_only => (0x0e209c00, false, Form::Neon(true)),
            And_asimdsame_only => (0x0e201c00, false, Form::NeonBits),
            Orr_asimdsame_only => (0x0ea01c00, false, Form::NeonBits),
            Eor_asimdsame_only => (0x2e201c00, false, Form::NeonBits),
            Fadd_asimdsame => (0x0e20d400, false, Form::FloatNeon),
            Fsub_asimdsame => (0x0ea0d400, false, Form::FloatNeon),
            Fmul_asimdsame => (0x2e20dc00, false, Form::FloatNeon),
            Fdiv_asimdsame => (0x2e20fc00, false, Form::FloatNeon),
            Dup_asimdgen => (0x4e000c00, false, Form::Dup(false)),
            Saddl_asimddiff_L => (0x0e200000, false, Form::Widen(suffix)),
            Ptrue_p_s_ => (0x2518e000, false, Form::Ptrue),
            Add_z_zz_ => (0x04200000, false, Form::SveArithmetic),
            Sub_z_zz_ => (0x04200400, false, Form::SveArithmetic),
            Mul_z_zz_ => (0x04206000, false, Form::SveArithmetic),
            Mul_z_p_zz_ => (0x04100000, false, Form::SveDestructive),
            Fadd_z_zz_ => (0x65000000, false, Form::SveArithmetic),
            Fsub_z_zz_ => (0x65000400, false, Form::SveArithmetic),
            Fmul_z_zz_ => (0x65000800, false, Form::SveArithmetic),
            Fdiv_z_p_zz_ => (0x650d8000, false, Form::SveDestructive),
            Fdivr_z_p_zz_ => (0x650c8000, false, Form::SveDestructive),
            Dup_z_r => (0x05203800, false, Form::Dup(true)),
            Cmplt_p_p_zi0 => (0x25002000, false, Form::SveCompareZero),
            Sel_z_p_zz => (0x0520c000, false, Form::SveSelect),
            Ld1w_z_p_bz_sxtw => (0x85604000, false, Form::SveIndexed(A::S, PredQual::Zero)),
            Ld1d_z_p_bz_lsl => (0xc5e0c000, false, Form::SveIndexed(A::D, PredQual::Zero)),
            And_z_zz_ => (0x04203000, false, Form::SveBits),
            Orr_z_zz_ => (0x04603000, false, Form::SveBits),
            Eor_z_zz_ => (0x04a03000, false, Form::SveBits),
            Ld1b_z_p_bi_u8 => (0xa400a000, false, Form::SveMemory(A::B, PredQual::Zero)),
            Ld1h_z_p_bi_u16 => (0xa4a0a000, false, Form::SveMemory(A::H, PredQual::Zero)),
            Ld1w_z_p_bi_u32 => (0xa540a000, false, Form::SveMemory(A::S, PredQual::Zero)),
            Ld1d_z_p_bi_u64 => (0xa5e0a000, false, Form::SveMemory(A::D, PredQual::Zero)),
            St1b_z_p_bi_ => (0xe400e000, false, Form::SveMemory(A::B, PredQual::None)),
            St1h_z_p_bi_ => (0xe4a0e000, false, Form::SveMemory(A::H, PredQual::None)),
            St1w_z_p_bi_ => (0xe540e000, false, Form::SveMemory(A::S, PredQual::None)),
            St1d_z_p_bi_ => (0xe5e0e000, false, Form::SveMemory(A::D, PredQual::None)),
            Eor3_z_zzz_ => (0x04203800, false, Form::SveTiedTernary),
            Bsl_z_zzz_ => (0x04203c00, false, Form::SveTiedTernary),
        };
        let (fields, branch) = form.fields(operands, wide)?;
        Ok((base | (u32::from(wide) << 31) | fields, branch))
    }
}

#[derive(Clone, Copy)]
enum BranchKind {
    Direct,
    Condition,
    Compare,
}

enum Form {
    None,
    AddImm(bool),
    Shift(bool),
    LogicalImmediate,
    VectorShiftImmediate { sve: bool, left: bool },
    Three,
    Multiply,
    Move,
    RegPair(Bank, Bank),
    ArrangedPair(A, A),
    ExtendedAdd,
    FloatScalar,
    FloatScalarUnary,
    FloatImmediate,
    FloatNeonImmediate,
    NeonIntImmediate,
    SveFloatImmediate,
    SveDupMask,
    SveSubImmediate,
    SveUnsignedCompareImmediate,
    SveHistcnt,
    SveMaskZero,
    FloatSelect,
    FloatFma,
    SveFma,
    SveCompact,
    FloatInt(bool),
    FloatNeonUnary,
    SveConvert,
    SveFloatWidth(A, A),
    SveExtend,
    SveUnary(u32),
    FloatCompare,
    NeonUnary(u32),
    NeonPairwiseLong,
    NeonExtend(bool),
    NeonNarrow(bool),
    NeonSign,
    NeonSum(bool),
    SveSum,
    SveFadda,
    SvePredBits,
    NeonSingle,
    FloatNeon,
    Dup(bool),
    SveCompareZero,
    SveCompare(bool),
    SveMaskValue,
    SveIndex,
    SveSelect,
    SveIndexed(A, PredQual),
    Bitfield,
    Select,
    Memory(i64),
    FloatMemory(i64),
    Branch(BranchKind),
    Indirect,
    Neon(bool),
    NeonBits,
    NeonTable,
    Widen(u8),
    Ptrue,
    PredFalse,
    PredPermute,
    SveArithmetic,
    SveDestructive,
    SveDestructiveWide,
    SveMultiplyAccumulate,
    PredBreak,
    PredTest,
    PredWhile,
    PredCount,
    PredNext,
    SveGprLane(bool),
    NeonGprLane(bool),
    SveSplice,
    NeonExt,
    SveExt,
    SveBits,
    SveMemory(A, PredQual),
    SveTiedTernary,
}

fn reg(o: O, bank: Bank, sp: bool, i: usize) -> Result<u32, EncodeError> {
    match o {
        O::Register(r) if r.bank == bank && (r.number != 31 || r.sp == sp) => Ok(r.number.into()),
        _ => Err(EncodeError::Operand(i)),
    }
}

fn vector(o: O, bank: Bank, i: usize) -> Result<(u32, A), EncodeError> {
    match o {
        O::Vector(r, a) if r.bank == bank => Ok((r.number.into(), a)),
        _ => Err(EncodeError::Operand(i)),
    }
}

fn arranged(o: O, bank: Bank, a: A, i: usize) -> Result<u32, EncodeError> {
    let (r, actual) = vector(o, bank, i)?;
    if actual != a {
        return Err(EncodeError::Operand(i));
    }
    Ok(r)
}

fn predicate(o: O, qual: PredQual, max: u8, i: usize) -> Result<u32, EncodeError> {
    match o {
        O::Predicate(p, actual) if actual == qual && p.bank == Bank::P && p.number <= max => {
            Ok(p.number.into())
        }
        _ => Err(EncodeError::Operand(i)),
    }
}

fn immediate(o: O, i: usize) -> Result<(u64, u8), EncodeError> {
    match o {
        O::Immediate(v) => Ok((v, 0)),
        O::ShiftedImmediate(v, n) => Ok((v, n)),
        _ => Err(EncodeError::Operand(i)),
    }
}

fn size(a: A) -> Option<u32> {
    match a {
        A::B => Some(0),
        A::H => Some(1),
        A::S => Some(2),
        A::D => Some(3),
        _ => None,
    }
}

// FMOV/FCPY operands use the architectural imm8 encoding.
fn float_immediate(o: O) -> Result<u32, EncodeError> {
    match o {
        O::Immediate(bits @ 0..=255) => Ok(bits as u32),
        _ => Err(EncodeError::Operand(1)),
    }
}

fn logical_immediate(value: u64, width: u32) -> Option<u32> {
    for size in [2, 4, 8, 16, 32, 64] {
        if size > width {
            break;
        }
        let mask = u64::MAX >> (64 - size);
        let element = value & mask;
        let ones = element.count_ones();
        let repeated = (u64::MAX >> (64 - width)) / mask * element;
        if value != repeated || ones == 0 || ones == size {
            continue;
        }
        // A logical immediate is a rotated run of set bits, repeated to the register width.
        let run = u64::MAX >> (64 - ones);
        for rotate in 0..size {
            let rotated = (run >> rotate) | (run.wrapping_shl(size - rotate) & mask);
            if rotated == element {
                let imms = (!(size * 2 - 1) & 63) | (ones - 1);
                return Some(u32::from(size == 64) << 22 | rotate << 16 | imms << 10);
            }
        }
    }
    None
}

impl Form {
    fn fields(self, o: &[O], wide: bool) -> Result<Encoded, EncodeError> {
        let expected = match self {
            Self::None => 0,
            Self::Indirect
            | Self::Branch(BranchKind::Direct)
            | Self::SveIndex
            | Self::PredFalse => 1,
            Self::Move
            | Self::RegPair(..)
            | Self::ArrangedPair(..)
            | Self::Dup(_)
            | Self::Memory(_)
            | Self::FloatMemory(_)
            | Self::Branch(_)
            | Self::Ptrue
            | Self::FloatCompare
            | Self::NeonUnary(_)
            | Self::NeonPairwiseLong
            | Self::NeonExtend(_)
            | Self::NeonNarrow(_)
            | Self::SveExtend
            | Self::NeonSign
            | Self::NeonSum(_)
            | Self::FloatInt(_)
            | Self::FloatNeonUnary
            | Self::FloatScalarUnary
            | Self::FloatImmediate
            | Self::FloatNeonImmediate
            | Self::NeonIntImmediate
            | Self::SveFloatImmediate
            | Self::SveDupMask
            | Self::SveSubImmediate
            | Self::SveMaskZero
            | Self::PredTest
            | Self::SveMaskValue => 2,
            Self::Multiply
            | Self::SveTiedTernary
            | Self::Bitfield
            | Self::Select
            | Self::SveSelect
            | Self::FloatSelect
            | Self::FloatFma
            | Self::SveFma
            | Self::SveCompare(_)
            | Self::SveUnsignedCompareImmediate
            | Self::SveHistcnt
            | Self::SveIndexed(..)
            | Self::SveDestructive
            | Self::SveDestructiveWide
            | Self::SveMultiplyAccumulate
            | Self::SveSplice
            | Self::NeonExt
            | Self::SveFadda
            | Self::SvePredBits
            | Self::SveExt => 4,
            _ => 3,
        };
        if o.len() != expected {
            return Err(EncodeError::OperandCount {
                expected,
                got: o.len(),
            });
        }
        let gpr = if wide { Bank::X } else { Bank::W };
        let width: u8 = if wide { 64 } else { 32 };
        let r = |i| reg(o[i], gpr, false, i);
        let three = || -> Result<u32, EncodeError> { Ok(r(0)? | r(1)? << 5 | r(2)? << 16) };
        let bits = match self {
            Self::None => 0,
            Self::LogicalImmediate => {
                let O::Immediate(value) = o[2] else {
                    return Err(EncodeError::Operand(2));
                };
                r(0)?
                    | r(1)? << 5
                    | logical_immediate(value, width.into()).ok_or(EncodeError::Operand(2))?
            }
            Self::VectorShiftImmediate { sve, left } => {
                let bank = if sve { Bank::Z } else { Bank::V };
                let (d, arr) = vector(o[0], bank, 0)?;
                let bits = match arr {
                    A::B | A::B16 => 8,
                    A::H | A::H8 => 16,
                    A::S | A::S4 => 32,
                    A::D | A::D2 => 64,
                    _ => return Err(EncodeError::Operand(0)),
                };
                if size(arr).is_some() != sve {
                    return Err(EncodeError::Operand(0));
                }
                let O::Immediate(n) = o[2] else {
                    return Err(EncodeError::Operand(2));
                };
                if (left && n >= bits) || (!left && (n == 0 || n > bits)) {
                    return Err(EncodeError::Operand(2));
                }
                let field = if left { bits + n } else { bits * 2 - n } as u32;
                let field = if sve {
                    (field & 31) | ((field & 96) << 1)
                } else {
                    field
                };
                d | arranged(o[1], bank, arr, 1)? << 5 | field << 16
            }
            Self::AddImm(flags) => {
                let (v, shift) = immediate(o[2], 2)?;
                if v > 4095 || !matches!(shift, 0 | 12) {
                    return Err(EncodeError::Operand(2));
                }
                reg(o[0], gpr, !flags, 0)?
                    | reg(o[1], gpr, true, 1)? << 5
                    | (v as u32) << 10
                    | u32::from(shift == 12) << 22
            }
            Self::Shift(logical) => {
                let (rm, shift, n) = match o[2] {
                    O::ShiftedRegister(rm, shift, n) => (reg(O::reg(rm), gpr, false, 2)?, shift, n),
                    _ => (r(2)?, ShiftKind::Lsl, 0),
                };
                if n >= width || (!logical && shift == ShiftKind::Ror) {
                    return Err(EncodeError::Operand(2));
                }
                r(0)? | r(1)? << 5 | rm << 16 | u32::from(n) << 10 | (shift as u32) << 22
            }
            Self::Three => three()?,
            Self::Select => {
                let O::Condition(c) = o[3] else {
                    return Err(EncodeError::Operand(3));
                };
                three()? | (c as u32) << 12
            }
            Self::Multiply => three()? | r(3)? << 10,
            Self::Move => {
                let (v, shift) = immediate(o[1], 1)?;
                if v > 65535 || shift >= width || shift % 16 != 0 {
                    return Err(EncodeError::Operand(1));
                }
                r(0)? | (v as u32) << 5 | u32::from(shift / 16) << 21
            }
            Self::RegPair(a, b) => reg(o[0], a, false, 0)? | reg(o[1], b, false, 1)? << 5,
            Self::ExtendedAdd => {
                let O::ShiftedRegister(index, ShiftKind::Lsl, shift @ 0..=4) = o[2] else {
                    return Err(EncodeError::Operand(2));
                };
                reg(o[0], Bank::X, true, 0)?
                    | reg(o[1], Bank::X, true, 1)? << 5
                    | u32::from(shift) << 10
                    | reg(O::reg(index), Bank::W, false, 2)? << 16
            }
            Self::FloatScalar | Self::FloatSelect | Self::FloatNeon | Self::FloatFma => {
                let (vd, a) = vector(o[0], Bank::V, 0)?;
                let bits = match (!matches!(self, Self::FloatNeon), a) {
                    (true, A::S) | (false, A::S2) => 0,
                    (true, A::D) => 1 << 22,
                    (false, A::S4) => 1 << 30,
                    (false, A::D2) => (1 << 30) | (1 << 22),
                    _ => return Err(EncodeError::Operand(0)),
                };
                let cond = if matches!(self, Self::FloatSelect) {
                    let O::Condition(cond) = o[3] else {
                        return Err(EncodeError::Operand(3));
                    };
                    (cond as u32) << 12
                } else if matches!(self, Self::FloatFma) {
                    arranged(o[3], Bank::V, a, 3)? << 10
                } else {
                    0
                };
                bits | vd
                    | cond
                    | arranged(o[1], Bank::V, a, 1)? << 5
                    | arranged(o[2], Bank::V, a, 2)? << 16
            }
            Self::Dup(sve) => {
                let (d, a) = vector(o[0], if sve { Bank::Z } else { Bank::V }, 0)?;
                let s = if sve {
                    size(a)
                } else {
                    match a {
                        A::B16 => Some(0),
                        A::H8 => Some(1),
                        A::S4 => Some(2),
                        A::D2 => Some(3),
                        _ => None,
                    }
                }
                .ok_or(EncodeError::Operand(0))?;
                let bank = if s == 3 { Bank::X } else { Bank::W };
                d | reg(o[1], bank, false, 1)? << 5 | if sve { s << 22 } else { (1 << s) << 16 }
            }
            Self::FloatCompare => {
                let (a, arr) = vector(o[0], Bank::V, 0)?;
                let s = match arr {
                    A::S => 0,
                    A::D => 1 << 22,
                    _ => return Err(EncodeError::Operand(0)),
                };
                s | a << 5 | arranged(o[1], Bank::V, arr, 1)? << 16
            }
            Self::FloatInt(to_float) => {
                let (v, g) = if to_float { (0, 1) } else { (1, 0) };
                let (vd, arr) = vector(o[v], Bank::V, v)?;
                let f = match arr {
                    A::S => 0,
                    A::D => 1 << 22,
                    _ => return Err(EncodeError::Operand(v)),
                };
                let O::Register(r) = o[g] else {
                    return Err(EncodeError::Operand(g));
                };
                if !matches!(r.bank, Bank::W | Bank::X) {
                    return Err(EncodeError::Operand(g));
                }
                let rs = reg(o[g], r.bank, false, g)?;
                f | (u32::from(r.bank == Bank::X) << 31)
                    | if to_float { vd | rs << 5 } else { rs | vd << 5 }
            }
            Self::FloatNeonUnary => {
                let (d, arr) = vector(o[0], Bank::V, 0)?;
                let f = match arr {
                    A::S2 => 0,
                    A::S4 => 1 << 30,
                    A::D2 => (1 << 30) | (1 << 22),
                    _ => return Err(EncodeError::Operand(0)),
                };
                d | f | arranged(o[1], Bank::V, arr, 1)? << 5
            }
            Self::FloatScalarUnary | Self::FloatImmediate => {
                let (d, arr) = vector(o[0], Bank::V, 0)?;
                let f = match arr {
                    A::S => 0,
                    A::D => 1 << 22,
                    _ => return Err(EncodeError::Operand(0)),
                };
                let rhs = if matches!(self, Self::FloatImmediate) {
                    float_immediate(o[1])? << 13
                } else {
                    arranged(o[1], Bank::V, arr, 1)? << 5
                };
                d | f | rhs
            }
            Self::FloatNeonImmediate => {
                let (d, arr) = vector(o[0], Bank::V, 0)?;
                let f = match arr {
                    A::S4 => 0,
                    A::D2 => 1 << 29,
                    _ => return Err(EncodeError::Operand(0)),
                };
                let imm = float_immediate(o[1])?;
                d | f | (imm & 31) << 5 | (imm >> 5) << 16
            }
            Self::NeonIntImmediate => {
                let (d, arr) = vector(o[0], Bank::V, 0)?;
                let cmode = match arr {
                    A::B16 => 0xe000,
                    A::H8 => 0x8000,
                    A::S4 => 0,
                    _ => return Err(EncodeError::Operand(0)),
                };
                let O::Immediate(imm @ 0..=255) = o[1] else {
                    return Err(EncodeError::Operand(1));
                };
                let imm = imm as u32;
                d | cmode | (imm & 31) << 5 | (imm >> 5) << 16
            }
            Self::SveDupMask => {
                let (d, arr) = vector(o[0], Bank::Z, 0)?;
                let width = 8 << size(arr).ok_or(EncodeError::Operand(0))?;
                let O::Immediate(value) = o[1] else {
                    return Err(EncodeError::Operand(1));
                };
                d | logical_immediate(value, width).ok_or(EncodeError::Operand(1))? >> 5
            }
            Self::SveSubImmediate => {
                let (d, arr) = vector(o[0], Bank::Z, 0)?;
                let s = size(arr).ok_or(EncodeError::Operand(0))?;
                let O::Immediate(imm @ 0..=255) = o[1] else {
                    return Err(EncodeError::Operand(1));
                };
                d | s << 22 | (imm as u32) << 5
            }
            Self::SveUnsignedCompareImmediate => {
                let (d, arr) = vector(o[0], Bank::P, 0)?;
                let s = size(arr).ok_or(EncodeError::Operand(0))?;
                let O::Immediate(imm @ 0..=127) = o[3] else {
                    return Err(EncodeError::Operand(3));
                };
                d | s << 22
                    | predicate(o[1], PredQual::Zero, 7, 1)? << 10
                    | arranged(o[2], Bank::Z, arr, 2)? << 5
                    | (imm as u32) << 14
            }
            Self::SveHistcnt => {
                let (d, arr) = vector(o[0], Bank::Z, 0)?;
                let s = match arr {
                    A::S => 0,
                    A::D => 1 << 22,
                    _ => return Err(EncodeError::Operand(0)),
                };
                d | s
                    | predicate(o[1], PredQual::Zero, 7, 1)? << 10
                    | arranged(o[2], Bank::Z, arr, 2)? << 5
                    | arranged(o[3], Bank::Z, arr, 3)? << 16
            }
            Self::SveMaskZero => {
                let (d, arr) = vector(o[0], Bank::Z, 0)?;
                let s = size(arr).ok_or(EncodeError::Operand(0))?;
                d | s << 22 | predicate(o[1], PredQual::Merge, 15, 1)? << 16
            }
            Self::SveFloatImmediate => {
                let (d, arr) = vector(o[0], Bank::Z, 0)?;
                let f = match arr {
                    A::S | A::D => size(arr).unwrap() << 22,
                    _ => return Err(EncodeError::Operand(0)),
                };
                d | f | float_immediate(o[1])? << 5
            }
            Self::SveConvert => {
                let (d, arr) = vector(o[0], Bank::Z, 0)?;
                let f = match arr {
                    A::S => 0,
                    A::D => 0x420000,
                    _ => return Err(EncodeError::Operand(0)),
                };
                d | f
                    | predicate(o[1], PredQual::Merge, 7, 1)? << 10
                    | arranged(o[2], Bank::Z, arr, 2)? << 5
            }
            Self::SveUnary(min) => {
                let (d, arr) = vector(o[0], Bank::Z, 0)?;
                let s = size(arr)
                    .filter(|&s| s >= min)
                    .ok_or(EncodeError::Operand(0))?;
                d | s << 22
                    | predicate(o[1], PredQual::Merge, 7, 1)? << 10
                    | arranged(o[2], Bank::Z, arr, 2)? << 5
            }
            Self::NeonUnary(max) => {
                let (d, a) = vector(o[0], Bank::V, 0)?;
                let s = match a {
                    A::B16 => 0,
                    A::H8 => 1,
                    A::S4 => 2,
                    A::D2 => 3,
                    _ => return Err(EncodeError::Operand(0)),
                };
                if s > max {
                    return Err(EncodeError::Operand(0));
                }
                d | arranged(o[1], Bank::V, a, 1)? << 5 | s << 22
            }
            Self::NeonPairwiseLong => {
                let (d, arr) = vector(o[0], Bank::V, 0)?;
                let (s, src) = match arr {
                    A::H8 => (0, A::B16),
                    A::S4 => (1, A::H8),
                    A::D2 => (2, A::S4),
                    _ => return Err(EncodeError::Operand(0)),
                };
                d | s << 22 | arranged(o[1], Bank::V, src, 1)? << 5
            }
            Self::NeonExtend(high) => {
                let (d, arr) = vector(o[0], Bank::V, 0)?;
                let (bits, lo, hi) = match arr {
                    A::H8 => (8, A::B8, A::B16),
                    A::S4 => (16, A::H4, A::H8),
                    A::D2 => (32, A::S2, A::S4),
                    _ => return Err(EncodeError::Operand(0)),
                };
                d | bits << 16 | arranged(o[1], Bank::V, if high { hi } else { lo }, 1)? << 5
            }
            Self::NeonNarrow(high) => {
                let (d, arr) = vector(o[0], Bank::V, 0)?;
                let (s, src) = match (arr, high) {
                    (A::B8, false) | (A::B16, true) => (0, A::H8),
                    (A::H4, false) | (A::H8, true) => (1, A::S4),
                    (A::S2, false) | (A::S4, true) => (2, A::D2),
                    _ => return Err(EncodeError::Operand(0)),
                };
                d | s << 22 | arranged(o[1], Bank::V, src, 1)? << 5
            }
            Self::ArrangedPair(dst, src) => {
                arranged(o[0], Bank::V, dst, 0)? | arranged(o[1], Bank::V, src, 1)? << 5
            }
            Self::SveExtend => {
                let (d, arr) = vector(o[0], Bank::Z, 0)?;
                let src = match arr {
                    A::H => A::B,
                    A::S => A::H,
                    A::D => A::S,
                    _ => return Err(EncodeError::Operand(0)),
                };
                d | size(arr).unwrap() << 22 | arranged(o[1], Bank::Z, src, 1)? << 5
            }
            Self::SveFloatWidth(dst, src) => {
                arranged(o[0], Bank::Z, dst, 0)?
                    | predicate(o[1], PredQual::Merge, 7, 1)? << 10
                    | arranged(o[2], Bank::Z, src, 2)? << 5
            }
            Self::NeonSign => {
                let (d, a) = vector(o[0], Bank::V, 0)?;
                let bits = match a {
                    A::B16 => 8,
                    A::H8 => 16,
                    A::S4 => 32,
                    A::D2 => 64,
                    _ => return Err(EncodeError::Operand(0)),
                };
                d | arranged(o[1], Bank::V, a, 1)? << 5 | (bits + 1) << 16
            }
            Self::NeonSum(pair) => {
                let (src, a) = vector(o[1], Bank::V, 1)?;
                let s = match (pair, a) {
                    (false, A::B16) | (true, A::D2) => 0,
                    (false, A::H8) => 1,
                    (false, A::S4) => 2,
                    _ => return Err(EncodeError::Operand(1)),
                };
                reg(o[0], Bank::V, false, 0)? | src << 5 | s << 22
            }
            Self::SveSum => {
                let (src, arr) = vector(o[2], Bank::Z, 2)?;
                let size = size(arr).ok_or(EncodeError::Operand(2))?;
                reg(o[0], Bank::V, false, 0)?
                    | predicate(o[1], PredQual::None, 7, 1)? << 10
                    | src << 5
                    | size << 22
            }
            Self::SveFadda => {
                let (d, arr) = vector(o[0], Bank::V, 0)?;
                let s = match arr {
                    A::S => 2,
                    A::D => 3,
                    _ => return Err(EncodeError::Operand(0)),
                };
                if arranged(o[2], Bank::V, arr, 2)? != d {
                    return Err(EncodeError::Operand(2));
                }
                d | s << 22
                    | predicate(o[1], PredQual::None, 7, 1)? << 10
                    | arranged(o[3], Bank::Z, arr, 3)? << 5
            }
            Self::SvePredBits => {
                arranged(o[0], Bank::P, A::B, 0)?
                    | predicate(o[1], PredQual::Zero, 15, 1)? << 10
                    | arranged(o[2], Bank::P, A::B, 2)? << 5
                    | arranged(o[3], Bank::P, A::B, 3)? << 16
            }
            Self::NeonSingle => {
                let O::Immediate(lane @ 0..=3) = o[2] else {
                    return Err(EncodeError::Operand(2));
                };
                arranged(o[0], Bank::V, A::S, 0)?
                    | arranged(o[1], Bank::V, A::S4, 1)? << 5
                    | (lane as u32) << 19
            }
            Self::NeonGprLane(insert) => {
                let (v, g) = if insert { (0, 1) } else { (1, 0) };
                let (vr, arr) = vector(o[v], Bank::V, v)?;
                let s = match arr {
                    A::B16 => 0,
                    A::H8 => 1,
                    A::S4 => 2,
                    A::D2 => 3,
                    _ => return Err(EncodeError::Operand(v)),
                };
                let O::Immediate(lane) = o[2] else {
                    return Err(EncodeError::Operand(2));
                };
                if lane >= 16 >> s {
                    return Err(EncodeError::Operand(2));
                }
                let gr = reg(o[g], if s == 3 { Bank::X } else { Bank::W }, false, g)?;
                let imm5 = ((lane as u32) * 2 + 1) << s;
                imm5 << 16
                    | if insert {
                        vr | gr << 5
                    } else {
                        gr | vr << 5 | u32::from(s == 3) << 30
                    }
            }
            Self::PredBreak => {
                arranged(o[0], Bank::P, A::B, 0)?
                    | predicate(o[1], PredQual::Zero, 15, 1)? << 10
                    | arranged(o[2], Bank::P, A::B, 2)? << 5
            }
            Self::PredTest => {
                predicate(o[0], PredQual::None, 15, 0)? << 10
                    | arranged(o[1], Bank::P, A::B, 1)? << 5
            }
            Self::PredWhile => {
                let (d, a) = vector(o[0], Bank::P, 0)?;
                d | size(a).ok_or(EncodeError::Operand(0))? << 22
                    | reg(o[1], Bank::X, false, 1)? << 5
                    | reg(o[2], Bank::X, false, 2)? << 16
            }
            Self::PredCount => {
                let (p, a) = vector(o[2], Bank::P, 2)?;
                reg(o[0], Bank::X, false, 0)?
                    | predicate(o[1], PredQual::None, 15, 1)? << 10
                    | p << 5
                    | size(a).ok_or(EncodeError::Operand(2))? << 22
            }
            Self::PredNext => {
                let (d, arr) = vector(o[0], Bank::P, 0)?;
                let s = size(arr).ok_or(EncodeError::Operand(0))?;
                if arranged(o[2], Bank::P, arr, 2)? != d {
                    return Err(EncodeError::Operand(2));
                }
                d | s << 22 | predicate(o[1], PredQual::None, 15, 1)? << 5
            }
            Self::SveGprLane(insert) => {
                let (zi, gi) = if insert { (0, 2) } else { (2, 0) };
                let (v, arr) = vector(o[zi], Bank::Z, zi)?;
                let s = size(arr).ok_or(EncodeError::Operand(zi))?;
                let g = reg(o[gi], if s == 3 { Bank::X } else { Bank::W }, false, gi)?;
                let qual = if insert {
                    PredQual::Merge
                } else {
                    PredQual::None
                };
                s << 22
                    | predicate(o[1], qual, 7, 1)? << 10
                    | if insert { v | g << 5 } else { g | v << 5 }
            }
            Self::SveFma | Self::SveCompact | Self::SveMultiplyAccumulate => {
                let (dst, a) = vector(o[0], Bank::Z, 0)?;
                let s = size(a)
                    .filter(|&s| s >= 2 || matches!(self, Self::SveMultiplyAccumulate))
                    .ok_or(EncodeError::Operand(0))?;
                let (qual, rhs) = if !matches!(self, Self::SveCompact) {
                    (PredQual::Merge, arranged(o[3], Bank::Z, a, 3)? << 16)
                } else {
                    (PredQual::None, 0)
                };
                dst | predicate(o[1], qual, 7, 1)? << 10
                    | arranged(o[2], Bank::Z, a, 2)? << 5
                    | rhs
                    | s << 22
            }
            Self::SveIndex => {
                let (d, a) = vector(o[0], Bank::Z, 0)?;
                d | size(a).ok_or(EncodeError::Operand(0))? << 22
            }
            Self::SveMaskValue => {
                let (d, a) = vector(o[0], Bank::Z, 0)?;
                d | predicate(o[1], PredQual::Zero, 15, 1)? << 16
                    | size(a).ok_or(EncodeError::Operand(0))? << 22
            }
            Self::SveCompareZero | Self::SveCompare(_) => {
                let (d, a) = vector(o[0], Bank::P, 0)?;
                let s = size(a).ok_or(EncodeError::Operand(0))?;
                if s == 0 && matches!(self, Self::SveCompare(true)) {
                    return Err(EncodeError::Operand(0));
                }
                let rhs = if matches!(self, Self::SveCompare(_)) {
                    arranged(o[3], Bank::Z, a, 3)? << 16
                } else {
                    0
                };
                d | predicate(o[1], PredQual::Zero, 7, 1)? << 10
                    | arranged(o[2], Bank::Z, a, 2)? << 5
                    | rhs
                    | s << 22
            }
            Self::SveSelect => {
                let (d, a) = vector(o[0], Bank::Z, 0)?;
                let s = size(a).ok_or(EncodeError::Operand(0))?;
                d | predicate(o[1], PredQual::None, 15, 1)? << 10
                    | arranged(o[2], Bank::Z, a, 2)? << 5
                    | arranged(o[3], Bank::Z, a, 3)? << 16
                    | s << 22
            }
            Self::SveIndexed(a, qual) => {
                let O::Memory(base, 0) = o[2] else {
                    return Err(EncodeError::Operand(2));
                };
                arranged(o[0], Bank::Z, a, 0)?
                    | predicate(o[1], qual, 7, 1)? << 10
                    | reg(O::reg(base), Bank::X, true, 2)? << 5
                    | arranged(o[3], Bank::Z, a, 3)? << 16
            }
            Self::Bitfield => {
                let (a, sa) = immediate(o[2], 2)?;
                let (b, sb) = immediate(o[3], 3)?;
                if a >= width.into() || sa != 0 {
                    return Err(EncodeError::Operand(2));
                }
                if b >= width.into() || sb != 0 {
                    return Err(EncodeError::Operand(3));
                }
                r(0)? | r(1)? << 5 | (a as u32) << 16 | (b as u32) << 10
            }
            Self::Memory(scale) | Self::FloatMemory(scale) => {
                let O::Memory(base, offset) = o[1] else {
                    return Err(EncodeError::Operand(1));
                };
                if offset < 0 || offset % scale != 0 || offset / scale > 4095 {
                    return Err(EncodeError::Operand(1));
                }
                let rt = if scale == 16 || matches!(self, Self::FloatMemory(_)) {
                    reg(o[0], Bank::V, false, 0)?
                } else {
                    r(0)?
                };
                rt | reg(O::reg(base), Bank::X, true, 1)? << 5 | ((offset / scale) as u32) << 10
            }
            Self::Indirect => reg(o[0], Bank::X, false, 0)? << 5,
            Self::Branch(kind) => {
                let direct = matches!(kind, BranchKind::Direct);
                let i = usize::from(!direct);
                let O::Label(target) = o[i] else {
                    return Err(EncodeError::Operand(i));
                };
                let fields = match (kind, o[0]) {
                    (BranchKind::Direct, _) => 0,
                    (BranchKind::Condition, O::Condition(c)) => c as u32,
                    (BranchKind::Compare, _) => r(0)?,
                    _ => return Err(EncodeError::Operand(0)),
                };
                let (bits, shift) = if direct { (26, 0) } else { (19, 5) };
                return Ok((
                    fields,
                    Some(Branch {
                        target,
                        bits,
                        shift,
                    }),
                ));
            }
            Self::Neon(_) | Self::NeonBits => {
                let (vd, a) = vector(o[0], Bank::V, 0)?;
                let (q, s) = match a {
                    A::B8 => (0, 0),
                    A::B16 => (1, 0),
                    A::H4 => (0, 1),
                    A::H8 => (1, 1),
                    A::S2 => (0, 2),
                    A::S4 => (1, 2),
                    A::D2 => (1, 3),
                    _ => return Err(EncodeError::Operand(0)),
                };
                if (matches!(self, Self::NeonBits) && s != 0)
                    || (matches!(self, Self::Neon(true)) && s == 3)
                {
                    return Err(EncodeError::Operand(0));
                }
                vd | arranged(o[1], Bank::V, a, 1)? << 5
                    | arranged(o[2], Bank::V, a, 2)? << 16
                    | q << 30
                    | s << 22
            }
            Self::NeonTable => {
                let (d, arr) = vector(o[0], Bank::V, 0)?;
                let q = match arr {
                    A::B8 => 0,
                    A::B16 => 1 << 30,
                    _ => return Err(EncodeError::Operand(0)),
                };
                d | q
                    | arranged(o[1], Bank::V, A::B16, 1)? << 5
                    | arranged(o[2], Bank::V, arr, 2)? << 16
            }
            Self::Widen(high) => {
                let (vd, dst) = vector(o[0], Bank::V, 0)?;
                let (s, low, upper) = match dst {
                    A::H8 => (0, A::B8, A::B16),
                    A::S4 => (1, A::H4, A::H8),
                    A::D2 => (2, A::S2, A::S4),
                    _ => return Err(EncodeError::Operand(0)),
                };
                let src = if high == 0 { low } else { upper };
                vd | arranged(o[1], Bank::V, src, 1)? << 5
                    | arranged(o[2], Bank::V, src, 2)? << 16
                    | u32::from(high) << 30
                    | s << 22
            }
            Self::Ptrue => {
                let (p, a) = vector(o[0], Bank::P, 0)?;
                let s = size(a).ok_or(EncodeError::Operand(0))?;
                let O::Named(n @ (0..=13 | 29..=31)) = o[1] else {
                    return Err(EncodeError::Operand(1));
                };
                p | n << 5 | s << 22
            }
            Self::PredFalse => arranged(o[0], Bank::P, A::B, 0)?,
            Self::PredPermute => {
                let (d, arr) = vector(o[0], Bank::P, 0)?;
                let s = size(arr).ok_or(EncodeError::Operand(0))?;
                d | s << 22
                    | arranged(o[1], Bank::P, arr, 1)? << 5
                    | arranged(o[2], Bank::P, arr, 2)? << 16
            }
            Self::SveArithmetic | Self::SveBits => {
                let (d, a) = vector(o[0], Bank::Z, 0)?;
                let s = if matches!(self, Self::SveBits) {
                    if a != A::D {
                        return Err(EncodeError::Operand(0));
                    }
                    0
                } else {
                    size(a).ok_or(EncodeError::Operand(0))?
                };
                d | arranged(o[1], Bank::Z, a, 1)? << 5
                    | arranged(o[2], Bank::Z, a, 2)? << 16
                    | s << 22
            }
            Self::NeonExt | Self::SveExt => {
                let sve = matches!(self, Self::SveExt);
                let vec = |i| {
                    if sve {
                        arranged(o[i], Bank::Z, A::B, i)
                    } else {
                        arranged(o[i], Bank::V, A::B16, i)
                    }
                };
                let O::Immediate(imm) = o[3] else {
                    return Err(EncodeError::Operand(3));
                };
                if imm > if sve { 255 } else { 15 } {
                    return Err(EncodeError::Operand(3));
                }
                let imm = imm as u32;
                let (d, a, b) = (vec(0)?, vec(1)?, vec(2)?);
                if sve {
                    if d != a {
                        return Err(EncodeError::Operand(1));
                    }
                    d | b << 5 | (imm & 7) << 10 | (imm >> 3) << 16
                } else {
                    d | a << 5 | b << 16 | imm << 11
                }
            }
            Self::SveDestructive | Self::SveDestructiveWide | Self::SveSplice => {
                let (d, a) = vector(o[0], Bank::Z, 0)?;
                let s = size(a)
                    .filter(|&s| s >= 2 || !matches!(self, Self::SveDestructiveWide))
                    .ok_or(EncodeError::Operand(0))?;
                let qualifier = if matches!(self, Self::SveSplice) {
                    PredQual::None
                } else {
                    PredQual::Merge
                };
                let p = predicate(o[1], qualifier, 7, 1)?;
                if arranged(o[2], Bank::Z, a, 2)? != d {
                    return Err(EncodeError::Operand(2));
                }
                d | arranged(o[3], Bank::Z, a, 3)? << 5 | p << 10 | s << 22
            }
            Self::SveMemory(a, qualifier) => {
                let O::Predicate(p, actual) = o[1] else {
                    return Err(EncodeError::Operand(1));
                };
                if actual != qualifier || p.bank != Bank::P || p.number > 7 {
                    return Err(EncodeError::Operand(1));
                }
                let (base, offset) = match o[2] {
                    O::Memory(base, 0) => (base, 0),
                    O::MemoryVl(base, offset @ -8..=7) => (base, offset),
                    _ => return Err(EncodeError::Operand(2)),
                };
                arranged(o[0], Bank::Z, a, 0)?
                    | reg(O::reg(base), Bank::X, true, 2)? << 5
                    | u32::from(p.number) << 10
                    | ((offset as u32) & 15) << 16
            }
            Self::SveTiedTernary => {
                let z = |i| arranged(o[i], Bank::Z, A::D, i);
                let d = z(0)?;
                if z(1)? != d {
                    return Err(EncodeError::Operand(1));
                }
                d | z(2)? << 16 | z(3)? << 5
            }
        };
        Ok((bits, None))
    }
}

#[cfg(test)]
mod tests {
    use super::logical_immediate;
    use std::collections::HashSet;

    /// DecodeBitMasks from the Arm ARM, for an independent reference.
    fn decode(n: u32, immr: u32, imms: u32, width: u32) -> Option<u64> {
        let combined = (n << 6) | (!imms & 0x3f);
        let len = combined.checked_ilog2()?;
        if (1 << len) > width {
            return None;
        }
        let size = 1u32 << len;
        let levels = size - 1;
        let s = imms & levels;
        let r = immr & levels;
        if s == levels {
            return None;
        }
        let mask = u64::MAX >> (64 - size);
        let run = u64::MAX >> (63 - s);
        let element = ((run >> r) | (run << ((size - r) % size))) & mask;
        let element = if r == 0 { run & mask } else { element };
        Some((u64::MAX >> (64 - width)) / mask * element)
    }

    #[test]
    fn logical_immediates_round_trip_through_every_encoding() {
        for width in [32, 64] {
            let mut encodable = HashSet::new();
            for n in 0..=u32::from(width == 64) {
                for immr in 0..64 {
                    for imms in 0..64 {
                        if (width == 32 && immr >= 32) || decode(n, immr, imms, width).is_none() {
                            continue;
                        }
                        let value = decode(n, immr, imms, width).unwrap();
                        encodable.insert(value);
                        let fields = logical_immediate(value, width)
                            .unwrap_or_else(|| panic!("{value:#x} is encodable at {width}"));
                        assert_eq!(
                            decode(
                                fields >> 22 & 1,
                                fields >> 16 & 0x3f,
                                fields >> 10 & 0x3f,
                                width
                            ),
                            Some(value),
                            "{value:#x} at {width} encodes as {fields:#x}"
                        );
                    }
                }
            }
            assert_eq!(encodable.len(), if width == 64 { 5334 } else { 1302 });
            let mut state = 0x2545_f491_4f6c_dd1du64;
            for _ in 0..100_000 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                for value in [state, state & 0xffff_0000, state >> 40, !(state >> 3)] {
                    let value = value & (u64::MAX >> (64 - width));
                    assert_eq!(
                        logical_immediate(value, width).is_some(),
                        encodable.contains(&value),
                        "{value:#x} at {width}"
                    );
                }
            }
        }
    }
}
