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

/// The encoder's instruction table: one row per exact instruction form.
///
/// `Name = base, extension, gpr, form;` gives the encoding with every
/// operand field zero, the extension the form needs (`Base` for none),
/// the width of its general-purpose operands (`w`, `x`, or `_` for a form
/// without them) and the [`Form`] that places the operands. The rows
/// generate [`Code`], [`Code::required_feature`] and the encoding.
macro_rules! codes {
    ($($code:ident = $base:literal, $feature:ident, $gpr:tt, $form:expr;)*) => {
        /// Exact instruction forms supported by the Lancy encoder.
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Code {
            $($code,)*
        }

        impl Code {
            pub const fn required_feature(self) -> Option<Feature> {
                match self {
                    $(Self::$code => codes!(@feature $feature),)*
                }
            }

            fn row(self) -> (u32, bool, Form) {
                use Form::*;
                match self {
                    $(Self::$code => ($base, codes!(@wide $gpr), $form),)*
                }
            }
        }
    };
    (@feature Base) => { None };
    (@feature $feature:ident) => { Some(Feature::$feature) };
    (@wide x) => { true };
    (@wide $gpr:tt) => { false };
}

codes! {
    Pnext_p_p_p =          0x2519c400, Sve,  _, PredNext;
    Orr_p_p_pp =           0x25804000, Sve,  _, SvePredBits;
    Lastb_r_p_z =          0x0521a000, Sve,  _, SveGprLane(false);
    Cpy_z_p_r =            0x0528a000, Sve,  _, SveGprLane(true);
    Fcvt_d_h =             0x1ee2c000, Base, _, ArrangedPair(A::D, A::H);
    Fcvt_h_d =             0x1e63c000, Base, _, ArrangedPair(A::H, A::D);
    Fcvtxn_s_d =           0x7e616800, Neon, _, ArrangedPair(A::S, A::D);
    Fminnm_scalar =        0x1e207800, Base, _, FloatScalar;
    Fmaxnm_scalar =        0x1e206800, Base, _, FloatScalar;
    And_32_log_imm =       0x12000000, Base, w, LogicalImmediate;
    And_64_log_imm =       0x92000000, Base, x, LogicalImmediate;
    Adcs_64 =              0xba000000, Base, x, Three;
    Sbcs_64 =              0xfa000000, Base, x, Three;
    Umulh_64 =             0x9bc07c00, Base, x, Three;
    And_z_zi =             0x05800000, Sve,  _, SveDupMask;
    Trn1_z_zz =            0x05207000, Sve,  _, SveArithmetic;
    Sdiv_z_p_zz =          0x04140000, Sve,  _, SveDestructiveWide;
    Udiv_z_p_zz =          0x04150000, Sve,  _, SveDestructiveWide;
    Mls_z_p_zz =           0x04006000, Sve,  _, SveMultiplyAccumulate;
    Brkb_p_p_p =           0x25904000, Sve,  _, PredBreak;
    Ptest_p_p =            0x2550c000, Sve,  _, PredTest;
    Whilelo_p_x_x =        0x25201c00, Sve,  _, PredWhile;
    Cntp_x_p_p =           0x25208000, Sve,  _, PredCount;
    Umov_lane =            0x0e003c00, Neon, _, NeonGprLane(false);
    Ins_lane =             0x4e001c00, Neon, _, NeonGprLane(true);
    Ld1b_z_p_bz_sxtw =     0x84404000, Sve,  _, SveIndexed(A::S, PredQual::Zero);
    Ld1h_z_p_bz_sxtw =     0x84e04000, Sve,  _, SveIndexed(A::S, PredQual::Zero);
    St1b_z_p_bz_sxtw =     0xe440c000, Sve,  _, SveIndexed(A::S, PredQual::None);
    St1h_z_p_bz_sxtw =     0xe4e0c000, Sve,  _, SveIndexed(A::S, PredQual::None);
    St1w_z_p_bz_sxtw =     0xe560c000, Sve,  _, SveIndexed(A::S, PredQual::None);
    St1d_z_p_bz_lsl =      0xe5a0a000, Sve,  _, SveIndexed(A::D, PredQual::None);
    Adc_64 =               0x9a000000, Base, x, Three;
    Sbc_64 =               0xda000000, Base, x, Three;
    Pfalse_p =             0x2518e400, Sve,  _, PredFalse;
    Trn1_p_pp =            0x05205000, Sve,  _, PredPermute;
    Trn2_z_zz =            0x05207400, Sve,  _, SveArithmetic;
    Trn2_asimd =           0x0e006800, Neon, _, Neon(false);
    Dup_z_zi =             0x05202000, Sve,  _, SveDupLane;
    Ucvtf_half_z_p_z =     0x6553a000, Sve,  _, SveFloatWidth(A::H, A::H);
    Sminv_asimd =          0x4e31a800, Neon, _, NeonSum(false);
    Smaxv_asimd =          0x4e30a800, Neon, _, NeonSum(false);
    Uminv_asimd =          0x6e31a800, Neon, _, NeonSum(false);
    Umaxv_asimd =          0x6e30a800, Neon, _, NeonSum(false);
    Sminv_v_p_z =          0x040a2000, Sve,  _, SveSum;
    Smaxv_v_p_z =          0x04082000, Sve,  _, SveSum;
    Uminv_v_p_z =          0x040b2000, Sve,  _, SveSum;
    Umaxv_v_p_z =          0x04092000, Sve,  _, SveSum;
    Andv_v_p_z =           0x041a2000, Sve,  _, SveSum;
    Orv_v_p_z =            0x04182000, Sve,  _, SveSum;
    Eorv_v_p_z =           0x04192000, Sve,  _, SveSum;
    Fadda_scalar_p_z =     0x65182000, Sve,  _, SveFadda;
    Bic_p_p_pp =           0x25004010, Sve,  _, SvePredBits;
    Fmov_scalar =          0x1e204000, Base, _, FloatScalarUnary;
    Dup_s_lane_asimd =     0x5e040400, Neon, _, NeonSingle;
    Faddp_d_asimd =        0x7e70d800, Neon, _, ArrangedPair(A::D, A::D2);
    Sxtl_asimd =           0x0f00a400, Neon, _, NeonExtend(false);
    Sxtl2_asimd =          0x4f00a400, Neon, _, NeonExtend(true);
    Uxtl_asimd =           0x2f00a400, Neon, _, NeonExtend(false);
    Uxtl2_asimd =          0x6f00a400, Neon, _, NeonExtend(true);
    Xtn_asimd =            0x0e212800, Neon, _, NeonNarrow(false);
    Xtn2_asimd =           0x4e212800, Neon, _, NeonNarrow(true);
    Fcvtl_asimd =          0x0e617800, Neon, _, ArrangedPair(A::D2, A::S2);
    Fcvtl2_asimd =         0x4e617800, Neon, _, ArrangedPair(A::D2, A::S4);
    Fcvtn_asimd =          0x0e616800, Neon, _, ArrangedPair(A::S2, A::D2);
    Fcvtn2_asimd =         0x4e616800, Neon, _, ArrangedPair(A::S4, A::D2);
    Sunpklo_z =            0x05303800, Sve,  _, SveExtend;
    Uunpklo_z =            0x05323800, Sve,  _, SveExtend;
    Zip1_z_zz =            0x05206000, Sve,  _, SveArithmetic;
    Uzp1_z_zz =            0x05206800, Sve,  _, SveArithmetic;
    Uzp1_asimd =           0x0e001800, Neon, _, Neon(false);
    Fcvt_z_d_p_s =         0x65cba000, Sve,  _, SveFloatWidth(A::D, A::S);
    Fcvt_z_s_p_d =         0x65caa000, Sve,  _, SveFloatWidth(A::S, A::D);
    Shl_asimd_imm =        0x4f005400, Neon, _, VectorShiftImmediate { sve: false, left: true };
    Sshr_asimd_imm =       0x4f000400, Neon, _, VectorShiftImmediate { sve: false, left: false };
    Ushr_asimd_imm =       0x6f000400, Neon, _, VectorShiftImmediate { sve: false, left: false };
    Lsl_z_zi =             0x04209c00, Sve,  _, VectorShiftImmediate { sve: true, left: true };
    Lsr_z_zi =             0x04209400, Sve,  _, VectorShiftImmediate { sve: true, left: false };
    Asr_z_zi =             0x04209000, Sve,  _, VectorShiftImmediate { sve: true, left: false };
    Tbl_z_zz =             0x05203000, Sve,  _, SveArithmetic;
    Sub_z_zi =             0x2521c000, Sve,  _, SveSubImmediate;
    Cmphi_p_p_zi =         0x24200010, Sve,  _, SveUnsignedCompareImmediate;
    Cmpls_p_p_zi =         0x24202010, Sve,  _, SveUnsignedCompareImmediate;
    Histcnt_z_p_zz =       0x45a0c000, Sve2, _, SveHistcnt;
    Tbl_asimd =            0x0e000000, Neon, _, NeonTable;
    Tbl2_asimd =           0x0e002000, Neon, _, NeonTable;
    Tbx_asimd =            0x0e001000, Neon, _, NeonTable;
    Ext_asimd =            0x6e000000, Neon, _, NeonExt;
    Ext_z_zzi =            0x05200000, Sve,  _, SveExt;
    Splice_z_p_zz =        0x052c8000, Sve,  _, SveSplice;
    Smin_asimd =           0x0e206c00, Neon, _, Neon(true);
    Smax_asimd =           0x0e206400, Neon, _, Neon(true);
    Umin_asimd =           0x2e206c00, Neon, _, Neon(true);
    Umax_asimd =           0x2e206400, Neon, _, Neon(true);
    Sqadd_asimd =          0x0e200c00, Neon, _, Neon(false);
    Uqadd_asimd =          0x2e200c00, Neon, _, Neon(false);
    Sqsub_asimd =          0x0e202c00, Neon, _, Neon(false);
    Uqsub_asimd =          0x2e202c00, Neon, _, Neon(false);
    Sshl_asimd =           0x0e204400, Neon, _, Neon(false);
    Ushl_asimd =           0x2e204400, Neon, _, Neon(false);
    Bit_asimd =            0x2ea01c00, Neon, _, NeonBits;
    Bif_asimd =            0x2ee01c00, Neon, _, NeonBits;
    Movi_asimd =           0x4f000400, Neon, _, NeonIntImmediate;
    Dupm_z =               0x05c00000, Sve,  _, SveDupMask;
    Smin_z_p_zz =          0x040a0000, Sve,  _, SveDestructive;
    Smax_z_p_zz =          0x04080000, Sve,  _, SveDestructive;
    Umin_z_p_zz =          0x040b0000, Sve,  _, SveDestructive;
    Umax_z_p_zz =          0x04090000, Sve,  _, SveDestructive;
    Sqadd_z_zz =           0x04201000, Sve,  _, SveArithmetic;
    Uqadd_z_zz =           0x04201400, Sve,  _, SveArithmetic;
    Sqsub_z_zz =           0x04201800, Sve,  _, SveArithmetic;
    Uqsub_z_zz =           0x04201c00, Sve,  _, SveArithmetic;
    Lsl_z_p_zz =           0x04138000, Sve,  _, SveDestructive;
    Lsr_z_p_zz =           0x04118000, Sve,  _, SveDestructive;
    Asr_z_p_zz =           0x04108000, Sve,  _, SveDestructive;
    Fneg_scalar =          0x1e214000, Base, _, FloatScalarUnary;
    Fabs_scalar =          0x1e20c000, Base, _, FloatScalarUnary;
    Fsqrt_scalar =         0x1e21c000, Base, _, FloatScalarUnary;
    Frintm_scalar =        0x1e254000, Base, _, FloatScalarUnary;
    Frintp_scalar =        0x1e24c000, Base, _, FloatScalarUnary;
    Frintz_scalar =        0x1e25c000, Base, _, FloatScalarUnary;
    Frinta_scalar =        0x1e264000, Base, _, FloatScalarUnary;
    Frintn_scalar =        0x1e244000, Base, _, FloatScalarUnary;
    Fneg_asimd =           0x2ea0f800, Neon, _, FloatNeonUnary;
    Fabs_asimd =           0x0ea0f800, Neon, _, FloatNeonUnary;
    Fsqrt_asimd =          0x2ea1f800, Neon, _, FloatNeonUnary;
    Frintm_asimd =         0x0e219800, Neon, _, FloatNeonUnary;
    Frintp_asimd =         0x0ea18800, Neon, _, FloatNeonUnary;
    Frintz_asimd =         0x0ea19800, Neon, _, FloatNeonUnary;
    Frinta_asimd =         0x2e218800, Neon, _, FloatNeonUnary;
    Frintn_asimd =         0x0e218800, Neon, _, FloatNeonUnary;
    Fneg_z_p_z =           0x041da000, Sve,  _, SveUnary(1);
    Fabs_z_p_z =           0x041ca000, Sve,  _, SveUnary(1);
    Fsqrt_z_p_z =          0x650da000, Sve,  _, SveUnary(1);
    Frintm_z_p_z =         0x6502a000, Sve,  _, SveUnary(1);
    Frintp_z_p_z =         0x6501a000, Sve,  _, SveUnary(1);
    Frintz_z_p_z =         0x6503a000, Sve,  _, SveUnary(1);
    Frinta_z_p_z =         0x6504a000, Sve,  _, SveUnary(1);
    Frintn_z_p_z =         0x6500a000, Sve,  _, SveUnary(1);
    Not_z_p_z =            0x041ea000, Sve,  _, SveUnary(0);
    Neg_z_p_z =            0x0417a000, Sve,  _, SveUnary(0);
    Abs_z_p_z =            0x0416a000, Sve,  _, SveUnary(0);
    Clz_z_p_z =            0x0419a000, Sve,  _, SveUnary(0);
    Rbit_z_p_z =           0x05278000, Sve,  _, SveUnary(0);
    Cnt_z_p_z =            0x041aa000, Sve,  _, SveUnary(0);
    Orn_32_log_shift =     0x2a200000, Base, w, Shift(true);
    Orn_64_log_shift =     0xaa200000, Base, x, Shift(true);
    Orr_32_log_imm =       0x32000000, Base, w, LogicalImmediate;
    Orr_64_log_imm =       0xb2000000, Base, x, LogicalImmediate;
    Eor_32_log_imm =       0x52000000, Base, w, LogicalImmediate;
    Clz_32 =               0x5ac01000, Base, _, RegPair(Bank::W, Bank::W);
    Clz_64 =               0xdac01000, Base, _, RegPair(Bank::X, Bank::X);
    Rbit_32 =              0x5ac00000, Base, _, RegPair(Bank::W, Bank::W);
    Rbit_64 =              0xdac00000, Base, _, RegPair(Bank::X, Bank::X);
    Csneg_32_condsel =     0x5a800400, Base, w, Select;
    Csneg_64_condsel =     0xda800400, Base, x, Select;
    Cnt_asimd =            0x4e205800, Neon, _, NeonUnary(0);
    Neg_asimd =            0x6e20b800, Neon, _, NeonUnary(3);
    Abs_asimd =            0x4e20b800, Neon, _, NeonUnary(3);
    Clz_asimd =            0x6e204800, Neon, _, NeonUnary(2);
    Rbit_asimd =           0x6e605800, Neon, _, NeonUnary(0);
    Rev16_asimd =          0x4e201800, Neon, _, NeonUnary(0);
    Rev32_asimd =          0x6e200800, Neon, _, NeonUnary(0);
    Uaddlp_asimd =         0x6e202800, Neon, _, NeonPairwiseLong;
    Fmov_imm_scalar =      0x1e201000, Base, _, FloatImmediate;
    Fmov_imm_asimd =       0x4f00f400, Neon, _, FloatNeonImmediate;
    Fmov_imm_z =           0x2539c000, Sve,  _, SveFloatImmediate;
    Fmadd_scalar =         0x1f000000, Base, _, FloatFma;
    Fmla_asimd =           0x0e20cc00, Neon, _, FloatNeon;
    Fmla_z_p_zz =          0x65200000, Sve,  _, SveFma;
    Compact_z_p_z =        0x05218000, Sve,  _, SveCompact;
    Fcvt_s_d =             0x1e624000, Base, _, RegPair(Bank::V, Bank::V);
    Fcvt_d_s =             0x1e22c000, Base, _, RegPair(Bank::V, Bank::V);
    Scvtf_scalar =         0x1e220000, Base, _, FloatInt(true);
    Ucvtf_scalar =         0x1e230000, Base, _, FloatInt(true);
    Fcvtzs_scalar =        0x1e380000, Base, _, FloatInt(false);
    Fcvtzu_scalar =        0x1e390000, Base, _, FloatInt(false);
    Scvtf_asimd =          0x0e21d800, Neon, _, FloatNeonUnary;
    Ucvtf_asimd =          0x2e21d800, Neon, _, FloatNeonUnary;
    Fcvtzs_asimd =         0x0ea1b800, Neon, _, FloatNeonUnary;
    Fcvtzu_asimd =         0x2ea1b800, Neon, _, FloatNeonUnary;
    Scvtf_z_p_z =          0x6594a000, Sve,  _, SveConvert;
    Ucvtf_z_p_z =          0x6595a000, Sve,  _, SveConvert;
    Fcvtzs_z_p_z =         0x659ca000, Sve,  _, SveConvert;
    Fcvtzu_z_p_z =         0x659da000, Sve,  _, SveConvert;
    Bsl_asimdsame =        0x2e601c00, Neon, _, NeonBits;
    Bic_asimdsame =        0x0e601c00, Neon, _, NeonBits;
    Bic_z_zz_ =            0x04e03000, Sve,  _, SveBits;
    Fcsel_scalar =         0x1e200c00, Base, _, FloatSelect;
    Addv_asimd =           0x4e31b800, Neon, _, NeonSum(false);
    Addp_d_asimd =         0x5ef1b800, Neon, _, NeonSum(true);
    Uaddv_d_p_z =          0x04012000, Sve,  _, SveSum;
    Fcmp_scalar =          0x1e202000, Base, _, FloatCompare;
    Cmeq_asimdsame =       0x2e208c00, Neon, _, Neon(false);
    Cmgt_asimdsame =       0x0e203400, Neon, _, Neon(false);
    Cmge_asimdsame =       0x0e203c00, Neon, _, Neon(false);
    Cmhi_asimdsame =       0x2e203400, Neon, _, Neon(false);
    Cmhs_asimdsame =       0x2e203c00, Neon, _, Neon(false);
    Fcmeq_asimdsame =      0x0e20e400, Neon, _, FloatNeon;
    Fcmgt_asimdsame =      0x2ea0e400, Neon, _, FloatNeon;
    Fcmge_asimdsame =      0x2e20e400, Neon, _, FloatNeon;
    Mvn_asimdmisc =        0x6e205800, Neon, _, NeonUnary(0);
    Cmeq0_asimdmisc =      0x4e209800, Neon, _, NeonUnary(3);
    Cmpeq_p_p_zi =         0x25008000, Sve,  _, SveCompareImmediate;
    Cmpne_p_p_zi =         0x25008010, Sve,  _, SveCompareImmediate;
    Cmpeq_p_p_zz =         0x2400a000, Sve,  _, SveCompare(false);
    Cmpne_p_p_zz =         0x2400a010, Sve,  _, SveCompare(false);
    Cmpgt_p_p_zz =         0x24008010, Sve,  _, SveCompare(false);
    Cmpge_p_p_zz =         0x24008000, Sve,  _, SveCompare(false);
    Cmphi_p_p_zz =         0x24000010, Sve,  _, SveCompare(false);
    Cmphs_p_p_zz =         0x24000000, Sve,  _, SveCompare(false);
    Fcmeq_p_p_zz =         0x65006000, Sve,  _, SveCompare(true);
    Fcmne_p_p_zz =         0x65006010, Sve,  _, SveCompare(true);
    Fcmuo_p_p_zz =         0x6500c000, Sve,  _, SveCompare(true);
    Fcmgt_p_p_zz =         0x65004010, Sve,  _, SveCompare(true);
    Fcmge_p_p_zz =         0x65004000, Sve,  _, SveCompare(true);
    Cpy_z_p_i =            0x05100000, Sve,  _, SveCopyImmediate;
    Index_z_ii =           0x04204000, Sve,  _, SveIndex;
    Add_32_addsub_imm =    0x11000000, Base, w, AddImm(false);
    Add_64_addsub_imm =    0x91000000, Base, x, AddImm(false);
    Sub_32_addsub_imm =    0x51000000, Base, w, AddImm(false);
    Sub_64_addsub_imm =    0xd1000000, Base, x, AddImm(false);
    Adds_32_addsub_imm =   0x31000000, Base, w, AddImm(true);
    Adds_64_addsub_imm =   0xb1000000, Base, x, AddImm(true);
    Subs_32_addsub_imm =   0x71000000, Base, w, AddImm(true);
    Subs_64_addsub_imm =   0xf1000000, Base, x, AddImm(true);
    Add_32_addsub_shift =  0x0b000000, Base, w, Shift(false);
    Add_64_addsub_shift =  0x8b000000, Base, x, Shift(false);
    Adds_64_addsub_shift = 0xab000000, Base, x, Shift(false);
    Sub_32_addsub_shift =  0x4b000000, Base, w, Shift(false);
    Sub_64_addsub_shift =  0xcb000000, Base, x, Shift(false);
    Subs_32_addsub_shift = 0x6b000000, Base, w, Shift(false);
    Subs_64_addsub_shift = 0xeb000000, Base, x, Shift(false);
    And_32_log_shift =     0x0a000000, Base, w, Shift(true);
    And_64_log_shift =     0x8a000000, Base, x, Shift(true);
    Orr_32_log_shift =     0x2a000000, Base, w, Shift(true);
    Orr_64_log_shift =     0xaa000000, Base, x, Shift(true);
    Eor_32_log_shift =     0x4a000000, Base, w, Shift(true);
    Eor_64_log_shift =     0xca000000, Base, x, Shift(true);
    Madd_32a_dp_3src =     0x1b000000, Base, w, Multiply;
    Madd_64a_dp_3src =     0x9b000000, Base, x, Multiply;
    Msub_32a_dp_3src =     0x1b008000, Base, w, Multiply;
    Msub_64a_dp_3src =     0x9b008000, Base, x, Multiply;
    Sdiv_32_dp_2src =      0x1ac00c00, Base, w, Three;
    Sdiv_64_dp_2src =      0x9ac00c00, Base, x, Three;
    Udiv_32_dp_2src =      0x1ac00800, Base, w, Three;
    Udiv_64_dp_2src =      0x9ac00800, Base, x, Three;
    Movz_32_movewide =     0x52800000, Base, w, Move;
    Movz_64_movewide =     0xd2800000, Base, x, Move;
    Movk_32_movewide =     0x72800000, Base, w, Move;
    Movk_64_movewide =     0xf2800000, Base, x, Move;
    Fmov_s_w =             0x1e270000, Base, _, RegPair(Bank::V, Bank::W);
    Fmov_w_s =             0x1e260000, Base, _, RegPair(Bank::W, Bank::V);
    Fmov_d_x =             0x9e670000, Base, _, RegPair(Bank::V, Bank::X);
    Fmov_x_d =             0x9e660000, Base, _, RegPair(Bank::X, Bank::V);
    Fadd_scalar =          0x1e202800, Base, _, FloatScalar;
    Fsub_scalar =          0x1e203800, Base, _, FloatScalar;
    Fmul_scalar =          0x1e200800, Base, _, FloatScalar;
    Fdiv_scalar =          0x1e201800, Base, _, FloatScalar;
    Sbfm_32m_bitfield =    0x13000000, Base, w, Bitfield;
    Sbfm_64m_bitfield =    0x93400000, Base, x, Bitfield;
    Ubfm_64m_bitfield =    0xd3400000, Base, x, Bitfield;
    Add_64_addsub_sxtw =   0x8b20c000, Base, _, ExtendedAdd;
    Ubfm_32m_bitfield =    0x53000000, Base, w, Bitfield;
    Lslv_32_dp_2src =      0x1ac02000, Base, w, Three;
    Lslv_64_dp_2src =      0x9ac02000, Base, x, Three;
    Rorv_32_dp_2src =      0x1ac02c00, Base, w, Three;
    Rorv_64_dp_2src =      0x9ac02c00, Base, x, Three;
    Asrv_32_dp_2src =      0x1ac02800, Base, w, Three;
    Asrv_64_dp_2src =      0x9ac02800, Base, x, Three;
    Lsrv_32_dp_2src =      0x1ac02400, Base, w, Three;
    Lsrv_64_dp_2src =      0x9ac02400, Base, x, Three;
    Csel_32_condsel =      0x1a800000, Base, w, Select;
    Csel_64_condsel =      0x9a800000, Base, x, Select;
    Csinv_32_condsel =     0x5a800000, Base, w, Select;
    Ldr_32_ldst_pos =      0xb9400000, Base, w, Memory(4);
    Ldr_64_ldst_pos =      0xf9400000, Base, x, Memory(8);
    Str_32_ldst_pos =      0xb9000000, Base, w, Memory(4);
    Str_64_ldst_pos =      0xf9000000, Base, x, Memory(8);
    Ldrb_32_ldst_pos =     0x39400000, Base, w, Memory(1);
    Strb_32_ldst_pos =     0x39000000, Base, w, Memory(1);
    Ldrh_32_ldst_pos =     0x79400000, Base, w, Memory(2);
    Strh_32_ldst_pos =     0x79000000, Base, w, Memory(2);
    Ldr_q_ldst_pos =       0x3dc00000, Neon, w, Memory(16);
    Str_q_ldst_pos =       0x3d800000, Neon, w, Memory(16);
    Ldr_s_ldst_pos =       0xbd400000, Base, _, FloatMemory(4);
    Str_s_ldst_pos =       0xbd000000, Base, _, FloatMemory(4);
    Ldr_d_ldst_pos =       0xfd400000, Base, _, FloatMemory(8);
    Str_d_ldst_pos =       0xfd000000, Base, _, FloatMemory(8);
    Nop_HI_hints =         0xd503201f, Base, _, None;
    B_only_branch_imm =    0x14000000, Base, _, Branch(BranchKind::Direct);
    Bl_only_branch_imm =   0x94000000, Base, _, Branch(BranchKind::Direct);
    B_only_condbranch =    0x54000000, Base, _, Branch(BranchKind::Condition);
    Cbz_32_compbranch =    0x34000000, Base, w, Branch(BranchKind::Compare);
    Cbz_64_compbranch =    0xb4000000, Base, x, Branch(BranchKind::Compare);
    Cbnz_32_compbranch =   0x35000000, Base, w, Branch(BranchKind::Compare);
    Cbnz_64_compbranch =   0xb5000000, Base, x, Branch(BranchKind::Compare);
    Br_64_branch_reg =     0xd61f0000, Base, _, Indirect;
    Blr_64_branch_reg =    0xd63f0000, Base, _, Indirect;
    Ret_64r_branch_reg =   0xd65f0000, Base, _, Indirect;
    Add_asimdsame_only =   0x0e208400, Neon, _, Neon(false);
    Sub_asimdsame_only =   0x2e208400, Neon, _, Neon(false);
    Mul_asimdsame_only =   0x0e209c00, Neon, _, Neon(true);
    And_asimdsame_only =   0x0e201c00, Neon, _, NeonBits;
    Orr_asimdsame_only =   0x0ea01c00, Neon, _, NeonBits;
    Eor_asimdsame_only =   0x2e201c00, Neon, _, NeonBits;
    Fadd_asimdsame =       0x0e20d400, Neon, _, FloatNeon;
    Fsub_asimdsame =       0x0ea0d400, Neon, _, FloatNeon;
    Fmul_asimdsame =       0x2e20dc00, Neon, _, FloatNeon;
    Fdiv_asimdsame =       0x2e20fc00, Neon, _, FloatNeon;
    Dup_asimdgen =         0x4e000c00, Neon, _, Dup(false);
    Saddl_asimd =          0x0e200000, Neon, _, Widen(false);
    Saddl2_asimd =         0x4e200000, Neon, _, Widen(true);
    Ptrue_p_s_ =           0x2518e000, Sve,  _, Ptrue;
    Add_z_zz_ =            0x04200000, Sve,  _, SveArithmetic;
    Sub_z_zz_ =            0x04200400, Sve,  _, SveArithmetic;
    Mul_z_zz_ =            0x04206000, Sve2, _, SveArithmetic;
    Mul_z_p_zz_ =          0x04100000, Sve,  _, SveDestructive;
    Fadd_z_zz_ =           0x65000000, Sve,  _, SveArithmetic;
    Fsub_z_zz_ =           0x65000400, Sve,  _, SveArithmetic;
    Fmul_z_zz_ =           0x65000800, Sve,  _, SveArithmetic;
    Fdiv_z_p_zz_ =         0x650d8000, Sve,  _, SveDestructive;
    Fdivr_z_p_zz_ =        0x650c8000, Sve,  _, SveDestructive;
    Dup_z_r =              0x05203800, Sve,  _, Dup(true);
    Cmplt_p_p_zi =         0x25002000, Sve,  _, SveCompareImmediate;
    Sel_z_p_zz =           0x0520c000, Sve,  _, SveSelect;
    Ld1w_z_p_bz_sxtw =     0x85604000, Sve,  _, SveIndexed(A::S, PredQual::Zero);
    Ld1d_z_p_bz_lsl =      0xc5e0c000, Sve,  _, SveIndexed(A::D, PredQual::Zero);
    And_z_zz_ =            0x04203000, Sve,  _, SveBits;
    Orr_z_zz_ =            0x04603000, Sve,  _, SveBits;
    Eor_z_zz_ =            0x04a03000, Sve,  _, SveBits;
    Ld1b_z_p_bi_u8 =       0xa400a000, Sve,  _, SveMemory(A::B, PredQual::Zero);
    Ld1h_z_p_bi_u16 =      0xa4a0a000, Sve,  _, SveMemory(A::H, PredQual::Zero);
    Ld1w_z_p_bi_u32 =      0xa540a000, Sve,  _, SveMemory(A::S, PredQual::Zero);
    Ld1d_z_p_bi_u64 =      0xa5e0a000, Sve,  _, SveMemory(A::D, PredQual::Zero);
    St1b_z_p_bi_ =         0xe400e000, Sve,  _, SveMemory(A::B, PredQual::None);
    St1h_z_p_bi_ =         0xe4a0e000, Sve,  _, SveMemory(A::H, PredQual::None);
    St1w_z_p_bi_ =         0xe540e000, Sve,  _, SveMemory(A::S, PredQual::None);
    St1d_z_p_bi_ =         0xe5e0e000, Sve,  _, SveMemory(A::D, PredQual::None);
    Eor3_z_zzz_ =          0x04203800, Sve2, _, SveTiedTernary;
    Bsl_z_zzz_ =           0x04203c00, Sve2, _, SveTiedTernary;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EncodeError {
    #[error("expected {expected} operands, got {got}")]
    OperandCount { expected: usize, got: usize },
    #[error("invalid operand at index {0}")]
    Operand(usize),
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
        aligned(pc)?;
        let (word, branch) = code.lower(operands)?;
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
    pub(super) fn encode(self, operands: &[O], pc: u64) -> Result<u32, EncodeError> {
        aligned(pc)?;
        let (word, branch) = self.lower(operands)?;
        Ok(word | branch.map_or(Ok(0), |branch| branch.field(pc))?)
    }

    pub fn is_available(self, features: FeatureSet) -> bool {
        self.required_feature().is_none_or(|f| features.contains(f))
    }

    fn lower(self, operands: &[O]) -> Result<Encoded, EncodeError> {
        let (base, wide, form) = self.row();
        let (fields, branch) = form.fields(operands, wide)?;
        Ok((base | fields, branch))
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
    NeonSum(bool),
    SveSum,
    SveFadda,
    SvePredBits,
    NeonSingle,
    FloatNeon,
    Dup(bool),
    SveCompareImmediate,
    SveCompare(bool),
    SveIndex,
    SveCopyImmediate,
    SveDupLane,
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
    Widen(bool),
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

/// A two's complement immediate of `bits` bits, as its field value.
fn signed(o: O, bits: u32, i: usize) -> Result<u32, EncodeError> {
    let limit = 1i64 << (bits - 1);
    match o {
        O::Immediate(v) if (-limit..limit).contains(&(v as i64)) => {
            Ok(v as u32 & ((1 << bits) - 1))
        }
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
            Self::Indirect | Self::Branch(BranchKind::Direct) | Self::PredFalse => 1,
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
            | Self::PredTest => 2,
            Self::Multiply
            | Self::SveTiedTernary
            | Self::Bitfield
            | Self::Select
            | Self::SveSelect
            | Self::FloatSelect
            | Self::FloatFma
            | Self::SveFma
            | Self::SveCompare(_)
            | Self::SveCompareImmediate
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
                    | signed(o[1], 5, 1)? << 5
                    | signed(o[2], 5, 2)? << 16
            }
            Self::SveCopyImmediate => {
                let (d, a) = vector(o[0], Bank::Z, 0)?;
                let merge = match o[1] {
                    O::Predicate(_, PredQual::Merge) => 1,
                    _ => 0,
                };
                let qual = if merge == 1 {
                    PredQual::Merge
                } else {
                    PredQual::Zero
                };
                d | predicate(o[1], qual, 15, 1)? << 16
                    | merge << 14
                    | signed(o[2], 8, 2)? << 5
                    | size(a).ok_or(EncodeError::Operand(0))? << 22
            }
            Self::SveDupLane => {
                let (d, a) = vector(o[0], Bank::Z, 0)?;
                let s = match a {
                    A::Q => 4,
                    _ => size(a).ok_or(EncodeError::Operand(0))?,
                };
                let O::Immediate(index) = o[2] else {
                    return Err(EncodeError::Operand(2));
                };
                if index >= 64 >> s {
                    return Err(EncodeError::Operand(2));
                }
                // imm2:tsz holds the lane size as the lowest set bit and the
                // index above it.
                let field = (index as u32) << (s + 1) | 1 << s;
                d | arranged(o[1], Bank::Z, a, 1)? << 5 | (field & 31) << 16 | (field >> 5) << 22
            }
            Self::SveCompareImmediate | Self::SveCompare(_) => {
                let (d, a) = vector(o[0], Bank::P, 0)?;
                let s = size(a).ok_or(EncodeError::Operand(0))?;
                if s == 0 && matches!(self, Self::SveCompare(true)) {
                    return Err(EncodeError::Operand(0));
                }
                let rhs = if matches!(self, Self::SveCompare(_)) {
                    arranged(o[3], Bank::Z, a, 3)? << 16
                } else {
                    signed(o[3], 5, 3)? << 16
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
                let src = if high { upper } else { low };
                vd | arranged(o[1], Bank::V, src, 1)? << 5
                    | arranged(o[2], Bank::V, src, 2)? << 16
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
