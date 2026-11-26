use lancy::aarch64::{Arrangement as A, Code, Encoder, Error, Instruction, Operand, Register as R};
use lancy::aarch64::{ConditionCode, Feature, FeatureSet, PredQual, ShiftKind};

fn add(immediate: u64) -> [Operand; 3] {
    [
        Operand::reg(R::X0),
        Operand::reg(R::X1),
        Operand::imm(immediate),
    ]
}

#[test]
fn a64_neon_sve_and_sve2_match_llvm_mc() {
    let mut storage = [0_u8; 21];
    let mut encoder = Encoder::new(0x4000, &mut storage[1..]).unwrap();
    encoder.push(Code::Add_64_addsub_imm, &add(7)).unwrap();
    encoder
        .push(
            Code::Add_asimdsame_only,
            &[
                Operand::vec(R::V2, A::S4),
                Operand::vec(R::V3, A::S4),
                Operand::vec(R::V4, A::S4),
            ],
        )
        .unwrap();
    encoder
        .push(
            Code::Ptrue_p_s_,
            &[Operand::vec(R::P0, A::S), Operand::named(8)],
        )
        .unwrap();
    let z5 = Operand::vec(R::Z5, A::D);
    encoder
        .push(
            Code::Eor3_z_zzz_,
            &[z5, z5, Operand::vec(R::Z6, A::D), Operand::vec(R::Z7, A::D)],
        )
        .unwrap();
    encoder
        .push(
            Code::Saddl2_asimd,
            &[
                Operand::vec(R::V0, A::S4),
                Operand::vec(R::V1, A::H8),
                Operand::vec(R::V2, A::H8),
            ],
        )
        .unwrap();
    // llvm-mc-21 -triple=aarch64 -mattr=+sve2 -show-encoding
    assert_eq!(
        encoder.bytes(),
        [
            0x9100_1c20_u32,
            0x4ea4_8462,
            0x2598_e100,
            0x0426_38e5,
            0x4e62_0020
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>()
    );
}

#[test]
fn forward_and_backward_relocations_use_the_instruction_address() {
    let mut storage = [0; 20];
    let mut encoder = Encoder::new(0x4000, &mut storage).unwrap();
    encoder.push(Code::Nop_HI_hints, &[]).unwrap();
    let forward = encoder
        .push(Code::B_only_branch_imm, &[Operand::label(0x4004)])
        .unwrap();
    encoder.push(Code::Nop_HI_hints, &[]).unwrap();
    let target = encoder.pc();
    encoder
        .push(Code::B_only_branch_imm, &[Operand::label(0x4000)])
        .unwrap();
    encoder
        .patch(forward, Code::B_only_branch_imm, &[Operand::label(target)])
        .unwrap();
    assert_eq!(encoder.pc(), 0x4010);
    assert_eq!(&encoder.bytes()[4..8], &0x1400_0002_u32.to_le_bytes());
    assert_eq!(&encoder.bytes()[12..16], &0x17ff_fffd_u32.to_le_bytes());
}

#[test]
fn failed_encoding_and_patch_leave_the_stream_unchanged() {
    let mut storage = [0xff; 7];
    let mut encoder = Encoder::new(0x4000, &mut storage).unwrap();
    assert!(matches!(
        encoder.push(Code::Add_64_addsub_imm, &add(4096)),
        Err(Error::Encoding(_))
    ));
    assert_eq!(encoder.pc(), 0x4000);
    assert!(encoder.bytes().is_empty());

    encoder.push(Code::Add_64_addsub_imm, &add(7)).unwrap();
    let before = [0x20, 0x1c, 0, 0x91];
    assert_eq!(
        encoder.push(Code::Nop_HI_hints, &[]),
        Err(Error::BufferFull)
    );
    assert_eq!(
        encoder.patch(1, Code::Nop_HI_hints, &[]),
        Err(Error::InvalidPatch(1))
    );
    assert_eq!(
        encoder.patch(4, Code::Nop_HI_hints, &[]),
        Err(Error::InvalidPatch(4))
    );
    for target in [0x4001, 0x4000 + (1 << 27)] {
        assert!(matches!(
            encoder.patch(0, Code::B_only_branch_imm, &[Operand::label(target)]),
            Err(Error::Encoding(_))
        ));
    }
    assert_eq!(encoder.pc(), 0x4004);
    assert_eq!(encoder.bytes(), before);
    assert_eq!(storage[4..], [0xff; 3]);
}

#[test]
fn rejects_invalid_target_address_ranges() {
    let mut storage = [0; 4];
    assert!(matches!(
        Encoder::new(3, &mut storage),
        Err(Error::UnalignedAddress(3))
    ));
    assert!(matches!(
        Encoder::new(u64::MAX - 3, &mut storage),
        Err(Error::AddressOverflow)
    ));
}

#[test]
fn reused_instruction_is_encoded_at_its_new_pc() {
    let branch =
        Instruction::with(Code::B_only_branch_imm, &[Operand::label(0x400c)], 0x4000).unwrap();
    let mut storage = [0; 8];
    let mut encoder = Encoder::new(0x4004, &mut storage).unwrap();
    encoder.push_instruction(&branch).unwrap();
    encoder.push_instruction(&branch).unwrap();
    assert_eq!(encoder.bytes(), &[2, 0, 0, 0x14, 1, 0, 0, 0x14]);
}

#[test]
fn scalar_and_vector_field_boundaries_match_llvm_mc() {
    use Code::*;
    let r = Operand::reg;
    let v = Operand::vec;
    let shifted = Operand::shifted;
    let immediate = Operand::imm_lsl;
    let memory = Operand::mem_off;
    let label = Operand::label;
    let cases: &[(Code, &[Operand], u32)] = &[
        (
            Adds_64_addsub_imm,
            &[r(R::XZR), r(R::SP), immediate(4095, 12)],
            0xb17f_ffff,
        ),
        (
            Sub_32_addsub_shift,
            &[r(R::W4), r(R::W3), shifted(R::W2, ShiftKind::Asr, 31)],
            0x4b82_7c64,
        ),
        (
            Eor_64_log_shift,
            &[r(R::X9), r(R::X3), shifted(R::X8, ShiftKind::Ror, 63)],
            0xcac8_fc69,
        ),
        (
            Madd_64a_dp_3src,
            &[r(R::X0), r(R::X1), r(R::X2), r(R::X3)],
            0x9b02_0c20,
        ),
        (
            Sdiv_32_dp_2src,
            &[r(R::W0), r(R::W1), r(R::W2)],
            0x1ac2_0c20,
        ),
        (
            Movk_64_movewide,
            &[r(R::X13), immediate(65535, 48)],
            0xf2ff_ffed,
        ),
        (
            Ldr_64_ldst_pos,
            &[r(R::X30), memory(R::SP, 32760)],
            0xf97f_fffe,
        ),
        (
            Str_q_ldst_pos,
            &[r(R::V31), memory(R::SP, 65520)],
            0x3dbf_ffff,
        ),
        (Cbnz_64_compbranch, &[r(R::X3), label(0xffc)], 0xb5ff_ffe3),
        (
            B_only_condbranch,
            &[Operand::cond(ConditionCode::Ne), label(0x100c)],
            0x5400_0061,
        ),
        (Ret_64r_branch_reg, &[r(R::X30)], 0xd65f_03c0),
        (
            Mul_asimdsame_only,
            &[v(R::V10, A::H8), v(R::V11, A::H8), v(R::V12, A::H8)],
            0x4e6c_9d6a,
        ),
        (
            Add_z_zz_,
            &[v(R::Z31, A::D), v(R::Z30, A::D), v(R::Z29, A::D)],
            0x04fd_03df,
        ),
    ];
    for &(code, operands, expected) in cases {
        assert_eq!(
            Instruction::with(code, operands, 0x1000).unwrap().encode(),
            Ok(expected),
            "{code:?}"
        );
    }
}

#[test]
fn rejects_reserved_fields_and_wrong_register_roles() {
    use Code::*;
    let r = Operand::reg;
    let v = Operand::vec;
    let cases: &[(Code, &[Operand])] = &[
        (Add_64_addsub_imm, &[r(R::XZR), r(R::SP), Operand::imm(1)]),
        (Ldr_64_ldst_pos, &[r(R::X0), Operand::mem(R::XZR)]),
        (Ldr_64_ldst_pos, &[r(R::X0), Operand::mem_off(R::SP, 7)]),
        (
            Add_32_addsub_shift,
            &[
                r(R::W0),
                r(R::W1),
                Operand::shifted(R::W2, ShiftKind::Lsl, 32),
            ],
        ),
        (
            Mul_asimdsame_only,
            &[v(R::V0, A::D2), v(R::V1, A::D2), v(R::V2, A::D2)],
        ),
        (Ptrue_p_s_, &[v(R::P0, A::S), Operand::named(14)]),
        (
            Eor3_z_zzz_,
            &[
                v(R::Z0, A::D),
                v(R::Z1, A::D),
                v(R::Z2, A::D),
                v(R::Z3, A::D),
            ],
        ),
    ];
    for &(code, operands) in cases {
        assert!(Instruction::with(code, operands, 0).is_err(), "{code:?}");
    }
}

#[test]
fn sve_memory_and_sve2_arithmetic_match_llvm_mc() {
    use Code::*;
    let z = Operand::vec;
    let pred = Operand::pred;
    let mem = Operand::mem_mul_vl;
    let zd = [z(R::Z31, A::D), z(R::Z30, A::D), z(R::Z29, A::D)];
    for (code, expected) in [
        (Sub_z_zz_, 0x04fd_07df),
        (Mul_z_zz_, 0x04fd_63df),
        (And_z_zz_, 0x043d_33df),
        (Orr_z_zz_, 0x047d_33df),
        (Eor_z_zz_, 0x04bd_33df),
    ] {
        assert_eq!(
            Instruction::with(code, &zd, 0).unwrap().encode(),
            Ok(expected)
        );
    }
    let load = [z(R::Z31, A::S), pred(R::P7, PredQual::Zero), mem(R::SP, -8)];
    let store = [z(R::Z31, A::D), pred(R::P7, PredQual::None), mem(R::SP, 7)];
    assert_eq!(
        Instruction::with(Ld1w_z_p_bi_u32, &load, 0)
            .unwrap()
            .encode(),
        Ok(0xa548_bfff)
    );
    assert_eq!(
        Instruction::with(St1d_z_p_bi_, &store, 0).unwrap().encode(),
        Ok(0xe5e7_ffff)
    );
    for operands in [
        [load[0], pred(R::P8, PredQual::Zero), load[2]],
        [load[0], pred(R::P7, PredQual::Merge), load[2]],
        [load[0], load[1], mem(R::SP, 8)],
    ] {
        assert!(Instruction::with(Ld1w_z_p_bi_u32, &operands, 0).is_err());
    }
    assert!(!Mul_z_zz_.is_available(FeatureSet::NONE.with(Feature::Sve)));
    assert!(Ld1w_z_p_bi_u32.is_available(FeatureSet::NONE.with(Feature::Sve2)));
}

#[test]
fn wide_arithmetic_and_permutations_match_llvm_mc() {
    use Code::*;
    let r = Operand::reg;
    let v = Operand::vec;
    let p = Operand::pred;
    let imm = Operand::imm;
    let cases: &[(Code, &[Operand], u32)] = &[
        (Adc_64, &[r(R::X0), r(R::X1), r(R::X2)], 0x9a02_0020),
        (
            Dup_z_zi,
            &[v(R::Z0, A::Q), v(R::Z1, A::Q), imm(0)],
            0x0530_2020,
        ),
        (
            Bsl_z_zzz_,
            &[
                v(R::Z7, A::D),
                v(R::Z7, A::D),
                v(R::Z11, A::D),
                v(R::Z19, A::D),
            ],
            0x042b_3e67,
        ),
        (
            Trn1_p_pp,
            &[v(R::P1, A::D), v(R::P3, A::D), v(R::P2, A::D)],
            0x05e2_5061,
        ),
        (
            Lsl_z_zi,
            &[v(R::Z0, A::D), v(R::Z1, A::D), imm(63)],
            0x04ff_9c20,
        ),
        (Orr_32_log_imm, &[r(R::W0), r(R::W1), imm(256)], 0x3218_0020),
        (Dupm_z, &[v(R::Z0, A::B), imm(7)], 0x05c0_0640),
        (
            Tbl_asimd,
            &[v(R::V0, A::B8), v(R::V1, A::B16), v(R::V2, A::B8)],
            0x0e02_0020,
        ),
        (
            Fcvt_z_d_p_s,
            &[v(R::Z0, A::D), p(R::P1, PredQual::Merge), v(R::Z1, A::S)],
            0x65cb_a420,
        ),
        (
            Histcnt_z_p_zz,
            &[
                v(R::Z0, A::S),
                p(R::P0, PredQual::Zero),
                v(R::Z1, A::S),
                v(R::Z2, A::S),
            ],
            0x45a2_c020,
        ),
    ];
    for &(code, operands, expected) in cases {
        assert_eq!(
            Instruction::with(code, operands, 0).unwrap().encode(),
            Ok(expected),
            "{code:?}"
        );
    }
    let out = v(R::V0, A::S4);
    let src = v(R::V1, A::S4);
    assert!(Instruction::with(Sshr_asimd_imm, &[out, src, imm(0)], 0).is_err());
    assert!(Instruction::with(Sshr_asimd_imm, &[out, src, imm(33)], 0).is_err());
    assert!(Instruction::with(Orr_32_log_imm, &[r(R::W0), r(R::W1), imm(0)], 0).is_err());
    let sve = FeatureSet::NONE.with(Feature::Sve);
    assert!(Dup_z_zi.is_available(sve));
    assert!(!Histcnt_z_p_zz.is_available(sve));
    assert!(!Bsl_z_zzz_.is_available(sve));
    assert!(Bsl_z_zzz_.is_available(sve.with(Feature::Sve2)));
}

#[test]
fn lane_and_immediate_operands_match_llvm_mc() {
    use Code::*;
    let r = Operand::reg;
    let v = Operand::vec;
    let p = Operand::pred;
    let imm = Operand::imm;
    let simm = Operand::simm;
    let cases: &[(Code, &[Operand], u32)] = &[
        (
            Cpy_z_p_i,
            &[v(R::Z0, A::S), p(R::P1, PredQual::Zero), simm(-1)],
            0x0591_1fe0,
        ),
        (
            Cpy_z_p_i,
            &[v(R::Z3, A::H), p(R::P2, PredQual::Merge), imm(0)],
            0x0552_4003,
        ),
        (
            Cpy_z_p_i,
            &[v(R::Z4, A::B), p(R::P15, PredQual::Zero), imm(127)],
            0x051f_0fe4,
        ),
        (
            Cpy_z_p_i,
            &[v(R::Z5, A::D), p(R::P0, PredQual::Zero), simm(-128)],
            0x05d0_1005,
        ),
        (Index_z_ii, &[v(R::Z0, A::S), imm(0), imm(1)], 0x04a1_4000),
        (
            Index_z_ii,
            &[v(R::Z1, A::D), simm(-16), imm(15)],
            0x04ef_4201,
        ),
        (
            Cmpeq_p_p_zi,
            &[
                v(R::P1, A::S),
                p(R::P4, PredQual::Zero),
                v(R::Z2, A::S),
                simm(-1),
            ],
            0x259f_9041,
        ),
        (
            Cmplt_p_p_zi,
            &[
                v(R::P1, A::D),
                p(R::P2, PredQual::Zero),
                v(R::Z3, A::D),
                imm(0),
            ],
            0x25c0_2861,
        ),
        (
            Cmpne_p_p_zi,
            &[
                v(R::P7, A::B),
                p(R::P7, PredQual::Zero),
                v(R::Z31, A::B),
                imm(15),
            ],
            0x250f_9ff7,
        ),
        (
            Dup_z_zi,
            &[v(R::Z0, A::Q), v(R::Z1, A::Q), imm(3)],
            0x05f0_2020,
        ),
        (
            Dup_z_zi,
            &[v(R::Z2, A::S), v(R::Z3, A::S), imm(15)],
            0x05fc_2062,
        ),
        (
            Dup_z_zi,
            &[v(R::Z0, A::B), v(R::Z1, A::B), imm(63)],
            0x05ff_2020,
        ),
        (
            Umov_lane,
            &[r(R::W9), v(R::V16, A::B16), imm(0)],
            0x0e01_3e09,
        ),
        (Umov_lane, &[r(R::X1), v(R::V2, A::D2), imm(1)], 0x4e18_3c41),
        (Ins_lane, &[v(R::V2, A::D2), r(R::X3), imm(1)], 0x4e18_1c62),
        (
            Sshr_asimd_imm,
            &[v(R::V0, A::D2), v(R::V1, A::D2), imm(63)],
            0x4f41_0420,
        ),
    ];
    for &(code, operands, expected) in cases {
        assert_eq!(
            Instruction::with(code, operands, 0).unwrap().encode(),
            Ok(expected),
            "{code:?}"
        );
    }
    let z0 = v(R::Z0, A::S);
    for (code, operands) in [
        (Cpy_z_p_i, vec![z0, p(R::P1, PredQual::Zero), imm(128)]),
        (Cpy_z_p_i, vec![z0, p(R::P1, PredQual::None), imm(1)]),
        (Index_z_ii, vec![z0, imm(16), imm(1)]),
        (Index_z_ii, vec![z0, simm(-17), imm(1)]),
        (Dup_z_zi, vec![v(R::Z0, A::Q), v(R::Z1, A::Q), imm(4)]),
        (Dup_z_zi, vec![z0, v(R::Z1, A::D), imm(0)]),
    ] {
        assert!(Instruction::with(code, &operands, 0).is_err(), "{code:?}");
    }
}
