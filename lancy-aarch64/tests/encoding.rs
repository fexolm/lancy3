use lancy_aarch64::{Arrangement as A, Code, Encoder, Error, Instruction, Operand, Register as R};

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
        .push_with_suffix(
            Code::Saddl_asimddiff_L,
            1,
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
