//! Encodings that `@aarch64` filetests look for, and tests that execute
//! AArch64 code.

/// The mask and value of the encoding `@aarch64` names `name`.
pub(crate) fn encoding(name: &str) -> Option<(u32, u32)> {
    ENCODINGS
        .iter()
        .find(|(known, ..)| *known == name)
        .map(|&(_, mask, value)| (mask, value))
}

/// `@aarch64` encodings: name, mask and value. Names follow the
/// disassembly (`llvm-mc`); `v.`/`z.`/`p.` mark NEON, SVE vector and SVE
/// predicate forms, and a suffix fixes an arrangement or immediate.
const ENCODINGS: &[(&str, u32, u32)] = &[
    ("msub", 0x7fe0_8000, 0x1b00_8000),
    ("sdiv", 0x7fe0_fc00, 0x1ac0_0c00),
    ("rorv", 0x7fe0_fc00, 0x1ac0_2c00),
    ("bl", 0xfc00_0000, 0x9400_0000),
    ("add.x29.sp.24", 0xffff_ffff, 0x9100_63fd),
    ("cbz.w", 0xff00_0000, 0x3400_0000),
    ("cbnz.w", 0xff00_0000, 0x3500_0000),
    ("v.fminnm", 0xbfa0_fc00, 0x0ea0_c400),
    ("v.fmaxnm", 0xbfa0_fc00, 0x0e20_c400),
    ("z.fminnm", 0xff3f_e000, 0x6505_8000),
    ("z.fmaxnm", 0xff3f_e000, 0x6504_8000),
    ("fmov.x.d", 0xffff_fc00, 0x9e66_0000),
    ("fcmp", 0xff20_fc07, 0x1e20_2000),
    ("z.lastb", 0xff3f_e000, 0x0521_a000),
    ("z.ld1b", 0xfff0_e000, 0xa400_a000),
    ("z.st1b", 0xfff0_e000, 0xe400_e000),
    ("blr", 0xffff_fc1f, 0xd63f_0000),
    ("add.w", 0xffe0_fc00, 0x0b00_0000),
    ("sub.w", 0xffe0_fc00, 0x4b00_0000),
    ("add.w.4095", 0xffff_fc00, 0x113f_fc00),
    ("sub.w.4096", 0xffff_fc00, 0x5140_0400),
    ("adc.x", 0xffe0_fc00, 0x9a00_0000),
    ("adds.x", 0xffe0_fc00, 0xab00_0000),
    ("sbc.x", 0xffe0_fc00, 0xda00_0000),
    ("subs.x", 0xffe0_fc00, 0xeb00_0000),
    ("umulh", 0xffe0_fc00, 0x9bc0_7c00),
    ("fmadd.s", 0xffe0_8000, 0x1f00_0000),
    ("fdiv.s", 0xffe0_fc00, 0x1e20_1800),
    ("fcvtxn.s.d", 0xffff_fc00, 0x7e61_6800),
    ("v.sqadd", 0xff20_fc00, 0x4e20_0c00),
    ("v.sqadd.16b", 0xffe0_fc00, 0x4e20_0c00),
    ("v.sshl", 0xff20_fc00, 0x4e20_4400),
    ("v.smin", 0xff20_fc00, 0x4e20_6c00),
    ("v.uqsub", 0xff20_fc00, 0x6e20_2c00),
    ("v.umax", 0xff20_fc00, 0x6e20_6400),
    ("v.xtn2", 0xff3f_fc00, 0x4e21_2800),
    ("v.ext.16b", 0xffe0_8400, 0x6e00_0000),
    ("v.tbl2.16b", 0xffe0_fc00, 0x4e00_2000),
    ("v.mul.16b", 0xffe0_fc00, 0x4e20_9c00),
    ("v.fmla.4s", 0xffe0_fc00, 0x4e20_cc00),
    ("v.fdiv.2d", 0xffe0_fc00, 0x6e60_fc00),
    ("v.sshr.16b.7", 0xffff_fc00, 0x4f09_0400),
    ("v.uaddlp.8h", 0xffff_fc00, 0x6e20_2800),
    ("v.frinta.4s", 0xffff_fc00, 0x6e21_8800),
    ("z.mls", 0xff20_e000, 0x0400_6000),
    ("z.sqadd", 0xff20_fc00, 0x0420_1000),
    ("z.uqsub", 0xff20_fc00, 0x0420_1c00),
    ("z.mul.unpredicated", 0xff20_fc00, 0x0420_6000),
    ("z.mul.b", 0xffff_e000, 0x0410_0000),
    ("z.tbl", 0xff20_fc00, 0x0520_3000),
    ("p.trn1", 0xff30_fc00, 0x0520_5000),
    ("z.umax", 0xff3f_e000, 0x0409_0000),
    ("z.smin", 0xff3f_e000, 0x040a_0000),
    ("z.asr", 0xff3f_e000, 0x0410_8000),
    ("z.lsl", 0xff3f_e000, 0x0413_8000),
    ("z.asr.b.7", 0xffff_fc00, 0x0429_9000),
    ("z.sdiv", 0xff3f_e000, 0x0414_0000),
    ("z.orv", 0xff3f_e000, 0x0418_2000),
    ("z.andv", 0xff3f_e000, 0x041a_2000),
    ("z.splice", 0xff3f_e000, 0x052c_8000),
    ("z.fadda", 0xff3f_e000, 0x6518_2000),
    ("z.sunpklo", 0xff3f_fc00, 0x0530_3800),
    ("z.histcnt", 0xffa0_e000, 0x45a0_c000),
    ("z.histcnt.s", 0xffe0_e000, 0x45a0_c000),
    ("z.histcnt.d", 0xffe0_e000, 0x45e0_c000),
    ("z.fmla.s", 0xffe0_e000, 0x65a0_0000),
    ("z.ld1h.gather.s", 0xffe0_e000, 0x84e0_4000),
    ("z.st1b.scatter.s", 0xffe0_e000, 0xe440_c000),
    ("p.fcmuo.s", 0xffe0_e010, 0x6580_c000),
    ("z.eor3", 0xffe0_fc00, 0x0420_3800),
    ("z.bsl", 0xffe0_fc00, 0x0420_3c00),
    ("z.cnt.h", 0xffff_e000, 0x045a_a000),
    ("z.compact.s", 0xffff_e000, 0x05a1_8000),
    ("z.ucvtf.h", 0xffff_e000, 0x6553_a000),
    ("z.frinta.s", 0xffff_e000, 0x6584_a000),
    ("z.fdiv.d", 0xffff_e000, 0x65cd_8000),
    ("z.ld1w", 0xffff_e000, 0xa540_a000),
    ("z.st1w", 0xffff_e000, 0xe540_e000),
    ("z.dup.q", 0xffff_fc00, 0x0530_2000),
    ("z.index.s.0.1", 0xffff_ffe0, 0x04a1_4000),
];

#[test]
fn encoding_names_are_unique_and_values_fit_their_masks() {
    for (i, &(name, mask, opcode)) in ENCODINGS.iter().enumerate() {
        assert_eq!(opcode & !mask, 0, "{name}");
        assert!(
            ENCODINGS[..i].iter().all(|&(other, ..)| other != name),
            "{name}"
        );
    }
}

// Executing AArch64 code needs an AArch64 host.
#[cfg(target_arch = "aarch64")]
mod runtime;

/// A conditional branch reaches ±1 MiB. Beyond that the assembler reports
/// that the code needs long branches, and with them inverts the condition
/// over an unconditional `B` (A64-07).
#[test]
fn far_conditional_branches_relax_to_an_inverted_skip_over_b() {
    use super::{Assembler, C, Cond, R};
    use splice_target::Profile;
    let assemble = |long| {
        let mut a = Assembler::new(Profile::Neon, long);
        let target = a.label();
        a.jump_if(Cond::Eq, target).unwrap();
        a.jump_if_zero(true, R::W3, target).unwrap();
        for _ in 0..(1 << 18) {
            a.put(C::Nop_HI_hints, &[]).unwrap();
        }
        a.bind(target);
        a.put(C::Nop_HI_hints, &[]).unwrap();
        a.finish().unwrap()
    };
    assert!(assemble(false).is_none());
    let code = assemble(true).unwrap();
    let words: Vec<u32> = code
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .take(4)
        .collect();
    // b.ne +8; b target; cbnz w3, +8; b target
    assert_eq!(words[0], 0x5400_0041);
    assert_eq!(words[1] & 0xfc00_0000, 0x1400_0000);
    assert_eq!(words[2], 0x3500_0043);
    assert_eq!(words[3] & 0xfc00_0000, 0x1400_0000);
}
