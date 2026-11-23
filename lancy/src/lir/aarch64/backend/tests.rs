use super::compile::Compiled;
use crate::filetests::{Case, Result};
use crate::{
    hir::{Function, ssa::VerifiedBody},
    module::{FunctionIds, Module},
};
use regalloc2::{Block, Function as _, InstOrEdit};
use splice_target::Profile;
use std::fmt::Write;

pub(crate) fn check_machine(
    case: &mut Case<'_>,
    target: &splice_target::Target,
    module: Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<()> {
    if !["aarch64-isel", "aarch64-alloc"]
        .iter()
        .any(|stage| case.checks.contains_key(stage))
        && case.encodings.is_empty()
    {
        return Ok(());
    }
    let simd = super::Config::new(target);
    let compiled = super::compile(module, ids, simd)
        .map_err(|error| format!("aarch64 allocation: {error}"))?;
    let root = &compiled.functions[compiled.call_graph.root()];
    case.check("aarch64-isel", || selected(root))?;
    case.check("aarch64-alloc", || allocated(root))?;
    if !case.encodings.is_empty() {
        let code = super::emit_compiled(&compiled, simd, false).map_err(|e| e.to_string())?;
        let select = match simd.profile {
            Profile::Neon => "neon",
            Profile::Sve => "sve",
            Profile::Sve2 => "sve2",
            profile => return Err(format!("no AArch64 encoding profile for {profile:?}")),
        };
        // AArch64 code has no literal pool: every emitted word is an instruction.
        let words = code.bytes.as_chunks::<4>().0;
        for (line, assertion) in std::mem::take(&mut case.encodings) {
            let invalid = || format!("line {line}: invalid AArch64 assertion: {assertion}");
            let parts: Vec<_> = assertion.split_whitespace().collect();
            let [profile, presence, name] = parts.as_slice() else {
                return Err(invalid());
            };
            let has = match *presence {
                "has" => true,
                "lacks" => false,
                _ => return Err(invalid()),
            };
            if !matches!(*profile, "all" | "neon" | "sve" | "sve2") {
                return Err(invalid());
            }
            let &(_, mask, opcode) = ENCODINGS
                .iter()
                .find(|(known, ..)| known == name)
                .ok_or_else(|| format!("line {line}: unknown AArch64 encoding {name}"))?;
            if *profile != "all" && *profile != select {
                continue;
            }
            let found = words
                .iter()
                .any(|word| u32::from_le_bytes(*word) & mask == opcode);
            if found != has {
                return Err(format!(
                    "line {line}: failed AArch64 assertion: {assertion}"
                ));
            }
        }
    }
    Ok(())
}

/// `@aarch64` encodings: name, mask and value. Names follow the
/// disassembly (`llvm-mc`); `v.`/`z.`/`p.` mark NEON, SVE vector and SVE
/// predicate forms, and a suffix fixes an arrangement or immediate.
const ENCODINGS: &[(&str, u32, u32)] = &[
    ("msub", 0x7fe0_8000, 0x1b00_8000),
    ("sdiv", 0x7fe0_fc00, 0x1ac0_0c00),
    ("rorv", 0x7fe0_fc00, 0x1ac0_2c00),
    ("bl", 0xfc00_0000, 0x9400_0000),
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

fn selected(machine: &Compiled) -> String {
    let mut text = String::new();
    let body = &machine.body;
    for i in 0..body.num_blocks() {
        let b = Block::new(i);
        writeln!(
            text,
            "{b:?} {:?} -> {:?}",
            body.block_params(b),
            body.block_succs(b)
        )
        .unwrap();
        for inst in body.block_insns(b).iter() {
            writeln!(
                text,
                "  {:?} {:?}",
                body.insts()[inst.index()],
                body.inst_operands(inst)
            )
            .unwrap();
        }
    }
    text
}

fn allocated(machine: &Compiled) -> String {
    let mut text = format!("spill words {}\n", machine.allocation.num_spillslots);
    let body = &machine.body;
    for i in 0..body.num_blocks() {
        let b = Block::new(i);
        writeln!(text, "{b:?}").unwrap();
        for entry in machine.allocation.block_insts_and_edits(body, b) {
            match entry {
                InstOrEdit::Edit(edit) => writeln!(text, "  {edit:?}").unwrap(),
                InstOrEdit::Inst(i) => writeln!(
                    text,
                    "  {:?} {:?}",
                    body.insts()[i.index()],
                    machine.allocation.inst_allocs(i)
                )
                .unwrap(),
            }
        }
    }
    text
}

// Executing AArch64 code needs an AArch64 host.
#[cfg(target_arch = "aarch64")]
mod runtime;
