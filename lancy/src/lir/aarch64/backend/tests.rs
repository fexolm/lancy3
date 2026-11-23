use super::compile::Compiled;
use crate::filetests::{Case, Result, input};
use crate::{
    CallError, Inlining, ScalarKernel,
    hir::{Function, ssa::VerifiedBody},
    module::{FunctionIds, Module},
};
use regalloc2::{Block, Function as _, InstOrEdit};
use splice_target::Profile;
use std::fmt::Write;

pub(crate) fn check_machine(
    case: &mut Case<'_>,
    module: Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<()> {
    if case.checks.is_empty() && case.encodings.is_empty() {
        return Ok(());
    }
    let simd = splice_target::simd().map_err(|error| error.to_string())?;
    let compiled = super::compile(module, ids, simd)
        .map_err(|error| format!("aarch64 allocation: {error}"))?;
    let root = &compiled.functions[compiled.call_graph.root()];
    case.check("aarch64-isel", || selected(root))?;
    case.check("aarch64-alloc", || allocated(root))?;
    case.check("layout", || crate::lir::layout::listing(&root.body))?;
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

fn parse(text: &str) -> std::sync::Arc<Function> {
    let lines: input::Lines<'_> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
        .collect();
    input::parse(&lines).unwrap()
}

fn lane_words(lanes: impl IntoIterator<Item = u128>, size: usize) -> Vec<u64> {
    let mut bytes = Vec::new();
    for lane in lanes {
        bytes.extend_from_slice(&lane.to_le_bytes()[..size]);
    }
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|word| u64::from_le_bytes(*word))
        .collect()
}

#[test]
fn packed_ordered_comparisons_evaluate_both_operands() {
    let width = ScalarKernel::vector_bytes();
    for (size, float, mask, qnan, snan, inf) in [
        (4, "f32", "i32", 0x7fc00042, 0xff800031, 0x7f800000),
        (
            8,
            "f64",
            "i64",
            0x7ff8000000000042,
            0xfff0000000000031,
            0x7ff0000000000000,
        ),
    ] {
        let rows = [
            (qnan, 2, false),
            (3, snan, false),
            (inf, inf | (1 << (size * 8 - 1)), true),
            (1 << (size * 8 - 1), 0, true),
            (snan, qnan, false),
            (1, 7, true),
        ];
        let count = width / size;
        for (op, invert) in [("ordered", false), ("unordered", true)] {
            let f = parse(&format!(
                "fn main a:vec.{float} b:vec.{float} -> vec.{mask}\n r = simd.{op} a b\n ret r"
            ));
            let kernel = ScalarKernel::compile(&f).unwrap();
            for start in 0..rows.len() {
                let lanes = rows.into_iter().cycle().skip(start).take(count);
                let mut args = lane_words(lanes.clone().map(|(x, _, _)| x), size);
                args.extend(lane_words(lanes.clone().map(|(_, y, _)| y), size));
                let mut out = vec![0; width / 8];
                kernel.call_words(&args, &mut out).unwrap();
                let expected =
                    lanes.map(|(_, _, ordered)| if ordered ^ invert { u128::MAX } else { 0 });
                assert_eq!(out, lane_words(expected, size), "{float} {op}");
            }
        }
    }
}

#[test]
fn rotations_wrap_counts_within_each_integer_lane() {
    let width = ScalarKernel::vector_bytes();
    for (ty, size) in [("i8", 1), ("u16", 2), ("i32", 4), ("u64", 8)] {
        let bits = size * 8;
        let mask = (1_u128 << bits) - 1;
        let high = 1_u128 << (bits - 1);
        let counts = [
            0,
            1,
            bits - 1,
            bits,
            bits + 1,
            bits * 2,
            mask,
            mask + 1 - bits,
            mask - bits,
            high,
        ];
        for (op, right) in [("rotateleft", false), ("rotateright", true)] {
            let expected = |data: u128, count: u128| {
                let c = if right { count.wrapping_neg() } else { count } % bits;
                ((data << c) | (data >> (bits - c))) & mask
            };
            let f = parse(&format!(
                "fn main a:{ty} b:{ty} -> {ty}\n r = {op} a b\n ret r"
            ));
            let kernel = ScalarKernel::compile(&f).unwrap();
            let data = high | 1;
            for c in counts {
                let mut out = [0];
                kernel
                    .call_words(&[data as u64, c as u64], &mut out)
                    .unwrap();
                assert_eq!(u128::from(out[0]) & mask, expected(data, c), "{op} {ty}");
            }
            let f = parse(&format!(
                "fn main a:vec.{ty} b:vec.{ty} -> vec.{ty}\n r = simd.{op} a b\n ret r"
            ));
            let kernel = ScalarKernel::compile(&f).unwrap();
            let n = width / size as usize;
            for offset in 0..counts.len() {
                let counts = counts.into_iter().cycle().skip(offset).take(n);
                let vals = (0..n).map(|i| (data ^ (i * 37) as u128) & mask);
                let mut args = lane_words(vals.clone(), size as usize);
                args.extend(lane_words(counts.clone(), size as usize));
                let mut out = vec![0; width / 8];
                kernel.call_words(&args, &mut out).unwrap();
                let want = vals.zip(counts).map(|(a, b)| expected(a, b));
                assert_eq!(out, lane_words(want, size as usize), "{op} {ty}");
            }
        }
    }
}

#[test]
fn packed_float_specials_preserve_payloads_and_form_whole_lane_masks() {
    let n = ScalarKernel::vector_bytes() / 8;
    let patterns = [
        0,
        1,
        0x000fffffffffffff,
        0x0010000000000000,
        0x7fefffffffffffff,
        0x7ff0000000000000,
        0x7ff0000000000042,
        0x8000000000000000,
        0x8000000000000001,
        0xfff8543210abcdef,
    ];
    for op in [
        "nextdown",
        "signum",
        "isnormal",
        "issubnormal",
        "isnan",
        "signbit",
    ] {
        let boolean = op.starts_with("is") || op == "signbit";
        let result = if boolean { "i64" } else { "f64" };
        let f = parse(&format!(
            "fn main a:vec.f64 -> vec.{result}\n r = simd.{op} a\n ret r"
        ));
        let kernel = ScalarKernel::compile(&f).unwrap();
        for offset in [0, 3, 7] {
            let args: Vec<_> = patterns.into_iter().cycle().skip(offset).take(n).collect();
            let mut out = vec![0; n];
            kernel.call_words(&args, &mut out).unwrap();
            let want = args
                .into_iter()
                .map(|bits| {
                    let x = f64::from_bits(bits);
                    let mask = |v| 0_u64.wrapping_sub(u64::from(v));
                    match op {
                        "nextdown" => x.next_down().to_bits(),
                        "signum" if x.is_nan() => 0x7ff8000000000000,
                        "signum" => x.signum().to_bits(),
                        "isnormal" => mask(x.is_normal()),
                        "issubnormal" => mask(x.is_subnormal()),
                        "isnan" => mask(x.is_nan()),
                        _ => mask(x.is_sign_negative()),
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(out, want, "{op}");
        }
    }

    let f = parse("fn main m:u128 a:u128 b:u128 -> u128\n r = bitselect m a b\n ret r");
    let kernel = ScalarKernel::compile(&f).unwrap();
    let args = [
        0x00ff00ffff00ff00,
        0xf0f00f0faa55aa55,
        0x0123456789abcdef,
        0xfedcba9876543210,
        0x0ff03cc355aa9966,
        0x123456789abcdef0,
    ];
    let mut out = [0; 2];
    kernel.call_words(&args, &mut out).unwrap();
    assert_eq!(out, [0x0f233c6789aacd66, 0xf2d45a7832fc76b0]);
}

#[test]
fn width_changing_float_conversions_use_distinct_lane_cursors() {
    let width = ScalarKernel::vector_bytes();
    for (ty, cases) in [
        (
            "f16",
            [
                (0x0001, 0x33800000),
                (0x03ff, 0x387fc000),
                (0x0400, 0x38800000),
                (0x8000, 0x80000000),
                (0x7c01, 0x7fc00000),
                (0xfc01, 0xffc00000),
            ],
        ),
        (
            "bf16",
            [
                (0x0001, 0x00010000),
                (0x007f, 0x007f0000),
                (0x0080, 0x00800000),
                (0x8000, 0x80000000),
                (0x7f81, 0x7fc00000),
                (0xff81, 0xffc00000),
            ],
        ),
    ] {
        let cases = cases
            .into_iter()
            .cycle()
            .take(width / 2)
            .collect::<Vec<_>>();
        let args = lane_words(cases.iter().map(|&(bits, _)| bits), 2);
        for (op, start) in [("widenlo", 0), ("widenhi", width / 4)] {
            let f = parse(&format!(
                "fn main a:vec.{ty} -> vec.f32\n r = simd.{op} f32 a\n ret r"
            ));
            let kernel = ScalarKernel::compile(&f).unwrap();
            let mut out = vec![0; width / 8];
            kernel.call_words(&args, &mut out).unwrap();
            let want = cases
                .iter()
                .skip(start)
                .take(width / 4)
                .map(|&(_, expected)| expected);
            assert_eq!(out, lane_words(want, 4), "{ty} {op}");
        }
    }

    let f = parse("fn main a:vec.f32 b:vec.f32 -> vec.bf16\n r = simd.narrow bf16 a b\n ret r");
    let kernel = ScalarKernel::compile(&f).unwrap();
    let cases = [
        (0x3f807fff, 0x3f80),
        (0x3f808000, 0x3f80),
        (0x3f808001, 0x3f81),
        (0x00008000, 0),
        (0x00008001, 1),
        (0x007f7fff, 0x007f),
        (0x007f8000, 0x0080),
        (0xff800001, 0xffc0),
        (0x80000000, 0x8000),
    ];
    let cases = cases
        .into_iter()
        .cycle()
        .take(width / 2)
        .collect::<Vec<_>>();
    let args = lane_words(cases.iter().map(|&(bits, _)| bits), 4);
    let mut out = vec![0; width / 8];
    kernel.call_words(&args, &mut out).unwrap();
    assert_eq!(
        out,
        lane_words(cases.iter().map(|&(_, expected)| expected), 2)
    );

    let f = parse("fn main a:vec.f16 b:vec.f16 -> vec.i8\n r = simd.narrow i8 a b\n ret r");
    let kernel = ScalarKernel::compile(&f).unwrap();
    let cases = [
        (0xbe00, 255),
        (0x3e00, 1),
        (0x5800, 127),
        (0xfc00, 128),
        (0x7c00, 127),
        (0xfe01, 0),
        (0x8000, 0),
        (0xd800, 128),
    ];
    let cases = cases.into_iter().cycle().take(width).collect::<Vec<_>>();
    let args = lane_words(cases.iter().map(|&(bits, _)| bits), 2);
    kernel.call_words(&args, &mut out).unwrap();
    assert_eq!(
        out,
        lane_words(cases.iter().map(|&(_, expected)| expected), 1)
    );

    let f = parse("fn main a:vec.f64 -> vec.i128\n r = simd.widenhi i128 a\n ret r");
    let kernel = ScalarKernel::compile(&f).unwrap();
    let fp = [
        1.875,
        -18_446_744_073_709_555_712.0,
        2.0_f64.powi(127),
        -2.0_f64.powi(127),
        f64::NAN,
        f64::INFINITY,
        2.0_f64.powi(128),
        2.0_f64.powi(128).next_down(),
    ];
    let args = fp[..width / 8]
        .iter()
        .map(|x| x.to_bits())
        .collect::<Vec<_>>();
    kernel.call_words(&args, &mut out).unwrap();
    assert_eq!(
        out,
        lane_words(
            fp[width / 16..width / 8].iter().map(|&x| x as i128 as u128),
            16
        )
    );

    let f = parse("fn main a:vec.u128 b:vec.u128 -> vec.f64\n r = simd.narrow f64 a b\n ret r");
    let kernel = ScalarKernel::compile(&f).unwrap();
    let mid = (1_u128 << 100) + (1 << 63) + (1 << 47);
    let vals = [
        mid - 1,
        mid + 1,
        mid,
        u128::MAX,
        1,
        (1 << 64) + 5,
        1 << 127,
        0,
    ];
    let vals = &vals[..width / 8];
    kernel
        .call_words(&lane_words(vals.iter().copied(), 16), &mut out)
        .unwrap();
    assert_eq!(
        out,
        vals.iter()
            .map(|&x| (x as f64).to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn wide_width_changes_use_logical_halves() {
    let n = ScalarKernel::vector_bytes() / 8;
    let source = [
        0x0123456789abcdef,
        0x8123456789abcdef,
        0x42,
        u64::MAX,
        1 << 63,
        i64::MAX as u64,
        0xfedcba9876543210,
        0x4000000000010020,
    ];
    for (ty, signed) in [("i64", true), ("u64", false)] {
        for (op, half) in [("widenlo", 0), ("widenhi", n / 2)] {
            let f = parse(&format!(
                "fn main a:vec.{ty} -> vec.u128\n r = simd.{op} u128 a\n ret r"
            ));
            let kernel = ScalarKernel::compile(&f).unwrap();
            let mut out = vec![0; n];
            kernel.call_words(&source[..n], &mut out).unwrap();
            let expected = source[half..half + n / 2]
                .iter()
                .flat_map(|&x| {
                    [
                        x,
                        if signed && (x as i64) < 0 {
                            u64::MAX
                        } else {
                            0
                        },
                    ]
                })
                .collect::<Vec<_>>();
            assert_eq!(out, expected);
        }
    }
    let f = parse("fn main a:vec.i128 b:vec.i128 -> vec.u64\n r = simd.narrow u64 a b\n ret r");
    let kernel = ScalarKernel::compile(&f).unwrap();
    let args: Vec<_> = source[..n]
        .iter()
        .enumerate()
        .flat_map(|(i, &x)| [x, u64::MAX - i as u64])
        .collect();
    let mut out = vec![0; n];
    kernel.call_words(&args, &mut out).unwrap();
    assert_eq!(out, source[..n]);
}

#[test]
fn physical_sve_tail_does_not_extend_iota_or_reductions() {
    let width = ScalarKernel::vector_bytes();
    let f = parse("fn main -> vec.bf16\n r = simd.iota bf16\n ret r");
    let kernel = ScalarKernel::compile(&f).unwrap();
    let mut out = vec![0; width / 8];
    kernel.call_words(&[], &mut out).unwrap();
    assert_eq!(
        out,
        lane_words(
            (0..width / 2).map(|i| ((i as f32).to_bits() >> 16) as u128),
            2
        )
    );

    for (ty, input, expect) in [
        ("f16", 0x3c20, [0x3d1d, 0x3e8b, 0x4158]),
        ("bf16", 0x3f84, [0x3fa3, 0x3fd0, 0x402a]),
    ] {
        let f = parse(&format!(
            "fn main a:{ty} -> {ty}\n v = simd.splat a\n r = simd.reduce mul v\n ret r"
        ));
        let kernel = ScalarKernel::compile(&f).unwrap();
        let mut out = [0];
        kernel.call_words(&[input], &mut out).unwrap();
        assert_eq!(out[0], expect[(width / 16).trailing_zeros() as usize]);
    }

    let f = parse("fn main a:vec.f32 -> f32\n r = simd.reduce min a\n ret r");
    let kernel = ScalarKernel::compile(&f).unwrap();
    let mut input = vec![7.0_f32.to_bits() as u128; width / 4];
    *input.last_mut().unwrap() = 0x7f800001;
    let mut out = [0];
    kernel.call_words(&lane_words(input, 4), &mut out).unwrap();
    assert_eq!(out[0], 7.0_f32.to_bits() as u64);

    let f = parse("fn main a:vec.u128 -> u128\n r = simd.reduce mul a\n ret r");
    let kernel = ScalarKernel::compile(&f).unwrap();
    let mut lanes = vec![1; width / 16];
    lanes[0] = (1 << 64) | 1;
    if lanes.len() > 1 {
        lanes[1] = lanes[0];
    }
    let want = lanes.iter().copied().fold(1u128, u128::wrapping_mul);
    let mut out = [0; 2];
    kernel.call_words(&lane_words(lanes, 16), &mut out).unwrap();
    assert_eq!(out, [want as u64, (want >> 64) as u64]);
}

#[test]
fn wide_rearrangement_and_mask_tests_keep_full_pairs() {
    let n = ScalarKernel::vector_bytes() / 16;
    let a = |i| (0xa000000000001000_u128 + 3 * i as u128) << 64 | (0x1000000000000100 + i as u128);
    let b = |i| (0xb000000000002000_u128 + 5 * i as u128) << 64 | (0x2000000000000200 + i as u128);
    let mut sources = lane_words((0..n).map(a), 16);
    sources.extend(lane_words((0..n).map(b), 16));
    let shuffle = parse(
        "fn main a:vec.u128 b:vec.u128 idx:vec.u128 -> vec.u128\n r = simd.shuffle a b idx\n ret r",
    );
    let shuffle = ScalarKernel::compile(&shuffle).unwrap();
    let cases = [
        (0, a(0)),
        ((n - 1) as u128, a(n - 1)),
        (n as u128, b(0)),
        ((2 * n - 1) as u128, b(n - 1)),
        ((2 * n) as u128, 0),
        (1 << 64, 0),
        ((1 << 64) | n as u128, 0),
        (1 << 63, 0),
        (u128::MAX, 0),
    ];
    for offset in 0..cases.len() {
        let cases = cases.into_iter().cycle().skip(offset).take(n);
        let mut args = sources.clone();
        args.extend(lane_words(cases.clone().map(|(idx, _)| idx), 16));
        let mut out = vec![0; n * 2];
        shuffle.call_words(&args, &mut out).unwrap();
        assert_eq!(out, lane_words(cases.map(|(_, want)| want), 16));
    }

    let expand =
        parse("fn main a:vec.i128 mask:vec.i128 -> vec.i128\n r = simd.expand a mask\n ret r");
    let expand = ScalarKernel::compile(&expand).unwrap();
    for mask in [
        [1 << 127; 4],
        [1 << 63; 4],
        [1 << 63, 1 << 127, 1, (1 << 127) | 1],
    ] {
        let mut args = lane_words((0..n).map(a), 16);
        args.extend(lane_words(mask[..n].iter().copied(), 16));
        let mut out = vec![0; n * 2];
        expand.call_words(&args, &mut out).unwrap();
        let mut inputs = (0..n).map(a);
        let expected = mask[..n].iter().map(|&x| {
            if x >> 127 != 0 {
                inputs.next().unwrap()
            } else {
                0
            }
        });
        assert_eq!(out, lane_words(expected, 16));
    }

    for (op, all) in [("all", true), ("any", false)] {
        let f = parse(&format!(
            "fn main a:vec.u128 -> i32\n r = simd.{op} a\n ret r"
        ));
        let kernel = ScalarKernel::compile(&f).unwrap();
        for values in [
            [1, 1 << 64, 1, 1 << 127],
            [1 << 64, 1, 1 << 127, 1],
            [0, 1, 1 << 64, 1],
            [0; 4],
        ] {
            let values = &values[..n];
            let want = if all {
                values.iter().all(|&x| x != 0)
            } else {
                values.iter().any(|&x| x != 0)
            };
            let mut out = [0];
            kernel
                .call_words(&lane_words(values.iter().copied(), 16), &mut out)
                .unwrap();
            assert_eq!(out[0] as i32, if want { -1 } else { 0 });
        }
    }
}

#[test]
fn vector_integer_division_checks_every_logical_lane() {
    let width = ScalarKernel::vector_bytes();
    for (size, signed, ty) in [
        (1, true, "i8"),
        (2, true, "i16"),
        (4, true, "i32"),
        (8, true, "i64"),
        (1, false, "u8"),
        (2, false, "u16"),
        (4, false, "u32"),
        (8, false, "u64"),
    ] {
        let lanes = width / size;
        let sign = 1i128 << (size * 8 - 1);
        let lhs = if signed {
            [-7, 7, -sign, sign - 1]
        } else {
            [sign * 2 - 1, sign, 7, 42]
        };
        let rhs = if signed { [3, -3, 2, -3] } else { [3, 3, 2, 3] };
        let mut args = lane_words(lhs.into_iter().cycle().take(lanes).map(|i| i as u128), size);
        args.extend(lane_words(
            rhs.into_iter().cycle().take(lanes).map(|i| i as u128),
            size,
        ));
        for (op, is_rem) in [("div", false), ("rem", true)] {
            let f = parse(&format!(
                "fn main a:vec.{ty} b:vec.{ty} -> vec.{ty}\n r = simd.{op} a b\n ret r"
            ));
            let code = ScalarKernel::compile(&f).unwrap();
            let mut actual = vec![0; width / 8];
            code.call_words(&args, &mut actual).unwrap();
            let want = lhs.into_iter().zip(rhs).cycle().take(lanes).map(|(a, b)| {
                if is_rem {
                    (a % b) as u128
                } else {
                    (a / b) as u128
                }
            });
            assert_eq!(actual, lane_words(want, size), "{op} {ty}");
        }
    }
}

#[test]
fn narrow_division_reports_the_earliest_fault_across_groups() {
    let width = ScalarKernel::vector_bytes();
    let boundary = width / 4;
    for op in ["div", "rem"] {
        let f = parse(&format!(
            "fn main a:vec.i8 b:vec.i8 -> vec.i8\n r = simd.{op} a b\n ret r"
        ));
        let code = ScalarKernel::compile(&f).unwrap();
        for (first, second, error) in [
            (0, 255, CallError::Trap(crate::Trap::DivisionByZero)),
            (255, 0, CallError::Trap(crate::Trap::DivisionOverflow)),
        ] {
            let mut rhs = vec![2; width];
            rhs[boundary - 1] = first;
            rhs[boundary] = second;
            let mut args = lane_words(std::iter::repeat_n(128, width), 1);
            args.extend(lane_words(rhs, 1));
            assert_eq!(code.call_words(&args, &mut vec![0; width / 8]), Err(error));
        }
    }
}

#[test]
fn compress_keeps_order_and_zeros_the_full_tail() {
    let width = ScalarKernel::vector_bytes();
    for (size, ty) in [(1, "u8"), (2, "u16"), (4, "u32"), (8, "u64"), (16, "u128")] {
        let n = width / size;
        let sign = 1u128 << (size * 8 - 1);
        let f = parse(&format!(
            "fn main a:vec.{ty} m:vec.{ty} -> vec.{ty}\n r = simd.compress a m\n ret r"
        ));
        let code = ScalarKernel::compile(&f).unwrap();
        let data: Vec<_> = (0..n).map(|i| sign | (3 * i + 1) as u128).collect();
        let sparse = (1u64 << (n - 1)) | (1 << (n.saturating_sub(2))) | (1 << (n / 4)) | 1;
        for bits in [0, u64::MAX, 1, 1 << (n - 1), sparse, 0x0070_0093_0100_8421] {
            let enabled = |i| bits & (1u64 << i) != 0;
            let mut args = lane_words(data.iter().copied(), size);
            args.extend(lane_words(
                (0..n).map(|i| (if enabled(i) { sign } else { sign >> 1 }) | 1),
                size,
            ));
            let mut want: Vec<_> = data
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(i, value)| enabled(i).then_some(value))
                .collect();
            want.resize(n, 0);
            let mut actual = vec![u64::MAX; width / 8];
            code.call_words(&args, &mut actual).unwrap();
            assert_eq!(actual, lane_words(want, size), "{ty}, mask {bits:#x}");
        }
    }
}

#[test]
fn wide_indexed_memory_uses_both_words_in_lane_order() {
    let n = ScalarKernel::vector_bytes() / 16;
    let join = |low: u64, high: u64| u128::from(low) | (u128::from(high) << 64);
    let original = [
        join(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210),
        join(0x45, 0xb8),
        join(0x1020_3040_5060_7080, 0x8877_6655_4433_2211),
    ];
    let active = 1u128 << 127;
    let inactive = join(1 << 63, 1);
    let indices = [
        join(u64::MAX, 0x1234_5678),
        join(1 << 40, 0),
        join((1 << 60) + 1, 0x8765_4321),
        u128::MAX,
    ];
    let flags = [active | 21, inactive, active | 1, active];
    let pass = [join(17, 33), join(18, 34), join(19, 35), join(20, 36)];
    let expected = [original[0], pass[1], original[2], original[0]];
    let load = ScalarKernel::compile(&parse(
        "fn main p:ptr i:vec.i128 m:vec.u128 a:vec.u128 -> vec.u128\n r = simd.maskedload p i m a\n ret r"
    )).unwrap();
    for rotate in 0..indices.len() {
        let words =
            |input: [u128; 4]| lane_words(input.into_iter().cycle().skip(rotate).take(n), 16);
        let mut args = vec![original.as_ptr().wrapping_add(1) as u64];
        for values in [indices, flags, pass] {
            args.extend(words(values));
        }
        let mut actual = vec![0; 2 * n];
        // SAFETY: the middle cell has both neighbors live and every other address is masked out.
        unsafe { load.call_words_raw(&args, &mut actual) }.unwrap();
        assert_eq!(actual, words(expected));
    }

    let store = ScalarKernel::compile(&parse(
        "fn main p:ptr i:vec.i128 m:vec.u128 a:vec.u128 -> i32\n simd.maskedstore p i m a\n z = const 0\n ret z"
    )).unwrap();
    for count in [n, n - 1] {
        let mut data = original;
        let mut args = vec![data.as_mut_ptr().wrapping_add(1) as u64];
        args.extend(lane_words(
            (0..n).map(|i| join(((i as u64) << 60).wrapping_sub(1), i as u64 + 1)),
            16,
        ));
        args.extend(lane_words(
            (0..n).map(|i| if i < count { active } else { inactive }),
            16,
        ));
        args.extend(lane_words(pass[..n].iter().copied(), 16));
        // SAFETY: every active index wraps to the first live mutable cell.
        unsafe { store.call_words_raw(&args, &mut [0]) }.unwrap();
        let mut expected = original;
        if count != 0 {
            expected[0] = pass[count - 1];
        }
        assert_eq!(data, expected);
    }
}

#[test]
fn wide_integer_operations_match_rust_for_both_words() {
    let n = ScalarKernel::vector_bytes() / 16;
    let hi = 1u128 << 64;
    let unsigned = [
        (hi + 1, hi + 1),
        (hi - 1, hi - 1),
        (u128::MAX, (1 << 127) + 1),
        (5 * hi + 7, hi),
        (((1 << 80) + hi + 123) * 5 + 4, 5),
        (((hi + 3) * (hi - 16)) + hi + 1, hi + 3),
        (1 << 127, u128::MAX),
        (hi + (1 << 63), hi),
    ];
    let signed = [
        (i128::MIN, 1),
        (i128::MIN, -2),
        (i128::MIN, 3),
        (i128::MIN + 1, -1),
        (-7, 3),
        (7, -3),
        (i128::MIN + u64::MAX as i128, i128::MAX),
        ((hi + (1 << 63)) as i128, hi as i128),
    ];
    for ty in ["i128", "u128"] {
        for op in [
            "mul", "div", "rem", "lt", "le", "eq", "ne", "ge", "gt", "min", "max",
        ] {
            let f = parse(&format!(
                "fn main a:vec.{ty} b:vec.{ty} -> vec.{ty}\n r = simd.{op} a b\n ret r"
            ));
            let kernel = ScalarKernel::compile(&f).unwrap();
            let pairs = if ty == "i128" {
                signed.map(|(a, b)| (a as u128, b as u128))
            } else {
                unsigned
            };
            for pairs in pairs.chunks(n) {
                let mut args = lane_words(pairs.iter().map(|&(a, _)| a), 16);
                args.extend(lane_words(pairs.iter().map(|&(_, b)| b), 16));
                let want = pairs.iter().map(|&(a, b)| {
                    let (sa, sb) = (a as i128, b as i128);
                    let order = if ty == "i128" { sa.cmp(&sb) } else { a.cmp(&b) };
                    match op {
                        "mul" => a.wrapping_mul(b),
                        "div" if ty == "i128" => (sa / sb) as u128,
                        "div" => a / b,
                        "rem" if ty == "i128" => (sa % sb) as u128,
                        "rem" => a % b,
                        "min" => {
                            if order.is_lt() {
                                a
                            } else {
                                b
                            }
                        }
                        "max" => {
                            if order.is_gt() {
                                a
                            } else {
                                b
                            }
                        }
                        _ => {
                            let mask = match op {
                                "lt" => order.is_lt(),
                                "le" => order.is_le(),
                                "eq" => order.is_eq(),
                                "ne" => order.is_ne(),
                                "ge" => order.is_ge(),
                                _ => order.is_gt(),
                            };
                            0u128.wrapping_sub(u128::from(mask))
                        }
                    }
                });
                let mut actual = vec![0; 2 * n];
                kernel.call_words(&args, &mut actual).unwrap();
                assert_eq!(actual, lane_words(want, 16), "{ty} {op}");
            }
        }
    }
}

#[test]
fn wide_shifts_saturation_and_counts_use_the_entire_pair() {
    let n = ScalarKernel::vector_bytes() / 16;
    let a = 0xfedc_ba98_7654_3210_89ab_cdef_0123_4567u128;
    let shifts = [
        0u128,
        1,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        u64::MAX as u128,
        2,
        126,
    ];
    for (op, ty, evaluate) in [
        ("shl", "u128", u128::wrapping_shl as fn(u128, u32) -> u128),
        ("shr", "u128", u128::wrapping_shr),
        ("shr", "i128", |a, n| (a as i128).wrapping_shr(n) as u128),
        ("rotateleft", "u128", u128::rotate_left),
        ("rotateright", "u128", u128::rotate_right),
    ] {
        let f = parse(&format!(
            "fn main a:vec.{ty} b:vec.{ty} -> vec.{ty}\n r = simd.{op} a b\n ret r"
        ));
        let kernel = ScalarKernel::compile(&f).unwrap();
        for counts in shifts.chunks(n) {
            let mut args = lane_words(std::iter::repeat_n(a, n), 16);
            args.extend(lane_words(counts.iter().map(|&i| i | (1 << 124)), 16));
            let mut actual = vec![0; 2 * n];
            kernel.call_words(&args, &mut actual).unwrap();
            assert_eq!(
                actual,
                lane_words(counts.iter().map(|&i| evaluate(a, i as u32)), 16),
                "{op} {ty}"
            );
        }
    }

    let signed = [
        (i128::MAX, 1),
        (i128::MIN, 1),
        (i128::MIN + u64::MAX as i128, -1),
        (i128::MAX - u64::MAX as i128, -1),
    ];
    let unsigned = [
        (u64::MAX as u128, 1),
        (1u128 << 64, 1),
        (u128::MAX, 1),
        (1, 1 << 64),
    ];
    for ty in ["i128", "u128"] {
        for (op, sub) in [("saturatingadd", false), ("saturatingsub", true)] {
            let f = parse(&format!(
                "fn main a:vec.{ty} b:vec.{ty} -> vec.{ty}\n r = simd.{op} a b\n ret r"
            ));
            let kernel = ScalarKernel::compile(&f).unwrap();
            let pairs = if ty == "i128" {
                signed.map(|(a, b)| (a as u128, b as u128))
            } else {
                unsigned
            };
            for pairs in pairs.chunks(n) {
                let mut args = lane_words(pairs.iter().map(|&(a, _)| a), 16);
                args.extend(lane_words(pairs.iter().map(|&(_, b)| b), 16));
                let want = pairs.iter().map(|&(a, b)| match (ty, sub) {
                    ("i128", true) => (a as i128).saturating_sub(b as i128) as u128,
                    ("i128", false) => (a as i128).saturating_add(b as i128) as u128,
                    (_, true) => a.saturating_sub(b),
                    (_, false) => a.saturating_add(b),
                });
                let mut actual = vec![0; 2 * n];
                kernel.call_words(&args, &mut actual).unwrap();
                assert_eq!(actual, lane_words(want, 16), "{ty} {op}");
            }
        }
    }

    let boundary = [
        0u128,
        1,
        1 << 63,
        1 << 64,
        1 << 127,
        (1 << 127) + 1,
        u128::MAX,
        a,
    ];
    for (op, evaluate) in [
        ("neg", u128::wrapping_neg as fn(u128) -> u128),
        ("abs", |a| (a as i128).wrapping_abs() as u128),
        ("popcnt", |a| a.count_ones() as u128),
        ("leadingzeros", |a| a.leading_zeros() as u128),
        ("trailingzeros", |a| a.trailing_zeros() as u128),
    ] {
        let f = parse(&format!(
            "fn main a:vec.i128 -> vec.i128\n r = simd.{op} a\n ret r"
        ));
        let kernel = ScalarKernel::compile(&f).unwrap();
        for input in boundary.chunks(n) {
            let mut actual = vec![u64::MAX; 2 * n];
            kernel
                .call_words(&lane_words(input.iter().copied(), 16), &mut actual)
                .unwrap();
            assert_eq!(
                actual,
                lane_words(input.iter().copied().map(evaluate), 16),
                "{op}"
            );
        }
    }
}

#[test]
fn packed_division_preserves_the_live_left_operand() {
    let f = parse(
        "
        fn quotient a:vec.f64 b:vec.f64 -> vec.f64
          q = simd.div a b
          ret q

        fn main a:vec.f64 b:vec.f64 -> vec.f64
          q = call quotient a b
          r = simd.sub a q
          ret r
    ",
    );
    let lanes = ScalarKernel::vector_bytes() / 8;
    let lhs = [6.0_f64, -10.0, 4.0, -0.0, 1.8, -40.5, 1.5, 1.0];
    let rhs = [2.0_f64, 4.0, -2.0, -2.0, 2.0, 4.5, -2.0, 3.0];
    let args: Vec<_> = lhs[..lanes]
        .iter()
        .chain(&rhs[..lanes])
        .map(|x| x.to_bits())
        .collect();
    let expected: Vec<_> = lhs[..lanes]
        .iter()
        .zip(&rhs)
        .map(|(a, b)| (a - a / b).to_bits())
        .collect();
    for policy in [Inlining::Always, Inlining::Never] {
        let jit = ScalarKernel::compile_with_inlining(&f, policy).unwrap();
        let mut actual = vec![0; lanes];
        jit.call_words(&args, &mut actual).unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn floating_and_wide_integer_casts_round_and_saturate_at_target_width() {
    let midpoint_32 = (1_u128 << 100) + (1 << 76);
    let midpoint_64 = (1_u128 << 100) + (1 << 63) + (1 << 47);
    let midpoint_bf16 = (1_u128 << 100) + (1 << 92);
    let unsigned_values = [
        0,
        1,
        1 << 63,
        1 << 64,
        midpoint_32 - 1,
        midpoint_32,
        midpoint_32 + 1,
        midpoint_64 - 1,
        midpoint_64,
        midpoint_64 + 1,
        u128::MAX,
    ];
    let fp_values = [
        0x43efffffffffffff,
        0x43f0000000000000,
        0x43f0000000000001,
        0xc3f0000000000001,
        0x47dfffffffffffff,
        0x47e0000000000000,
        0xc7dfffffffffffff,
        0xc7e0000000000000,
        0x47efffffffffffff,
        0x47f0000000000000,
        0xfff0000000000001,
        0x7ff8000000000000,
        0xfff0000000000000,
        0x7ff0000000000000,
        0x3fefffffffffffff,
        0xbfefffffffffffff,
    ];
    for (from, to, input, expected) in [
        (
            "u128",
            "f32",
            unsigned_values.to_vec(),
            unsigned_values
                .map(|x| (x as f32).to_bits() as u128)
                .to_vec(),
        ),
        (
            "u128",
            "f64",
            unsigned_values.to_vec(),
            unsigned_values
                .map(|x| (x as f64).to_bits() as u128)
                .to_vec(),
        ),
        (
            "i128",
            "f64",
            vec![i128::MIN as u128, i128::MAX as u128, (-1_i128) as u128],
            vec![0xc7e0000000000000, 0x47e0000000000000, 0xbff0000000000000],
        ),
        (
            "u128",
            "bf16",
            vec![
                midpoint_bf16 - 1,
                midpoint_bf16,
                midpoint_bf16 + 1,
                u128::MAX,
            ],
            vec![0x7180, 0x7180, 0x7181, 0x7f80],
        ),
        (
            "u128",
            "f16",
            vec![2049, 2051, 65519, 65520, u128::MAX],
            vec![0x6800, 0x6802, 0x7bff, 0x7c00, 0x7c00],
        ),
        (
            "f64",
            "u128",
            fp_values.map(u128::from).to_vec(),
            fp_values.map(|x| f64::from_bits(x) as u128).to_vec(),
        ),
        (
            "f64",
            "i128",
            fp_values.map(u128::from).to_vec(),
            fp_values
                .map(|x| f64::from_bits(x) as i128 as u128)
                .to_vec(),
        ),
        (
            "f32",
            "u128",
            vec![0x7f7fffff, 0x7f800000, 0xff800000],
            vec![u128::MAX - ((1 << 104) - 1), u128::MAX, 0],
        ),
        (
            "bf16",
            "u128",
            vec![0x7f7f, 0xff00, 0x7f80, 0x8001],
            vec![u128::MAX - ((1 << 120) - 1), 0, u128::MAX, 0],
        ),
        (
            "f16",
            "i128",
            vec![0x3e00, 0xbe00, 0xfc00, 0xfe09],
            vec![1, u128::MAX, i128::MIN as u128, 0],
        ),
    ] {
        let f = parse(&format!(
            "fn main a:{from} -> {to}\n r = cast {to} a\n ret r"
        ));
        let kernel = ScalarKernel::compile(&f).unwrap();
        for (bits, want) in input.into_iter().zip(expected) {
            let args = [bits as u64, (bits >> 64) as u64];
            let args = &args[..if from.ends_with("128") { 2 } else { 1 }];
            let mut output = [0; 2];
            let out = &mut output[..if to.ends_with("128") { 2 } else { 1 }];
            kernel.call_words(args, out).unwrap();
            assert_eq!(
                u128::from(output[0]) | (u128::from(output[1]) << 64),
                want,
                "{from} → {to}, {bits:#x}"
            );
        }
    }
}

#[test]
fn wide_root_return_and_outline_preserve_all_words() {
    let f = parse(
        "
        fn twice v:vec.u64 -> vec.u64
          result = simd.add v v
          ret result

        fn main a:vec.u64 -> vec.u64
          b = call twice a
          ret b
    ",
    );
    let args: Vec<_> = (0..ScalarKernel::vector_bytes() / 8)
        .map(|i| u64::MAX - i as u64)
        .collect();
    let expected: Vec<_> = args.iter().map(|v| v.wrapping_mul(2)).collect();
    for policy in [Inlining::Always, Inlining::Never] {
        let code = ScalarKernel::compile_with_inlining(&f, policy).unwrap();
        let mut result = vec![0; args.len()];
        code.call_words(&args, &mut result).unwrap();
        assert_eq!(result, expected);
    }
}

#[test]
fn wide_arithmetic_and_iota_preserve_lane_pairs() {
    let lanes = ScalarKernel::vector_bytes() / 16;
    let lhs = [u128::MAX, 0x1111 << 64, 0, (11 << 64) | 9][..lanes].to_vec();
    let rhs = [1u128, 2, 1, (13 << 64) | 7][..lanes].to_vec();
    let words = |value: u128| [value as u64, (value >> 64) as u64];
    let args: Vec<_> = lhs.iter().chain(&rhs).copied().flat_map(words).collect();
    for (op, scalar) in [
        ("add", u128::wrapping_add as fn(u128, u128) -> u128),
        ("sub", u128::wrapping_sub),
    ] {
        let f = parse(&format!(
            "fn main a:vec.u128 b:vec.u128 -> vec.u128\n r = simd.{op} a b\n ret r"
        ));
        let jit = ScalarKernel::compile(&f).unwrap();
        let mut actual = vec![0; 2 * lanes];
        jit.call_words(&args, &mut actual).unwrap();
        let expected: Vec<_> = lhs
            .iter()
            .zip(&rhs)
            .flat_map(|(&a, &b)| words(scalar(a, b)))
            .collect();
        assert_eq!(actual, expected, "{op}");
    }
    let f = parse("fn main -> vec.u128\n r = simd.iota u128\n ret r");
    let mut actual = vec![u64::MAX; 2 * lanes];
    ScalarKernel::compile(&f)
        .unwrap()
        .call_words(&[], &mut actual)
        .unwrap();
    assert_eq!(
        actual,
        (0..lanes as u128).flat_map(words).collect::<Vec<_>>()
    );
}

#[test]
fn sve_logical_width_and_shorter_thread_are_checked() {
    let config = splice_target::simd().unwrap();
    let lanes = config.bytes / 4;
    let f = parse(
        "
        fn main dst:ptr src:ptr -> i32
          input = simd.load i32 src
          doubled = simd.add input input
          simd.store dst doubled
          lanes = simd.lanes i32
          ret lanes
    ",
    );
    let code = ScalarKernel::compile(&f).unwrap();
    let original: Vec<_> = (0_i32..64).map(|i| i.wrapping_mul(123456789)).collect();
    let mut output = vec![-1i32; original.len()];
    // SAFETY: live disjoint arrays each contain at least the largest SVE register.
    let result = unsafe { code.call_raw(&[output.as_mut_ptr() as u64, original.as_ptr() as u64]) };
    assert_eq!(result, Ok(lanes as u64));
    assert_eq!(
        output[..lanes],
        original[..lanes]
            .iter()
            .map(|v| v.wrapping_mul(2))
            .collect::<Vec<_>>()
    );
    assert!(output[lanes..].iter().all(|&v| v == -1));

    if config.bytes > 16 && matches!(config.profile, Profile::Sve | Profile::Sve2) {
        std::thread::spawn(move || {
            unsafe extern "C" {
                fn prctl(option: i32, ...) -> i32;
            }
            // SAFETY: Linux PR_SVE_GET_VL takes four unused unsigned long arguments.
            let initial = unsafe { prctl(51, 0usize, 0usize, 0usize, 0usize) };
            assert!(initial >= 0);
            let initial = (initial & 0xffff) as usize;
            // SAFETY: Linux PR_SVE_SET_VL takes unsigned long arguments and
            // changes only the vector length of this test thread.
            let vl = unsafe { prctl(50, 16usize, 0usize, 0usize, 0usize) };
            assert_eq!(vl, 16);
            output.fill(-1);
            // SAFETY: both arrays remain live for the call and cover the original width.
            let result =
                unsafe { code.call_raw(&[output.as_mut_ptr() as u64, original.as_ptr() as u64]) };
            assert_eq!(
                result,
                Err(CallError::Target(
                    splice_target::TargetError::VectorLength {
                        required: config.bytes,
                        available: 16
                    }
                ))
            );
            assert!(output.iter().all(|&v| v == -1));
            // SAFETY: use the length read from this thread before reducing its VL.
            let restored = unsafe { prctl(50, initial, 0usize, 0usize, 0usize) };
            assert_eq!(restored, initial as i32);
            // SAFETY: the same disjoint, live arrays cover the configured width.
            let result =
                unsafe { code.call_raw(&[output.as_mut_ptr() as u64, original.as_ptr() as u64]) };
            assert_eq!(result, Ok(lanes as u64));
            assert!(
                output[..lanes]
                    .iter()
                    .zip(&original)
                    .all(|(&out, input)| out == input.wrapping_mul(2))
            );
            assert!(output[lanes..].iter().all(|&v| v == -1));
        })
        .join()
        .unwrap();
    }
}
