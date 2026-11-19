use crate::{CallError, CompileError, Inlining, ScalarKernel, hir::Function};
use crate::{
    filetests::{Result, Run, input, integer},
    hir::transforms::passes,
};
use filecheck::{CheckerBuilder, NO_VARIABLES};
use regalloc2::{Block, Function as _, InstOrEdit};
use splice_target::Profile;
use std::fmt::Write;

use super::compile::Compiled;

fn fixture(file: &str, text: &str) {
    let lines: Vec<_> = text.lines().collect();
    for case in lines.split(|line| line.trim() == "---") {
        check(case).unwrap_or_else(|error| panic!("{file}: {error}"));
    }
}

fn check(lines: &[&str]) -> Result<()> {
    let mut ir = input::Lines::new();
    let mut runs = Vec::new();
    let mut directives = Vec::new();
    let mut selection = Vec::new();
    let mut allocation = Vec::new();
    let mut reject = None;
    let mut policy = Inlining::Always;
    for (i, line) in lines.iter().map(|line| line.trim()).enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(run) = line.strip_prefix("@run ") {
            runs.push(Run::parse(i + 1, run)?);
        } else if let Some(reason) = line.strip_prefix("@reject ") {
            reject = Some(reason);
        } else if let Some(assertion) = line.strip_prefix("@aarch64 ") {
            directives.push(assertion);
        } else if let Some(assertion) = line.strip_prefix("@aarch64-isel ") {
            selection.push(assertion);
        } else if let Some(assertion) = line.strip_prefix("@aarch64-alloc ") {
            allocation.push(assertion);
        } else if line == "@policy never" {
            policy = Inlining::Never;
        } else if !line.starts_with('@') {
            ir.push((i + 1, line));
        }
    }
    if runs.is_empty()
        && directives.is_empty()
        && selection.is_empty()
        && allocation.is_empty()
        && reject.is_none()
    {
        return Ok(());
    }
    let f = input::parse(&ir)?;
    let verified = f.verified_module();
    if let Some(reason) = reject {
        return match verified {
            Err(CompileError::InvalidIr(e)) if e.to_string().contains(reason) => Ok(()),
            _ => Err(format!("expected verifier error: {reason}")),
        };
    }
    let (module, ids) = verified.map_err(|e| e.to_string())?;
    for run in &runs {
        run.interpret("source", &f)?;
    }
    if !directives.is_empty() || !selection.is_empty() || !allocation.is_empty() {
        let simd = splice_target::simd()?;
        let compiled = passes::optimize_and_lower(&module, &ids, policy, |module, ids| {
            super::compile(module, ids, simd)
        })
        .map_err(|e| e.to_string())?;
        let root = &compiled.functions[compiled.call_graph.root()];
        check_listing("isel", &selection, || selected(root))?;
        check_listing("alloc", &allocation, || allocated(root))?;
        let code = super::emit_compiled(&compiled, simd).map_err(|e| e.to_string())?;
        for assertion in directives {
            let parts: Vec<_> = assertion.split_whitespace().collect();
            let [profile, presence, mask, opcode] = parts.as_slice() else {
                return Err(format!("invalid AArch64 assertion: {assertion}"));
            };
            if !matches!(*profile, "all" | "neon" | "sve" | "sve2")
                || !matches!(*presence, "has" | "lacks")
            {
                return Err(format!("invalid AArch64 assertion: {assertion}"));
            }
            let select = match simd.profile {
                Profile::Neon => "neon",
                Profile::Sve => "sve",
                Profile::Sve2 => "sve2",
                _ => unreachable!(),
            };
            if *profile != "all" && *profile != select {
                continue;
            }
            let mask = integer(mask)? as u32;
            let opcode = integer(opcode)? as u32;
            let found = code
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .any(|word| u32::from_le_bytes(*word) & mask == opcode);
            if found != (*presence == "has") {
                return Err(format!("failed AArch64 assertion: {assertion}"));
            }
        }
    }
    let kernel = ScalarKernel::compile_with_inlining(&f, policy).map_err(|e| e.to_string())?;
    for run in runs {
        // SAFETY: checked interpretation above validates every fixture access.
        unsafe { run.native(&kernel)? };
    }
    Ok(())
}

fn check_listing(stage: &str, directives: &[&str], listing: impl FnOnce() -> String) -> Result<()> {
    if directives.is_empty() {
        return Ok(());
    }
    let mut builder = CheckerBuilder::new();
    for directive in directives {
        if !builder.directive(directive).map_err(|e| e.to_string())? {
            return Err(format!("unknown FileCheck directive {directive}"));
        }
    }
    let listing = listing();
    if std::env::var_os("LANCY_FILETEST_DUMP").is_some() {
        eprintln!("[aarch64-{stage}]\n{listing}");
    }
    let (ok, explanation) = builder
        .finish()
        .explain(&listing, NO_VARIABLES)
        .map_err(|e| e.to_string())?;
    if ok {
        Ok(())
    } else {
        Err(format!("aarch64-{stage}:\n{explanation}"))
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

include!(concat!(env!("OUT_DIR"), "/filetests.rs"));

fn parse(text: &str) -> std::sync::Arc<Function> {
    let lines: input::Lines<'_> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
        .collect();
    input::parse(&lines).unwrap()
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
            assert_eq!(result, Err(CallError::VectorLength));
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
