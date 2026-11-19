//! File-driven tests for the actual Lancy passes. See lancy/filetests/README.md.
mod input;
mod listing;
mod run;

use crate::hir::{Function, ssa::VerifiedBody};
use crate::lir::{
    regalloc,
    x64::{
        compile::{compile, written_regs},
        emit::emit_compiled,
        isel,
    },
};
use crate::module::{FunctionIds, Module};
use crate::{CompileError, Inlining, ScalarKernel};
use filecheck::{CheckerBuilder, NO_VARIABLES};
use run::Run;
use std::{collections::BTreeMap, sync::Arc};

const STAGES: [&str; 7] = ["source", "cfg", "inline", "opt", "isel", "alloc", "asm"];
type Result<T> = std::result::Result<T, String>;

#[derive(Default)]
struct Case<'a> {
    ir: input::Lines<'a>,
    checks: BTreeMap<&'a str, (usize, CheckerBuilder)>,
    runs: Vec<Run>,
    policy: Inlining,
    reject: Option<&'a str>,
}

fn integer(s: &str) -> Result<i32> {
    let s = s.replace('_', "");
    if let Some(hex) = s.strip_prefix("0x") {
        u32::from_str_radix(hex, 16).map(|i| i as i32)
    } else {
        s.parse()
    }
    .map_err(|_| format!("expected i32, got {s}"))
}

impl<'a> Case<'a> {
    fn line(&mut self, n: usize, line: &'a str) -> Result<()> {
        let Some(directive) = line.strip_prefix('@') else {
            self.ir.push((n, line));
            return Ok(());
        };
        let (stage, text) = directive
            .split_once(char::is_whitespace)
            .ok_or("empty directive")?;
        let text = text.trim();
        match stage {
            "policy" => {
                self.policy = match text {
                    "always" => Inlining::Always,
                    "never" => Inlining::Never,
                    _ => return Err(format!("unknown policy {text}")),
                }
            }
            "reject" if !text.is_empty() && self.reject.is_none() => self.reject = Some(text),
            "run" => self.runs.push(Run::parse(n, text)?),
            stage if STAGES.contains(&stage) => {
                let (_, builder) = self
                    .checks
                    .entry(stage)
                    .or_insert_with(|| (n, CheckerBuilder::new()));
                if !builder.directive(text).map_err(|e| e.to_string())? {
                    return Err(format!("unknown FileCheck directive {text}"));
                }
            }
            _ => return Err(format!("unknown directive @{stage}")),
        }
        Ok(())
    }

    fn check(&mut self, stage: &str, text: impl FnOnce() -> String) -> Result<()> {
        if let Some((line, mut builder)) = self.checks.remove(stage) {
            let listing = text();
            if std::env::var_os("LANCY_FILETEST_DUMP").is_some() {
                eprintln!("[{stage}]\n{listing}");
            }
            let checker = builder.finish();
            let (ok, explanation) = checker
                .explain(&listing, NO_VARIABLES)
                .map_err(|e| format!("{stage}: {e}"))?;
            if !ok {
                return Err(format!(
                    "{stage} (checks from line {line}):\n{checker}\n{explanation}"
                ));
            }
        }
        Ok(())
    }

    fn interpret(&self, stage: &str, f: &Function) -> Result<()> {
        for run in &self.runs {
            run.interpret(stage, f)?;
        }
        Ok(())
    }

    fn execute(mut self) -> Result<()> {
        let f = input::parse(&self.ir)?;
        let verified = f.verified_module();
        if let Some(reason) = self.reject {
            if !self.checks.is_empty() || !self.runs.is_empty() {
                return Err("@reject requires a verifier-only case".into());
            }
            return match verified {
                Err(CompileError::InvalidIr(e)) if e.to_string().contains(reason) => Ok(()),
                Err(e) => Err(format!("verify: expected '{reason}', got {e}")),
                Ok(_) => Err(format!("verify: accepted IR, expected '{reason}'")),
            };
        }
        if self.checks.is_empty() && self.runs.is_empty() {
            return Err("case has no expectations".into());
        }
        let (source, ids) = verified.map_err(|e| format!("verify: {e}"))?;
        self.check("source", || listing::hir(&source, &ids))?;
        self.check("cfg", || listing::cfg(&source))?;
        self.interpret("source", &f)?;
        if self.checks.is_empty() && self.runs.is_empty() {
            return Ok(());
        }
        let expanded = if self.policy == Inlining::Always && source.functions.len() > 1 {
            Arc::new(
                source
                    .inline_all(&ids)
                    .map_err(|e| format!("inline: {e}"))?,
            )
        } else {
            Arc::clone(&f)
        };
        let (module, ids) = expanded
            .verified_module()
            .map_err(|e| format!("inline: {e}"))?;
        self.check("inline", || listing::hir(&module, &ids))?;
        self.interpret("inline", &expanded)?;
        if self.checks.is_empty() && self.runs.is_empty() {
            return Ok(());
        }
        let opt = module
            .simplify_bodies(&ids)
            .map_err(|e| format!("opt: {e}"))?;
        let (module, ids) = opt.verified_module().map_err(|e| format!("opt: {e}"))?;
        self.check("opt", || listing::hir(&module, &ids))?;
        self.interpret("opt", &opt)?;
        if self.checks.contains_key("isel") {
            let text = selected(&module, &ids).map_err(|e| format!("isel: {e}"))?;
            self.check("isel", || text)?;
        }
        if self.checks.contains_key("alloc") || self.checks.contains_key("asm") {
            let module = compile(module, &ids).map_err(|e| format!("alloc: {e}"))?;
            self.check("alloc", || {
                let mut out = String::new();
                for (id, f) in module.functions.iter() {
                    listing::lir(&mut out, id, &f.body, Some(&f.allocation));
                }
                out
            })?;
            if self.checks.contains_key("asm") {
                let code = emit_compiled(&module).map_err(|e| format!("asm: {e}"))?;
                let asm = listing::asm(&code.bytes[..code.text_len])?;
                self.check("asm", || asm)?;
            }
        }
        if !self.runs.is_empty() {
            let kernel = ScalarKernel::compile_with_inlining(&f, self.policy)
                .map_err(|e| format!("JIT: {e}"))?;
            for run in self.runs {
                // SAFETY: these inputs passed checked interpretation on f above;
                // the kernel was compiled from f and remains live across the call.
                unsafe {
                    run.native(&kernel)?;
                }
            }
        }
        Ok(())
    }
}

// ISel does not depend on the root allocation. For outlined calls it needs
// the exact transitive clobber set of each already-allocated child.
fn selected(
    module: &Module<VerifiedBody>,
    ids: &FunctionIds<Function>,
) -> std::result::Result<String, CompileError> {
    let mut out = String::new();
    let mut clobbers = crate::support::slotmap::DenseMap::new();
    let mut ctx = regalloc2::Ctx::default();
    for (id, src) in module.functions.iter() {
        let root = id == module.call_graph.root();
        let f = isel::select(src.function, &src.cfg, root, |callee| {
            let id = ids[&std::ptr::from_ref(callee)];
            (id, clobbers[id])
        })?;
        listing::lir(&mut out, id, &f, None);
        if !root {
            clobbers.insert(written_regs(&f, &regalloc::allocate(&f, &mut ctx)?).0);
        }
    }
    Ok(out)
}

fn run_text(text: &str) -> Result<()> {
    let mut case = Case::default();
    let mut start = 1;
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "---" {
            std::mem::take(&mut case)
                .execute()
                .map_err(|e| format!("case at line {start}: {e}"))?;
            start = i + 2;
        } else {
            case.line(i + 1, line)
                .map_err(|e| format!("line {}: {e}", i + 1))?;
        }
    }
    case.execute()
        .map_err(|e| format!("case at line {start}: {e}"))
}

fn fixture(file: &str, text: &str) {
    if let Err(error) = run_text(text) {
        panic!("{file}: {error}");
    }
}

include!(concat!(env!("OUT_DIR"), "/filetests.rs"));

#[test]
fn mismatches_identify_the_stage_even_when_native_execution_would_succeed() {
    for stage in STAGES {
        let source =
            format!("fn main x:i32 -> i32\nret x\n@{stage} check: absent_opcode\n@run 8 -> 8");
        let error = run_text(&source).unwrap_err();
        assert!(
            error.contains(&format!("{stage} (checks from line 3)")),
            "{error}"
        );
        assert!(error.contains("absent_opcode"), "{error}");
    }
}

#[test]
fn invalid_filetest_memory_is_rejected_by_checked_interpretation() {
    let source = "fn main p:ptr -> i32\nx = load i32 p\nret x\n@run -> 0 | 00 | 00";
    let error = run_text(source).unwrap_err();
    assert!(
        error.contains("source") && error.contains("access outside buffer"),
        "{error}"
    );
}
