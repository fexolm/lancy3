//! Parse and check a fixture against the HIR passes and the host JIT.
pub(crate) mod input;
pub(crate) mod listing;
mod run;

use crate::hir::{Function, ssa::VerifiedBody, transforms::passes};
use crate::module::{FunctionIds, Module};
use crate::{CompileError, Inlining, ScalarKernel, jit, lir};
use filecheck::{CheckerBuilder, NO_VARIABLES};
use run::Run;
use std::collections::BTreeMap;

pub(crate) type Result<T> = std::result::Result<T, String>;

pub(crate) fn integer(s: &str) -> Result<i32> {
    let s = s.replace('_', "");
    if let Some(hex) = s.strip_prefix("0x") {
        u32::from_str_radix(hex, 16).map(|i| i as i32)
    } else {
        s.parse()
    }
    .map_err(|_| format!("expected i32, got {s}"))
}

const COMMON_STAGES: [&str; 5] = ["source", "cfg", "inline", "opt", "layout"];
const X64_STAGES: [&str; 3] = ["isel", "alloc", "asm"];
const ARM_STAGES: [&str; 2] = ["aarch64-isel", "aarch64-alloc"];

#[derive(Default)]
pub(crate) struct Case<'a> {
    ir: input::Lines<'a>,
    pub(crate) checks: BTreeMap<&'a str, (Vec<usize>, CheckerBuilder)>,
    pub(crate) encodings: Vec<(usize, &'a str)>,
    runs: Vec<Run>,
    policy: Inlining,
    reject: Option<(usize, &'a str)>,
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
            "reject" if !text.is_empty() && self.reject.is_none() => self.reject = Some((n, text)),
            "run" => self.runs.push(Run::parse(n, text)?),
            "aarch64" => self.encodings.push((n, text)),
            stage
                if COMMON_STAGES.contains(&stage)
                    || X64_STAGES.contains(&stage)
                    || ARM_STAGES.contains(&stage) =>
            {
                let (lines, builder) = self
                    .checks
                    .entry(stage)
                    .or_insert_with(|| (Vec::new(), CheckerBuilder::new()));
                if !builder.directive(text).map_err(|e| e.to_string())? {
                    return Err(format!("unknown FileCheck directive {text}"));
                }
                lines.push(n);
            }
            _ => return Err(format!("unknown directive @{stage}")),
        }
        Ok(())
    }

    pub(crate) fn check(&mut self, stage: &str, text: impl FnOnce() -> String) -> Result<()> {
        if let Some((lines, mut builder)) = self.checks.remove(stage) {
            let listing = text();
            if std::env::var_os("LANCY_FILETEST_DUMP").is_some() {
                eprintln!("[{stage}]\n{listing}");
            }
            let checker = builder.finish();
            let (ok, explanation) = checker
                .explain(&listing, NO_VARIABLES)
                .map_err(|e| format!("{stage}: {e}"))?;
            if !ok {
                let directives = checker
                    .to_string()
                    .lines()
                    .zip(lines)
                    .map(|(directive, line)| format!("line {line}: {directive}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                return Err(format!("{stage}:\n{directives}\n{explanation}"));
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

    fn done(&self) -> bool {
        self.runs.is_empty() && self.checks.is_empty() && self.encodings.is_empty()
    }

    fn check_optimized(
        &mut self,
        module: Module<VerifiedBody<'_>>,
        ids: &FunctionIds<Function>,
    ) -> Result<()> {
        self.check("opt", || listing::hir(&module, ids))?;
        self.interpret("opt", module.functions[module.call_graph.root()].function)?;
        lir::check_machine(self, module, ids)
    }

    fn execute(mut self) -> Result<()> {
        let f = input::parse(&self.ir)?;
        let verified = f.verified_module();
        if let Some((line, reason)) = self.reject {
            if !self.done() {
                return Err(format!(
                    "line {line}: @reject requires a verifier-only case"
                ));
            }
            return match verified {
                Err(CompileError::InvalidIr(e)) if e.to_string().contains(reason) => Ok(()),
                Err(e) => Err(format!("line {line}, verify: expected '{reason}', got {e}")),
                Ok(_) => Err(format!(
                    "line {line}, verify: accepted IR, expected '{reason}'"
                )),
            };
        }
        if self.done() {
            return Err("case has no expectations".into());
        }
        #[cfg(target_arch = "aarch64")]
        self.checks.retain(|stage, _| !X64_STAGES.contains(stage));
        #[cfg(target_arch = "x86_64")]
        {
            self.checks.retain(|stage, _| !ARM_STAGES.contains(stage));
            self.encodings.clear();
        }
        let (source, ids) = verified.map_err(|e| format!("verify: {e}"))?;
        self.check("source", || listing::hir(&source, &ids))?;
        self.check("cfg", || listing::cfg(&source))?;
        self.interpret("source", &f)?;
        if self.done() {
            return Ok(());
        }
        if self.checks.contains_key("inline") || !self.runs.is_empty() {
            if self.policy.should_inline(&source) {
                let expanded =
                    passes::inline_all(&source, &ids).map_err(|e| format!("inline: {e}"))?;
                let (module, ids) = expanded
                    .verified_module()
                    .map_err(|e| format!("inline: {e}"))?;
                self.check("inline", || listing::hir(&module, &ids))?;
                self.interpret("inline", &expanded)?;
            } else {
                self.check("inline", || listing::hir(&source, &ids))?;
                self.interpret("inline", &f)?;
            }
        }
        if self.done() {
            return Ok(());
        }
        jit::optimize_and_lower(&source, &ids, self.policy, |module, ids| {
            // Keep fixture assertion failures separate from compiler errors that trigger fallback.
            Ok(self.check_optimized(module, ids))
        })
        .map_err(|e| format!("opt: {e}"))??;
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
    let target = if cfg!(target_arch = "aarch64") {
        &ARM_STAGES[..]
    } else {
        &X64_STAGES[..]
    };
    for &stage in COMMON_STAGES.iter().chain(target) {
        let source = format!(
            "fn main x:i32 -> i32\nret x\n@{stage} regex: ANY=.*\n@run 8 -> 8\n@{stage} check: absent_opcode"
        );
        let error = run_text(&source).unwrap_err();
        assert!(
            error.contains(&format!("{stage}:"))
                && error.contains("line 5: #1 check: absent_opcode"),
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

#[test]
fn unknown_directives_report_the_absolute_line_in_later_cases() {
    let text = "fn main x:i32 -> i32\nret x\n@run 2 -> 2\n---\nfn main x:i32 -> i32\nret x\n@po  check: v0";
    let error = run_text(text).unwrap_err();
    assert!(error.contains("line 7: unknown directive @po"), "{error}");
}

#[test]
fn foreign_machine_expectations_leave_common_checks_active() {
    let foreign = if cfg!(target_arch = "aarch64") {
        "isel"
    } else {
        "aarch64-isel"
    };
    let case =
        format!("fn main x:i32 -> i32\nret x\n@{foreign} check: absent_opcode\n@opt check: ret v0");
    run_text(&case).unwrap();
}
