//! Parse and check a fixture against the HIR passes and the host JIT.
pub(crate) mod listing;
mod run;

use crate::hir::{self, Function, ssa::VerifiedBody, text::ParseError, transforms::passes};
use crate::module::Module;
use crate::{CompileError, Inlining, ScalarKernel, jit, lir};
use filecheck::{CheckerBuilder, NO_VARIABLES};
use run::Run;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) type Result<T> = std::result::Result<T, String>;

const COMMON_STAGES: [&str; 5] = ["source", "cfg", "inline", "opt", "layout"];
const X64_STAGES: [&str; 3] = ["isel", "alloc", "asm"];
const ARM_STAGES: [&str; 2] = ["aarch64-isel", "aarch64-alloc"];

#[derive(Default)]
pub(crate) struct Case<'a> {
    /// The IR lines of the case with their line numbers.
    ir: Vec<(usize, &'a str)>,
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

    /// Every directive must reach its stage: a check that silently never
    /// runs would pass regardless of the compiler's output.
    fn all_checks_reached(&self) -> Result<()> {
        if let Some(stage) = self.checks.keys().next() {
            return Err(format!("@{stage} checks were never reached"));
        }
        if !self.encodings.is_empty() {
            return Err("@aarch64 checks were never reached".into());
        }
        Ok(())
    }

    fn done(&self) -> bool {
        self.runs.is_empty() && self.checks.is_empty() && self.encodings.is_empty()
    }

    fn check_optimized(&mut self, module: Module<VerifiedBody<'_>>) -> Result<()> {
        self.check("opt", || listing::hir(&module))?;
        let root = module.functions[module.call_graph.root()].function;
        // Optimized values are not numbered in text order; the text read
        // back is, and then prints the same again.
        let once = reprint(&root.to_string())?;
        if reprint(&once)? != once {
            return Err(format!("the optimized text does not read back:\n{once}"));
        }
        self.interpret("opt", root)?;
        lir::check_machine(self, module)
    }

    /// The IR of the case, on the lines of the file it came from.
    fn text(&self) -> String {
        let mut text = String::new();
        for &(line, ir) in &self.ir {
            while text.lines().count() + 1 < line {
                text.push('\n');
            }
            text.push_str(ir);
            text.push('\n');
        }
        text
    }

    fn execute(mut self) -> Result<()> {
        let parsed = hir::text::parse(&self.text(), &RUST_FUNCTIONS);
        if let Some((line, reason)) = self.reject {
            if !self.done() {
                return Err(format!(
                    "line {line}: @reject requires a verifier-only case"
                ));
            }
            // The builder checks each operation and the verifier the whole function.
            let verified = match &parsed {
                Err(ParseError::Invalid(_, e)) => Err(CompileError::InvalidIr(e.clone())),
                Err(ParseError::Syntax(e)) => return Err(e.clone()),
                Ok(f) => f.verified_module().map(drop),
            };
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
        let f = parsed.map_err(|e| e.to_string())?;
        let printed = f.to_string();
        if reprint(&printed)? != printed {
            return Err(format!(
                "the text of the source does not read back:\n{printed}"
            ));
        }
        let source = f.verified_module().map_err(|e| format!("verify: {e}"))?;
        self.check("source", || listing::hir(&source))?;
        self.check("cfg", || listing::cfg(&source))?;
        self.interpret("source", &f)?;
        if self.done() {
            return Ok(());
        }
        if self.checks.contains_key("inline") || !self.runs.is_empty() {
            if self.policy.should_inline(&source) {
                let expanded = passes::inline_all(&source).map_err(|e| format!("inline: {e}"))?;
                let module = expanded
                    .verified_module()
                    .map_err(|e| format!("inline: {e}"))?;
                self.check("inline", || listing::hir(&module))?;
                self.interpret("inline", &expanded)?;
            } else {
                self.check("inline", || listing::hir(&source))?;
                self.interpret("inline", &f)?;
            }
        }
        if self.done() {
            return Ok(());
        }
        jit::optimize_and_lower(&source, self.policy, |module| {
            // Keep fixture assertion failures separate from compiler errors that trigger fallback.
            Ok(self.check_optimized(module))
        })
        .map_err(|e| format!("opt: {e}"))??;
        self.all_checks_reached()?;
        if !self.runs.is_empty() {
            let kernel = ScalarKernel::compile(
                &f,
                &crate::test_target::options().with_inlining(self.policy),
            )
            .map_err(|e| format!("JIT: {e}"))?;
            for run in self.runs {
                // SAFETY: these inputs passed checked interpretation on f above;
                // the kernel was compiled from f and remains live across the call.
                unsafe {
                    run.native(&f, &kernel)?;
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

/// Every `.lancy` file under `filetests/`, found at run time so that no build
/// script runs for consumers of the crate. `passes/experimental/` runs only
/// with `experimental-simd`.
fn fixtures(root: &Path, dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let relative = path.strip_prefix(root).unwrap();
        if path.is_dir() {
            let skip = relative == Path::new("passes/experimental")
                && !cfg!(feature = "experimental-simd");
            if !skip {
                fixtures(root, &path, files)?;
            }
        } else if path.extension().is_some_and(|ext| ext == "lancy") {
            files.push(relative.to_owned());
        }
    }
    Ok(())
}

/// Run every fixture, or those whose path contains `LANCY_FILETEST`, in
/// parallel, and report all failures together.
#[test]
fn filetests() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("filetests");
    let mut files = Vec::new();
    fixtures(&root, &root, &mut files).expect("read lancy/filetests");
    files.sort();
    assert!(!files.is_empty(), "no .lancy tests found");
    if let Some(filter) = std::env::var_os("LANCY_FILETEST") {
        let filter = filter.to_string_lossy().into_owned();
        files.retain(|file| file.to_string_lossy().contains(&filter));
        assert!(!files.is_empty(), "no .lancy test matches {filter}");
    }
    let next = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    let mut failures: Vec<String> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..workers.min(files.len()))
            .map(|_| {
                scope.spawn(|| {
                    let mut failures = Vec::new();
                    while let Some(file) = files.get(next.fetch_add(1, Ordering::Relaxed)) {
                        let text = fs::read_to_string(root.join(file)).unwrap();
                        let name = file.display().to_string().replace('\\', "/");
                        match std::panic::catch_unwind(|| run_text(&text)) {
                            Ok(Ok(())) => {}
                            Ok(Err(error)) => failures.push(format!("{name}: {error}")),
                            Err(_) => failures.push(format!("{name}: panicked")),
                        }
                    }
                    failures
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect()
    });
    failures.sort();
    assert!(
        failures.is_empty(),
        "{} of {} filetests failed:\n\n{}",
        failures.len(),
        files.len(),
        failures.join("\n\n")
    );
}

#[test]
fn mismatches_identify_the_stage_even_when_native_execution_would_succeed() {
    for &stage in COMMON_STAGES.iter().chain(&X64_STAGES).chain(&ARM_STAGES) {
        let source = format!(
            "fn main(x:i32) -> i32\nret x\n@{stage} regex: ANY=.*\n@run 8 -> 8\n@{stage} check: absent_opcode"
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
    let source = "fn main(p:ptr) -> i32\nx = load i32 p\nret x\n@run -> 0 | 00 | 00";
    let error = run_text(source).unwrap_err();
    assert!(
        error.contains("source") && error.contains("access outside buffer"),
        "{error}"
    );
}

#[test]
fn unknown_directives_report_the_absolute_line_in_later_cases() {
    let text = "fn main(x:i32) -> i32\nret x\n@run 2 -> 2\n---\nfn main(x:i32) -> i32\nret x\n@po  check: v0";
    let error = run_text(text).unwrap_err();
    assert!(error.contains("line 7: unknown directive @po"), "{error}");
}

#[test]
fn both_machines_are_checked_on_every_host() {
    for stage in ["isel", "aarch64-isel"] {
        let case = format!("fn main(x:i32) -> i32\nret x\n@{stage} check: absent_opcode");
        let error = run_text(&case).unwrap_err();
        assert!(error.contains(&format!("{stage}:")), "{error}");
    }
    run_text("fn main(x:i32) -> i32\nret x\n@isel check: ret\n@aarch64-isel check: Return")
        .unwrap();
}

#[test]
fn checks_that_never_reach_their_stage_are_errors() {
    let mut case = Case::default();
    case.line(1, "@opt check: ret").unwrap();
    let error = case.all_checks_reached().unwrap_err();
    assert!(error.contains("@opt checks were never reached"), "{error}");
}

#[test]
fn run_literals_must_match_the_signature() {
    for (run, error) in [
        ("@run 1.5f64 -> 1", "expected F32, written as f64"),
        ("@run 1 2 -> 1", "signature takes 1 values and 0 buffers"),
        ("@run 1 -> vec.f32[1]", "expected F32, written as vec.f32"),
        ("@run 1 -> 1 | 00", "bytes before | bytes after"),
        ("@run 1 -> !nope", "unknown trap !nope"),
    ] {
        let text = format!("fn main(x:f32) -> f32\nret x\n{run}");
        let actual = run_text(&text).unwrap_err();
        assert!(actual.contains(error), "{run}: {actual}");
    }
}

#[test]
fn aarch64_encodings_are_named_and_profiles_known() {
    for (check, error) in [
        ("all has no.such", "unknown AArch64 encoding no.such"),
        ("sme has blr", "invalid AArch64 assertion"),
    ] {
        let text = format!("fn main(x:i32) -> i32\nret x\n@aarch64 {check}");
        let actual = run_text(&text).unwrap_err();
        assert!(actual.contains(error), "{check}: {actual}");
    }
}

/// The text of the function that `text` reads as.
fn reprint(text: &str) -> Result<String> {
    hir::text::parse(text, &RUST_FUNCTIONS)
        .map(|f| f.to_string())
        .map_err(|e| format!("{e} in\n{text}"))
}

/// The Rust functions fixtures call by name.
static RUST_FUNCTIONS: [&hir::RustFunction; 3] = [
    &hir::RustFunction {
        name: "identity_or_error",
        entry: identity_or_error,
    },
    &hir::RustFunction {
        name: "double",
        entry: double,
    },
    &hir::RustFunction {
        name: "sum4",
        entry: sum4,
    },
];

/// The operation, or the trap whose code is the word.
unsafe extern "C" fn identity_or_error(_: *mut u8, value: u32, _: *const u8, fail: i32) -> i64 {
    match crate::Trap::from_code(fail as u32) {
        Some(trap) => trap.status(),
        None => i64::from(value),
    }
}

unsafe extern "C" fn double(_: *mut u8, value: u32, _: *const u8, _: i32) -> i64 {
    i64::from(value.wrapping_mul(2))
}

/// The sum of the packet's four `i32` words.
///
/// # Safety
/// The packet holds four `i32` words.
unsafe extern "C" fn sum4(_: *mut u8, _: u32, packet: *const u8, _: i32) -> i64 {
    let word = |i: usize| {
        // SAFETY: the caller's packet holds four words.
        unsafe { packet.cast::<i32>().add(i).read_unaligned() }
    };
    i64::from((0..4).map(word).fold(0, i32::wrapping_add) as u32)
}
