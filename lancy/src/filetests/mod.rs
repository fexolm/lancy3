//! Parse and check a fixture against the HIR passes and the host JIT.
pub(crate) mod listing;
mod run;

use crate::hir::{self, Function, ssa::VerifiedBody, text::ParseError, transforms::passes};
use crate::lir::{self, Stage, Stages};
use crate::module::Module;
use crate::{CompileError, Inlining, Kernel, jit};
use filecheck::{CheckerBuilder, NO_VARIABLES};
use run::Run;
use splice_target::{Arch, Target};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) type Result<T> = std::result::Result<T, String>;

const COMMON_STAGES: [&str; 4] = ["source", "cfg", "inline", "opt"];
/// The directives of each machine's selection, allocation, layout and
/// assembly. `@aarch64` checks AArch64 code by its encodings instead.
const MACHINES: [(Arch, [Option<&str>; 4]); 2] = [
    (
        Arch::X86_64,
        [Some("isel"), Some("alloc"), Some("layout"), Some("asm")],
    ),
    (
        Arch::Aarch64,
        [Some("aarch64-isel"), Some("aarch64-alloc"), None, None],
    ),
];

/// Lower a module for a target, reporting the stages: production's
/// `lir::emit`, or a stand-in that fails as a target limit would.
type Emit<'e> = &'e mut dyn FnMut(
    &Target,
    Module<VerifiedBody<'_>>,
    Stages<'_>,
) -> std::result::Result<lir::Code, CompileError>;

/// What one attempt at lowering produced. Production may discard an
/// attempt and lower again without inlining, so the case checks only the
/// attempt that succeeded.
struct Lowered {
    opt: String,
    /// The optimized text reads back and interprets as the runs expect.
    optimized: Result<()>,
    listings: BTreeMap<&'static str, Result<String>>,
    /// AArch64 code and the profile it was compiled for.
    aarch64: Option<(Vec<u8>, &'static str)>,
}

fn machine_stage(stage: &str) -> bool {
    MACHINES
        .iter()
        .any(|(_, stages)| stages.contains(&Some(stage)))
}

#[derive(Default)]
pub(crate) struct Case<'a> {
    /// The IR lines of the case with their line numbers.
    ir: Vec<(usize, &'a str)>,
    checks: BTreeMap<&'a str, (Vec<usize>, CheckerBuilder)>,
    encodings: Vec<(usize, &'a str)>,
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
            stage if COMMON_STAGES.contains(&stage) || machine_stage(stage) => {
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

    fn check(&mut self, stage: &str, text: impl FnOnce() -> String) -> Result<()> {
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

    /// The optimized root reads back as it prints and interprets as the
    /// runs expect.
    fn check_root(&self, root: &Function) -> Result<()> {
        // Optimized values are not numbered in text order; the text read
        // back is, and then prints the same again.
        let once = reprint(&root.to_string())?;
        if reprint(&once)? != once {
            return Err(format!("the optimized text does not read back:\n{once}"));
        }
        self.interpret("opt", root)
    }

    /// Whether any directive checks code for `arch`.
    fn wants(&self, arch: Arch) -> bool {
        let (_, stages) = MACHINES.iter().find(|(a, _)| *a == arch).unwrap();
        stages
            .iter()
            .flatten()
            .any(|stage| self.checks.contains_key(stage))
            || (arch == Arch::Aarch64 && !self.encodings.is_empty())
    }

    /// Lower an optimized module for each machine the case checks, keeping
    /// what each stage reports. Compiler errors return, so that a limit
    /// falls back as in production; fixture failures wait in `Lowered`.
    fn lower(
        &self,
        module: Module<VerifiedBody<'_>>,
        targets: &[(Arch, Target)],
        emit: Emit<'_>,
    ) -> std::result::Result<Lowered, CompileError> {
        let root = module.functions[module.call_graph.root()].function;
        let mut lowered = Lowered {
            opt: listing::hir(&module),
            optimized: self.check_root(root),
            listings: BTreeMap::new(),
            aarch64: None,
        };
        let mut module = Some(module);
        for (arch, target) in targets {
            if !self.wants(*arch) {
                continue;
            }
            // Lowering consumes the module; the root verifies to it again.
            let module = match module.take() {
                Some(module) => module,
                None => root.verified_module()?,
            };
            let (_, names) = MACHINES.iter().find(|(a, _)| a == arch).unwrap();
            let mut report = |stage: Stage<'_>| {
                let (index, text) = match stage {
                    Stage::Select(text) => (0, text),
                    Stage::Allocate(text) => (1, text),
                    Stage::Layout(text) => (2, text),
                    Stage::Assemble(code) => {
                        if let Some(name) = names[3].filter(|name| self.checks.contains_key(name)) {
                            lowered.listings.insert(name, lir::disassemble(code));
                        }
                        return;
                    }
                };
                if let Some(name) = names[index].filter(|name| self.checks.contains_key(name)) {
                    let listing = lowered.listings.entry(name).or_insert(Ok(String::new()));
                    if let Ok(listing) = listing {
                        listing.push_str(&text());
                    }
                }
            };
            let code = emit(target, module, &mut report)?;
            if *arch == Arch::Aarch64 {
                let features = target.arm64().unwrap();
                let profile = match (features.sve2, features.sve) {
                    (true, _) => "sve2",
                    (_, true) => "sve",
                    _ => "neon",
                };
                lowered.aarch64 = Some((code.bytes, profile));
            }
        }
        Ok(lowered)
    }

    /// The `@aarch64` assertions on AArch64 code compiled for `profile`.
    fn check_encodings(&mut self, code: &[u8], profile: &str) -> Result<()> {
        // AArch64 code has no literal pool: every emitted word is an instruction.
        let words = code.as_chunks::<4>().0;
        for (line, assertion) in std::mem::take(&mut self.encodings) {
            let invalid = || format!("line {line}: invalid AArch64 assertion: {assertion}");
            let parts: Vec<_> = assertion.split_whitespace().collect();
            let [only, presence, name] = parts.as_slice() else {
                return Err(invalid());
            };
            let has = match *presence {
                "has" => true,
                "lacks" => false,
                _ => return Err(invalid()),
            };
            if !matches!(*only, "all" | "neon" | "sve" | "sve2") {
                return Err(invalid());
            }
            let (mask, opcode) = lir::aarch64::encoding(name)
                .ok_or_else(|| format!("line {line}: unknown AArch64 encoding {name}"))?;
            if *only != "all" && *only != profile {
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
        Ok(())
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

    fn execute(self) -> Result<()> {
        self.execute_with(&mut |target, module, stages| lir::emit(target, module, false, stages))
    }

    fn execute_with(mut self, emit: Emit<'_>) -> Result<()> {
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
            // An expansion over a limit stays calls, as in production.
            let expanded = if self.policy.should_inline(&source) {
                match passes::inline_all(&source, crate::lir::MAX_VALUES) {
                    Err(e) if e.is_limit() => None,
                    Err(e) => return Err(format!("inline: {e}")),
                    Ok(expanded) => Some(expanded),
                }
            } else {
                None
            };
            if let Some(expanded) = expanded {
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
        let targets = [Arch::X86_64, Arch::Aarch64]
            .map(|arch| lir::filetest_target(arch).map(|target| (arch, target)));
        let targets = targets.into_iter().collect::<Result<Vec<_>>>()?;
        let lowered = jit::optimize_and_lower(&source, self.policy, |module| {
            self.lower(module, &targets, emit)
        })
        .map_err(|e| format!("opt: {e}"))?;
        self.check("opt", || lowered.opt)?;
        lowered.optimized?;
        for (stage, listing) in lowered.listings {
            let listing = listing?;
            self.check(stage, || listing)?;
        }
        if let Some((code, profile)) = lowered.aarch64 {
            self.check_encodings(&code, profile)?;
        }
        self.all_checks_reached()?;
        if !self.runs.is_empty() {
            let kernel = Kernel::compile(
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
    for &stage in COMMON_STAGES.iter().chain(
        MACHINES
            .iter()
            .flat_map(|(_, stages)| stages.iter().flatten()),
    ) {
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
    run_text("fn main(x:i32) -> i32\nret x\n@isel check: ret\n@aarch64-isel check: ret").unwrap();
}

#[test]
fn checks_that_never_reach_their_stage_are_errors() {
    let mut case = Case::default();
    case.line(1, "@opt check: ret").unwrap();
    let error = case.all_checks_reached().unwrap_err();
    assert!(error.contains("@opt checks were never reached"), "{error}");
}

/// A target limit hit while lowering the inlined body is a compiler error,
/// not a fixture failure: the case falls back to calls, as production
/// does, and its checks read the lowering that succeeded.
#[test]
fn lowering_limits_fall_back_to_calls_as_in_production() {
    let text = "fn plus(x:i32 inc:i32) -> i32\n  sum = add x inc\n  ret sum\n\
        fn main(x:i32) -> i32\n  one = const 1\n  r = call plus(x one)\n  ret r\n\
        @isel check: call f0\n@aarch64-isel check: call f0\n@run 2 -> 3";
    let mut case = Case::default();
    for (i, line) in text.lines().enumerate() {
        case.line(i + 1, line).unwrap();
    }
    let mut inlined = 0;
    case.execute_with(&mut |target, module, stages| {
        if module.functions.len() == 1 {
            inlined += 1;
            return Err(CompileError::Unsupported(crate::Limit::Registers));
        }
        lir::emit(target, module, false, stages)
    })
    .unwrap();
    assert_eq!(
        inlined, 1,
        "the inlined body is lowered once, then the calls"
    );
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
