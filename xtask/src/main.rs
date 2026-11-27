//! `cargo xtask <command>`: the checks CI runs, in one place.
//!
//! - `ci [--quick]`: everything below that runs on this host; `--quick` is
//!   only `lint` and `test`.
//! - `lint`: rustfmt and Clippy for the default build, every feature and the
//!   Lancy `experimental-simd` build.
//! - `test [--debug] [--target T]`: the workspace tests.
//! - `features`: tests that need non-default features, the build without
//!   default features and the benchmarks in `--test` mode.
//! - `profile SIMD [--target T] [--expect-bytes N]`: the packages that
//!   generate code, tested with `SPLICE_SIMD=SIMD`. Skipped when the host
//!   cannot run that profile, unless `--expect-bytes` demands N-byte vectors.
//! - `static-llvm`: link LLVM statically and check that the GDB JIT
//!   interface has one definition (Linux; on Ubuntu the static link line
//!   also needs `libpolly-19-dev`).
//! - `doc`: rustdoc for the workspace with warnings denied.
//! - `bench [ARGS]`: the backend comparison bench, with ARGS for Criterion.
//! - `bench-tables [--run] [SIMD...]`: Markdown tables of the backend bench
//!   with the commit, toolchain versions and the ISA of each profile. With
//!   `--run`, first benchmark every given profile (default: all that this
//!   host runs) into its own Criterion baseline.
//!
//! Cargo runs with the toolchain that started `cargo xtask`, so
//! `cargo +1.94.1 xtask ci` checks with the CI toolchain.
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// Every package whose generated code depends on the SIMD profile.
const PROFILE_PACKAGES: &[&str] = &[
    "lancy",
    "splice-target",
    "splice-lancy",
    "splice-cranelift",
    "splice-llvm",
    "splice-kernel",
    "splice-testkit",
];

type Result<T = ()> = std::result::Result<T, String>;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("xtask: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result {
    let (command, rest) = args.split_first().ok_or_else(usage)?;
    if command == "bench" {
        let mut bench = cargo(&["bench", "-p", "splice-kernel", "--features", "llvm"]);
        bench.args(["--bench", "backends", "--"]).args(rest);
        return status(bench);
    }
    let mut options = Options::parse(rest)?;
    match command.as_str() {
        "ci" => {
            let quick = options.flag("--quick");
            options.finish()?;
            lint()?;
            test(false, &[])?;
            if !quick {
                features()?;
                doc()?;
                if cfg!(target_os = "linux") {
                    static_llvm()?;
                }
            }
            Ok(())
        }
        "test" => {
            let debug = options.flag("--debug");
            let target = options.target();
            options.finish()?;
            test(debug, &target)
        }
        "bench-tables" => {
            let run = options.flag("--run");
            let profiles = std::mem::take(&mut options.positional);
            options.finish()?;
            bench_tables(run, &profiles)
        }
        "profile" => {
            let simd = options
                .positional
                .pop()
                .ok_or("profile needs a SIMD profile")?;
            let target = options.target();
            let bytes = options.value("--expect-bytes");
            options.finish()?;
            profile(&simd, &target, bytes)
        }
        "lint" | "features" | "static-llvm" | "doc" => {
            options.finish()?;
            match command.as_str() {
                "lint" => lint(),
                "features" => features(),
                "static-llvm" => static_llvm(),
                _ => doc(),
            }
        }
        _ => Err(usage()),
    }
}

fn usage() -> String {
    "usage: cargo xtask ci [--quick] | lint | test [--debug] [--target T] | features | \
     profile SIMD [--target T] [--expect-bytes N] | static-llvm | doc | bench [ARGS] | \
     bench-tables [--run] [SIMD...]"
        .into()
}

struct Options {
    flags: Vec<String>,
    values: Vec<(String, String)>,
    positional: Vec<String>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self> {
        let mut options = Self {
            flags: Vec::new(),
            values: Vec::new(),
            positional: Vec::new(),
        };
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--target" | "--expect-bytes" => {
                    let value = args.next().ok_or_else(|| format!("{arg} needs a value"))?;
                    options.values.push((arg.clone(), value.clone()));
                }
                flag if flag.starts_with("--") => options.flags.push(arg.clone()),
                _ => options.positional.push(arg.clone()),
            }
        }
        Ok(options)
    }

    fn flag(&mut self, name: &str) -> bool {
        let before = self.flags.len();
        self.flags.retain(|flag| flag != name);
        self.flags.len() != before
    }

    fn value(&mut self, name: &str) -> Option<String> {
        let index = self.values.iter().position(|(key, _)| key == name)?;
        Some(self.values.remove(index).1)
    }

    fn target(&mut self) -> Vec<String> {
        self.value("--target")
            .map(|target| vec!["--target".into(), target])
            .unwrap_or_default()
    }

    /// Reject what no step consumed, so a misspelt option is not ignored.
    fn finish(self) -> Result {
        let unused: Vec<_> = self
            .flags
            .into_iter()
            .chain(self.values.into_iter().map(|(key, _)| key))
            .chain(self.positional)
            .collect();
        if unused.is_empty() {
            Ok(())
        } else {
            Err(format!("unused arguments: {}", unused.join(" ")))
        }
    }
}

fn cargo(args: &[&str]) -> Command {
    let mut command = Command::new(env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    command.args(args);
    command
}

fn status(mut command: Command) -> Result {
    eprintln!("xtask: {command:?}");
    let status = command
        .status()
        .map_err(|error| format!("{command:?}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{command:?} failed: {status}"))
    }
}

fn clippy(args: &[&str]) -> Result {
    let mut command = cargo(&["clippy"]);
    command
        .args(args)
        .args(["--all-targets", "--", "-D", "warnings"]);
    status(command)
}

fn lint() -> Result {
    status(cargo(&["fmt", "--all", "--", "--check"]))?;
    clippy(&["--workspace"])?;
    clippy(&["--workspace", "--all-features"])?;
    clippy(&["-p", "lancy", "--features", "experimental-simd"])
}

fn test(debug: bool, target: &[String]) -> Result {
    let mut command = cargo(&["test", "--workspace", "--no-fail-fast"]);
    if !debug {
        command.arg("--release");
    }
    command.args(target);
    status(command)
}

fn features() -> Result {
    status(cargo(&[
        "test",
        "--release",
        "-p",
        "lancy",
        "--features",
        "experimental-simd",
    ]))?;
    status(cargo(&[
        "test",
        "--release",
        "-p",
        "splice-kernel",
        "--all-features",
    ]))?;
    // Without the optional backends: the facade, and its tests with only the
    // Rust backends, which need no system LLVM.
    status(cargo(&[
        "check",
        "-p",
        "splice-kernel",
        "--no-default-features",
    ]))?;
    status(cargo(&["test", "--release", "-p", "splice-kernel"]))?;
    status(cargo(&[
        "bench",
        "-p",
        "splice-kernel",
        "--features",
        "llvm",
        "--bench",
        "backends",
        "--",
        "--test",
    ]))
}

/// What `SPLICE_SIMD=simd` selects here: the profile and width, then the
/// enabled extensions. `Err` carries why the host cannot run it.
fn probe(simd: &str, target: &[String]) -> Result<std::result::Result<(String, String), String>> {
    let mut probe = cargo(&["run", "--quiet", "--release", "-p", "splice-target"]);
    probe
        .args(["--example", "simd_profile"])
        .args(target)
        .env("SPLICE_SIMD", simd);
    let output = probe
        .output()
        .map_err(|error| format!("{probe:?}: {error}"))?;
    if !output.status.success() {
        return Ok(Err(String::from_utf8_lossy(&output.stderr).trim().into()));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    let selected = lines.next().unwrap_or_default().trim().to_owned();
    let features = lines.next().unwrap_or_default().trim().to_owned();
    Ok(Ok((selected, features)))
}

fn profile(simd: &str, target: &[String], bytes: Option<String>) -> Result {
    let selected = match probe(simd, target)? {
        Ok((selected, _)) => selected,
        Err(error) if bytes.is_none() => {
            eprintln!("xtask: skipping SPLICE_SIMD={simd}: {error}");
            return Ok(());
        }
        Err(error) => return Err(format!("SPLICE_SIMD={simd} is unavailable: {error}")),
    };
    eprintln!("xtask: SPLICE_SIMD={simd} selects {selected}");
    if let Some(bytes) = bytes
        && !selected.ends_with(&format!(", {bytes} bytes"))
    {
        return Err(format!("expected {bytes}-byte vectors, got {selected}"));
    }
    let mut command = cargo(&["test", "--release", "--tests"]);
    for package in PROFILE_PACKAGES {
        command.args(["-p", package]);
    }
    command.args(target).env("SPLICE_SIMD", simd);
    status(command)
}

const ALL_PROFILES: &[&str] = &["avx512", "avx2", "sse", "sve2", "sve", "neon"];
/// One sample setting for every row, so all numbers are comparable.
const BENCH_SETTINGS: &[&str] = &[
    "--sample-size",
    "20",
    "--warm-up-time",
    "1",
    "--measurement-time",
    "3",
];
const BACKENDS: &[(&str, &str)] = &[
    ("lancy", "Lancy"),
    ("cranelift", "Cranelift"),
    ("llvm_manual_simd", "LLVM"),
];

/// SPL-05: tables of the backend bench, regenerated rather than edited.
/// Each profile's results live in the Criterion baseline `splice-SIMD`,
/// next to the ISA it ran with.
fn bench_tables(run: bool, profiles: &[String]) -> Result {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let criterion = env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| root.join("target"), PathBuf::from)
        .join("criterion");
    let profiles: Vec<String> = if profiles.is_empty() {
        ALL_PROFILES.iter().map(|&p| p.into()).collect()
    } else {
        profiles.to_vec()
    };
    let mut sections = Vec::new();
    for simd in &profiles {
        let isa = criterion.join(format!("splice-{simd}.isa"));
        if run {
            let (selected, features) = match probe(simd, &[])? {
                Ok(found) => found,
                Err(error) => {
                    eprintln!("xtask: skipping SPLICE_SIMD={simd}: {error}");
                    continue;
                }
            };
            let mut bench = cargo(&["bench", "-p", "splice-kernel", "--features", "llvm"]);
            bench
                .args(["--bench", "backends", "--", "--noplot", "--save-baseline"])
                .arg(format!("splice-{simd}"))
                .args(BENCH_SETTINGS)
                .env("SPLICE_SIMD", simd)
                // The tables go to stdout; Criterion's report does not.
                .stdout(std::process::Stdio::from(std::io::stderr()));
            status(bench)?;
            std::fs::write(&isa, format!("{selected}\n{features}\n{}\n", commit(root)))
                .map_err(|e| format!("{}: {e}", isa.display()))?;
        }
        let Ok(description) = std::fs::read_to_string(&isa) else {
            continue;
        };
        let rows = baseline(&criterion, &format!("splice-{simd}"))?;
        if !rows.is_empty() {
            sections.push((simd.clone(), description, rows));
        }
    }
    if sections.is_empty() {
        return Err(format!(
            "no baselines in {}; run with --run",
            criterion.display()
        ));
    }
    print!("{}", render(root, &sections));
    Ok(())
}

/// `(group, backend) -> median ns` of one Criterion baseline.
type Rows = std::collections::BTreeMap<(String, String), f64>;

fn baseline(criterion: &Path, name: &str) -> Result<Rows> {
    let mut rows = Rows::new();
    let mut pending = vec![criterion.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.filter_map(|entry| entry.ok()) {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if path.file_name().is_some_and(|file| file == name) {
                let read = |file| {
                    std::fs::read_to_string(path.join(file))
                        .map_err(|e| format!("{}: {e}", path.display()))
                };
                let benchmark = read("benchmark.json")?;
                let estimates = read("estimates.json")?;
                let group = json_string(&benchmark, "group_id");
                let function = json_string(&benchmark, "function_id");
                let median = estimates
                    .split_once("\"median\":")
                    .and_then(|(_, rest)| json_number(rest, "point_estimate"));
                if let (Some(group), Some(function), Some(median)) = (group, function, median) {
                    rows.insert((group, function), median);
                }
            } else {
                pending.push(path);
            }
        }
    }
    Ok(rows)
}

fn json_string(json: &str, key: &str) -> Option<String> {
    let (_, rest) = json.split_once(&format!("\"{key}\":\""))?;
    Some(rest.split_once('"')?.0.to_owned())
}

fn json_number(json: &str, key: &str) -> Option<f64> {
    let (_, rest) = json.split_once(&format!("\"{key}\":"))?;
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn render(root: &Path, sections: &[(String, String, Rows)]) -> String {
    use std::fmt::Write;
    let run = |program: &str, args: &[&str]| {
        Command::new(program)
            .args(args)
            .current_dir(root)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            .unwrap_or_else(|| "unknown".into())
    };
    // The commit each profile was measured at, recorded by `--run`.
    let mut commits: Vec<&str> = sections
        .iter()
        .map(|(_, description, _)| description.lines().nth(2).unwrap_or("unknown"))
        .collect();
    commits.dedup();
    let rustc = run(
        &env::var("RUSTC").unwrap_or_else(|_| "rustc".into()),
        &["--version"],
    );
    let llvm = env::var_os("LLVM_SYS_191_PREFIX")
        .map(|prefix| PathBuf::from(prefix).join("bin/llvm-config"))
        .unwrap_or_else(|| "llvm-config-19".into());
    let llvm = run(&llvm.to_string_lossy(), &["--version"]);
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).unwrap_or_default();
    let cranelift = lock
        .split("[[package]]")
        .find(|package| package.contains("name = \"cranelift-codegen\""))
        .and_then(|package| json_like_value(package, "version"))
        .unwrap_or_else(|| "unknown".into());
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find_map(|line| line.strip_prefix("model name"))
                .map(|name| name.trim_start_matches([' ', '\t', ':']).to_owned())
        })
        .unwrap_or_else(|| env::consts::ARCH.into());

    let mut out = String::new();
    let _ = writeln!(
        out,
        "<!-- Generated by `cargo xtask bench-tables`; do not edit by hand. -->\n"
    );
    let _ = writeln!(out, "- Commit: {}", commits.join(", "));
    let _ = writeln!(
        out,
        "- Toolchain: {rustc}; LLVM {llvm}; Cranelift {cranelift}"
    );
    let _ = writeln!(out, "- CPU: {cpu}");
    let _ = writeln!(
        out,
        "- Criterion `{}`; median time. LLVM is `LlvmManualSimd` (loop and SLP \
         vectorizers off).",
        BENCH_SETTINGS.join(" ")
    );
    for (simd, description, rows) in sections {
        let mut lines = description.lines();
        let selected = lines.next().unwrap_or_default();
        let features = lines.next().unwrap_or_default();
        let _ = writeln!(out, "\n### `SPLICE_SIMD={simd}`: {selected}\n");
        let _ = writeln!(
            out,
            "Every backend gets the same extensions: {}.\n",
            if features.is_empty() {
                "the baseline only"
            } else {
                features
            }
        );
        for route in ["build", "from_ir"] {
            let mut workloads: Vec<&str> = rows
                .keys()
                .filter_map(|(group, _)| {
                    group
                        .strip_suffix(&format!("/execute/{route}"))
                        .or_else(|| group.strip_suffix(&format!("/compile/{route}")))
                })
                .collect();
            workloads.sort_unstable();
            workloads.dedup();
            if workloads.is_empty() {
                continue;
            }
            let _ = writeln!(
                out,
                "Route `{route}`: compile (with installation), then a warm call.\n"
            );
            let mut header = String::from("| Workload |");
            let mut rule = String::from("| --- |");
            for stage in ["compile", "call"] {
                for (_, name) in BACKENDS {
                    let _ = write!(header, " {name} {stage} |");
                    rule.push_str(" ---: |");
                }
            }
            let _ = writeln!(out, "{header}\n{rule}");
            for workload in workloads {
                let _ = write!(out, "| `{workload}` |");
                for stage in ["compile", "execute"] {
                    for (backend, _) in BACKENDS {
                        let group = format!("{workload}/{stage}/{route}");
                        let time = rows.get(&(group, (*backend).into()));
                        let _ = write!(out, " {} |", time.map_or("–".into(), |&ns| duration(ns)));
                    }
                }
                out.push('\n');
            }
            out.push('\n');
        }
    }
    out
}

/// HEAD, marked when tracked files differ from it.
fn commit(root: &Path) -> String {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    let head = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    match git(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(dirty) if !dirty.is_empty() => format!("`{head}` with uncommitted changes"),
        _ => format!("`{head}`"),
    }
}

fn json_like_value(toml: &str, key: &str) -> Option<String> {
    let (_, rest) = toml.split_once(&format!("{key} = \""))?;
    Some(rest.split_once('"')?.0.to_owned())
}

fn duration(ns: f64) -> String {
    let (value, unit) = if ns >= 1e6 {
        (ns / 1e6, "ms")
    } else if ns >= 1e3 {
        (ns / 1e3, "µs")
    } else {
        (ns, "ns")
    };
    let digits = if value >= 100.0 {
        0
    } else if value >= 10.0 {
        1
    } else {
        2
    };
    format!("{value:.digits$} {unit}")
}

/// BE-05: with LLVM linked statically, its GDB JIT interface and splice-jit's
/// weak one must resolve to a single definition.
fn static_llvm() -> Result {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let real = env::var_os("LLVM_SYS_191_PREFIX")
        .map(|prefix| PathBuf::from(prefix).join("bin/llvm-config"))
        .unwrap_or_else(|| "llvm-config-19".into());
    // splice-llvm prefers libLLVM.so; a prefix whose llvm-config reports no
    // shared library makes llvm-sys link the static archives instead.
    let prefix = root.join("target/static-llvm/prefix");
    std::fs::create_dir_all(prefix.join("bin")).map_err(|e| e.to_string())?;
    let config = prefix.join("bin/llvm-config");
    std::fs::write(
        &config,
        format!(
            "#!/bin/sh\ncase \" $* \" in\n  *\" --link-shared \"*) exit 1 ;;\n  \
             *\" --shared-mode \"*) echo static; exit 0 ;;\nesac\nexec '{}' \"$@\"\n",
            real.display()
        ),
    )
    .map_err(|e| e.to_string())?;
    status({
        let mut chmod = Command::new("chmod");
        chmod.arg("+x").arg(&config);
        chmod
    })?;
    let target_dir = root.join("target/static-llvm");
    let mut command = cargo(&["test", "--release", "-p", "splice-kernel"]);
    command
        .args(["--features", "llvm,cranelift", "--test", "debug_symbols"])
        .env("LLVM_SYS_191_PREFIX", &prefix)
        .env("CARGO_TARGET_DIR", &target_dir);
    status(command)?;
    let binary = std::fs::read_dir(target_dir.join("release/deps"))
        .map_err(|e| e.to_string())?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .filter(|path| {
            let name = path.file_name().unwrap().to_string_lossy();
            name.starts_with("debug_symbols-") && path.extension().is_none()
        })
        .max_by_key(|path| path.metadata().and_then(|m| m.modified()).ok())
        .ok_or("no debug_symbols test binary")?;
    let ldd = Command::new("ldd")
        .arg(&binary)
        .output()
        .map_err(|e| e.to_string())?;
    if String::from_utf8_lossy(&ldd.stdout).contains("libLLVM") {
        return Err(format!("{} links libLLVM dynamically", binary.display()));
    }
    let nm = Command::new("nm")
        .arg(&binary)
        .output()
        .map_err(|e| e.to_string())?;
    let symbols = String::from_utf8_lossy(&nm.stdout);
    for symbol in ["__jit_debug_descriptor", "__jit_debug_register_code"] {
        let definitions = symbols
            .lines()
            .filter(|line| line.ends_with(&format!(" {symbol}")))
            .count();
        if definitions != 1 {
            return Err(format!("{symbol} has {definitions} definitions"));
        }
    }
    eprintln!("xtask: static LLVM links with one GDB JIT interface");
    Ok(())
}

fn doc() -> Result {
    let mut command = cargo(&["doc", "--workspace", "--no-deps"]);
    command.env("RUSTDOCFLAGS", "-D warnings");
    status(command)
}
