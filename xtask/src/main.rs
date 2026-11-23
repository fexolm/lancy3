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
//!   interface has one definition (Linux).
//! - `doc`: rustdoc for the workspace with warnings denied.
//! - `bench [ARGS]`: the backend comparison bench, with ARGS for Criterion.
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
     profile SIMD [--target T] [--expect-bytes N] | static-llvm | doc | bench [ARGS]"
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
    let mut command = cargo(&["test", "--workspace"]);
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

fn profile(simd: &str, target: &[String], bytes: Option<String>) -> Result {
    let mut probe = cargo(&["run", "--quiet", "--release", "-p", "splice-target"]);
    probe
        .args(["--example", "simd_profile"])
        .args(target)
        .env("SPLICE_SIMD", simd);
    let output = probe
        .output()
        .map_err(|error| format!("{probe:?}: {error}"))?;
    let selected = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !output.status.success() && bytes.is_none() {
        eprintln!(
            "xtask: skipping SPLICE_SIMD={simd}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return Ok(());
    }
    if !output.status.success() {
        return Err(format!(
            "SPLICE_SIMD={simd} is unavailable: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
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
