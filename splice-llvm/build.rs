// The LLVM distribution bundled with Visual Studio ships LLVM-C.lib but has
// no llvm-config. Linux uses llvm-sys's usual llvm-config based discovery.
use std::{
    collections::BTreeSet,
    env,
    fmt::Write,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    println!("cargo:rerun-if-env-changed=LLVM_SYS_191_PREFIX");
    println!("cargo:rerun-if-env-changed=ProgramFiles");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=windows/target.c");
    let prefix = env::var_os("LLVM_SYS_191_PREFIX").map(PathBuf::from)
        .or_else(find_windows_llvm)
        .expect("set LLVM_SYS_191_PREFIX to an LLVM 19.1 installation with bin/LLVM-C.dll and lib/LLVM-C.lib");
    let lib = prefix.join("lib/LLVM-C.lib");
    let dll = prefix.join("bin/LLVM-C.dll");
    assert!(
        lib.is_file() && dll.is_file(),
        "missing LLVM-C.lib or LLVM-C.dll under {}",
        prefix.display()
    );
    println!("cargo:rerun-if-changed={}", lib.display());
    println!("cargo:rerun-if-changed={}", dll.display());
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let include = prefix.join("include");
    let mut wrappers = cc::Build::new();
    if include.join("llvm-c/Target.h").is_file() {
        println!(
            "cargo:rerun-if-changed={}",
            include.join("llvm-c/Target.h").display()
        );
        wrappers.include(include);
    } else {
        let header = import_target_header(&prefix, &lib);
        fs::write(out.join("import_targets.h"), header).expect("write LLVM target initializers");
        wrappers
            .include(&out)
            .define("SPLICE_LLVM_IMPORT_TARGETS", None);
    }
    wrappers
        .file("windows/target.c")
        .compile("splice_llvm_targets");
    println!(
        "cargo:rustc-link-search=native={}",
        prefix.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=LLVM-C");
    // Cargo exposes link-search directories inside target/ to its executables.
    // Keep tests, examples and benches runnable without a global PATH change.
    fs::copy(dll, out.join("LLVM-C.dll")).expect("copy LLVM-C.dll to Cargo's runtime search path");
    println!("cargo:rustc-link-search=native={}", out.display());
}

// Mirrors the LLVM 19 llvm-c/Target.h inline routines when Visual Studio
// provides only LLVM-C.lib. Each component is derived independently because a
// compiled target may have a printer but no parser or disassembler.
fn import_target_header(prefix: &Path, lib: &Path) -> String {
    let nm = prefix.join("bin/llvm-nm.exe");
    let output = Command::new(&nm)
        .args(["--defined-only", "--just-symbol-name"])
        .arg(lib)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "cannot inspect {} with {}: {e}",
                lib.display(),
                nm.display()
            )
        });
    assert!(
        output.status.success(),
        "llvm-nm failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).expect("read LLVM-C import names");
    let names: BTreeSet<_> = text
        .lines()
        .map(str::trim)
        .filter_map(|s| s.strip_prefix("LLVMInitialize"))
        .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric()))
        .collect();
    let targets: Vec<_> = names
        .iter()
        .filter_map(|s| s.strip_suffix("TargetInfo"))
        .collect();
    assert!(
        targets.contains(&"X86"),
        "expected X86 target initializers in {}",
        lib.display()
    );

    let mut header = String::new();
    for suffix in [
        "TargetInfo",
        "Target",
        "TargetMC",
        "AsmPrinter",
        "AsmParser",
        "Disassembler",
    ] {
        let symbols: Vec<_> = targets
            .iter()
            .map(|t| format!("{t}{suffix}"))
            .filter(|s| names.contains(s.as_str()))
            .collect();
        for symbol in &symbols {
            writeln!(header, "void LLVMInitialize{symbol}(void);").unwrap();
        }
        writeln!(header, "static void LLVMInitializeAll{suffix}s(void) {{").unwrap();
        for symbol in &symbols {
            writeln!(header, "    LLVMInitialize{symbol}();").unwrap();
        }
        header.push_str("}\n");
    }
    for (component, suffixes) in [
        ("Target", &["TargetInfo", "Target", "TargetMC"][..]),
        ("AsmParser", &["AsmParser"]),
        ("AsmPrinter", &["AsmPrinter"]),
        ("Disassembler", &["Disassembler"]),
    ] {
        writeln!(
            header,
            "static int LLVMInitializeNative{component}(void) {{"
        )
        .unwrap();
        if suffixes
            .iter()
            .all(|s| names.contains(format!("X86{s}").as_str()))
        {
            for suffix in suffixes {
                writeln!(header, "    LLVMInitializeX86{suffix}();").unwrap();
            }
            header.push_str("    return 0;\n");
        } else {
            header.push_str("    return 1;\n");
        }
        header.push_str("}\n");
    }
    header
}

fn find_windows_llvm() -> Option<PathBuf> {
    let programs = PathBuf::from(env::var_os("ProgramFiles")?);
    let mut candidates = vec![programs.join("LLVM")];
    for year in fs::read_dir(programs.join("Microsoft Visual Studio"))
        .into_iter()
        .flatten()
        .flatten()
    {
        for edition in fs::read_dir(year.path()).into_iter().flatten().flatten() {
            candidates.push(edition.path().join("VC/Tools/Llvm/x64"));
        }
    }
    candidates.into_iter().find(|prefix| {
        prefix.join("lib/LLVM-C.lib").is_file()
            && prefix.join("bin/LLVM-C.dll").is_file()
            && Command::new(prefix.join("bin/clang.exe"))
                .arg("--version")
                .output()
                .is_ok_and(|output| {
                    String::from_utf8_lossy(&output.stdout).contains("clang version 19.1.")
                })
    })
}
