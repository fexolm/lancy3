pub(crate) mod aarch64;
mod constants;
mod fallback;
mod frame;
mod function;
mod inst;
mod isel;
mod layout;
mod printer;
mod regalloc;
mod schedule;
mod x64;

use crate::hir::ssa::VerifiedBody;
use crate::module::Module;
use crate::{CompileError, debug::FunctionInfo};
pub(crate) use fallback::Sites;
use function::Function;
use inst::TargetInst;
use splice_target::{Arch, Target};
#[cfg(test)]
pub(crate) use x64::disassemble;

/// The most values one function may have: both backends number them as
/// regalloc2 virtual registers. The inliner refuses larger expansions.
pub(crate) const MAX_VALUES: usize = regalloc2::VReg::MAX;

/// Machine code for one target, ready to install.
pub(crate) struct Code {
    pub bytes: Vec<u8>,
    /// Workspace size in 64-bit words.
    pub words: usize,
    pub debug: Vec<FunctionInfo>,
    /// The numeric fallback sites the code addresses.
    pub sites: fallback::Sites,
}

/// A step of machine lowering, reported as it happens so that filetests
/// read the same lowering that production runs. A listing is built only
/// when the receiver asks for it.
pub(crate) enum Stage<'a> {
    /// One function after instruction selection.
    Select(&'a dyn Fn() -> String),
    /// One function after register allocation.
    Allocate(&'a dyn Fn() -> String),
    /// The block order of the root function.
    Layout(&'a dyn Fn() -> String),
    /// The machine instructions of the module, without the constant pool.
    Assemble(&'a [u8]),
}

/// Where lowering reports its stages.
pub(crate) type Stages<'a> = &'a mut dyn FnMut(Stage<'_>);

/// The stages as `lancy::lir` trace events; a listing is printed only when
/// the event is enabled.
pub(crate) fn trace(stage: Stage<'_>) {
    const TARGET: &str = "lancy::lir";
    match stage {
        Stage::Select(text) => tracing::trace!(target: TARGET, "selected\n{}", text()),
        Stage::Allocate(text) => tracing::trace!(target: TARGET, "allocated\n{}", text()),
        Stage::Layout(text) => tracing::trace!(target: TARGET, "layout\n{}", text()),
        Stage::Assemble(code) => tracing::trace!(target: TARGET, bytes = code.len(), "assembled"),
    }
}

/// Both backends compile on every host; only running the code needs a
/// matching machine. `debug` keeps source points and outlined calls.
pub(crate) fn emit(
    target: &Target,
    source: Module<VerifiedBody<'_>>,
    debug: bool,
    stages: Stages<'_>,
) -> Result<Code, CompileError> {
    Ok(match target.arch() {
        Arch::X86_64 => {
            let code = x64::emit(target, source, debug, stages)?;
            Code {
                bytes: code.bytes,
                words: code.words,
                debug: code.debug,
                sites: code.sites,
            }
        }
        Arch::Aarch64 => aarch64::emit(target, source, debug, stages)?,
    })
}

/// The targets whose machine listings filetests check: this machine's for
/// its own architecture, a fixed one for the other.
#[cfg(test)]
pub(crate) fn filetest_target(arch: Arch) -> Result<Target, String> {
    if Arch::host() == Some(arch) {
        return Target::host().map_err(|error| error.to_string());
    }
    let target = match arch {
        Arch::X86_64 => Target::from_profile(splice_target::Profile::Avx2),
        Arch::Aarch64 => Target::aarch64(
            splice_target::Arm64Features {
                sve: true,
                sve2: true,
            },
            32,
        ),
    };
    target.map_err(|error| error.to_string())
}
