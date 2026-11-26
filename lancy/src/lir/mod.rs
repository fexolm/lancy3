pub mod aarch64;
mod constants;
mod fallback;
mod function;
mod inst;
mod isel;
mod layout;
mod regalloc;
mod schedule;
mod x64;

use crate::hir::ssa::VerifiedBody;
use crate::module::Module;
use crate::{CompileError, jit::DebugFunction};
pub(crate) use fallback::Sites;
use function::Function;
use inst::TargetInst;
use splice_target::{Arch, Target};

/// The most values one function may have: both backends number them as
/// regalloc2 virtual registers. The inliner refuses larger expansions.
pub(crate) const MAX_VALUES: usize = regalloc2::VReg::MAX;

/// Machine code for one target, ready to install.
pub(crate) struct Code {
    pub bytes: Vec<u8>,
    /// Workspace size in 64-bit words.
    pub words: usize,
    pub debug: Vec<DebugFunction>,
    /// The numeric fallback sites the code addresses.
    pub sites: fallback::Sites,
}

/// Both backends compile on every host; only running the code needs a
/// matching machine.
pub(crate) fn emit(
    target: &Target,
    source: Module<VerifiedBody<'_>>,
) -> Result<Code, CompileError> {
    emit_mode(target, source, false)
}

pub(crate) fn emit_debug(
    target: &Target,
    source: Module<VerifiedBody<'_>>,
) -> Result<Code, CompileError> {
    emit_mode(target, source, true)
}

fn emit_mode(
    target: &Target,
    source: Module<VerifiedBody<'_>>,
    debug: bool,
) -> Result<Code, CompileError> {
    Ok(match target.arch() {
        Arch::X86_64 => {
            let code = if debug {
                x64::emit_debug(target, source)?
            } else {
                x64::emit(target, source)?
            };
            Code {
                bytes: code.bytes,
                words: code.words,
                debug: code.debug,
                sites: code.sites,
            }
        }
        Arch::Aarch64 => {
            let code = if debug {
                aarch64::emit_debug(target, source)?
            } else {
                aarch64::emit(target, source)?
            };
            Code {
                bytes: code.bytes,
                words: code.words,
                debug: code.debug,
                sites: code.sites,
            }
        }
    })
}

/// The targets whose machine listings filetests check: this machine's for
/// its own architecture, a fixed one for the other.
#[cfg(test)]
fn filetest_target(arch: Arch) -> Result<Target, String> {
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

/// Check the x64 and AArch64 listings of an optimized module. `@layout`
/// follows the x64 blocks.
#[cfg(test)]
pub(crate) fn check_machine(
    case: &mut crate::filetests::Case<'_>,
    module: Module<VerifiedBody<'_>>,
) -> crate::filetests::Result<()> {
    let root = module.functions[module.call_graph.root()].function;
    x64::check_machine(case, &filetest_target(Arch::X86_64)?, module)?;
    let module = root
        .verified_module()
        .map_err(|error| format!("verify: {error}"))?;
    aarch64::check_machine(case, &filetest_target(Arch::Aarch64)?, module)
}
