pub mod aarch64;
mod function;
mod inst;
mod isel;
mod regalloc;
#[cfg(target_arch = "x86_64")]
mod x64;

use function::Function;
use inst::TargetInst;

#[cfg(target_arch = "aarch64")]
pub(crate) use aarch64::emit;
#[cfg(target_arch = "x86_64")]
pub(crate) use x64::emit;
