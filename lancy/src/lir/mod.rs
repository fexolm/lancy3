mod function;
mod inst;
mod isel;
mod regalloc;
mod x64;

use function::Function;
use inst::TargetInst;

pub(crate) use x64::{Features, emit};
