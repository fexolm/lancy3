mod block;
mod instruction;
mod ssa;
mod x64;

use block::{BlockId, EdgeId};
use instruction::{CallSiteId, FunctionId};

pub(crate) use x64::emit;
