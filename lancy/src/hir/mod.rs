//! Canonical Lancy SSA. Successor arguments supply basic block parameters.
mod passes;
pub mod ssa;
pub use passes::optimize;

pub(crate) use crate::input::check_forwarding;
pub use crate::input::{Binary, Buffer, BufferArg, BufferKind};
pub use ssa::*;
