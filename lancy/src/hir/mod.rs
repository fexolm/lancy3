mod error;
pub mod ssa;
pub mod transforms;

pub use error::IrError;
pub use ssa::{Binary, Block, Call, Edge, Function, Inst, Op, Scale, Terminator, Type, Value};
