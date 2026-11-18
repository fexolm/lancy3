//! High-level intermediate representations for scalar control flow and column projections.

pub mod function;
pub mod projection;
pub(crate) mod vectorize;

pub use function::{
    Binary, Buffer, BufferArg, BufferKind, Function, FunctionBuilder, Instr, Label, Reg,
};
pub use projection::{Assumption, BuildError, BuildValue, Builder, Op, Program, Value};
