mod error;
#[cfg(any(test, target_arch = "aarch64"))]
pub(crate) mod eval;
pub(crate) mod operation;
pub mod ssa;
pub mod transforms;

pub use error::IrError;
pub use ssa::{
    Binary, Block, Call, Edge, Function, Inst, Lane, Op, Scale, Simd, Terminator, Ternary, Type,
    Unary, Value, VectorAddress,
};
