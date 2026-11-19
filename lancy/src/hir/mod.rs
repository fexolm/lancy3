mod error;
pub mod ssa;
pub mod transforms;

pub use error::IrError;
pub use ssa::{
    Binary, Block, Call, Edge, Function, Inst, Lane, Op, Scale, Simd, Terminator, Ternary, Type,
    Unary, Value, VectorAddress,
};
