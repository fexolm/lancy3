mod error;
#[cfg(any(test, target_arch = "aarch64"))]
pub(crate) mod eval;
mod integer;
pub(crate) mod operation;
pub mod ssa;
pub mod transforms;

pub(crate) const MIN_VECTOR_BYTES: usize = 16;
pub(crate) const MAX_VECTOR_BYTES: usize = 64;

pub use error::IrError;
pub use ssa::{
    Binary, Block, Call, Edge, Function, Inst, Lane, Op, PacketCall, RustContextImport, RustImport,
    RustPacketImport, Scale, Simd, Terminator, Ternary, Type, Unary, Value, VectorAddress,
};
