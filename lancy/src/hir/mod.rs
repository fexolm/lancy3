mod error;
pub(crate) mod eval;
mod integer;
pub(crate) mod operation;
pub mod ssa;
pub mod transforms;

/// Width-independent proofs (`Count::at_most`) hold for every target width
/// in this range.
pub(crate) use splice_target::{MAX_VECTOR_BYTES, MIN_VECTOR_BYTES};

pub use error::IrError;
pub use ssa::{
    Binary, Block, Call, Edge, Function, Inst, Lane, Op, PacketCall, RustContextImport, RustImport,
    RustPacketImport, Scale, Simd, Terminator, Ternary, Type, Unary, Value, VectorAddress,
};
