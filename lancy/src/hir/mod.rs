mod error;
pub(crate) mod operation;
pub(crate) mod ssa;
pub mod text;
pub(crate) mod transforms;

/// Width-independent proofs (`Count::at_most`) hold for every target width
/// in this range.
pub(crate) use splice_target::{MAX_VECTOR_BYTES, MIN_VECTOR_BYTES};

pub use error::IrError;
pub use ssa::{
    Binary, Block, Call, Edge, Function, Inst, Lane, Op, RustEntry, RustFunction, Scale, Simd,
    SliceParam, Terminator, Ternary, Type, Unary, Value, VectorAddress,
};
pub(crate) use ssa::{Effects, TrapKind};
