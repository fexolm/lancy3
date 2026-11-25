mod function;
pub mod inst;
#[cfg(test)]
mod tests;

pub(crate) use function::VerifiedBody;
pub use function::{Block, Call, Edge, Function, Inst, SliceParam};
#[cfg(test)]
pub(crate) use function::{InterpretError, buffer_address};
pub use inst::{
    Access, Binary, Effects, Lane, Op, RustEntry, RustFunction, Scale, Simd, Terminator, Ternary,
    TrapKind, Type, Unary, Value, VectorAddress,
};
