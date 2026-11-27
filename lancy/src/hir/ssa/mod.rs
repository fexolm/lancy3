mod function;
pub(crate) mod inst;
#[cfg(test)]
mod tests;

pub(crate) use function::VerifiedBody;
pub use function::{Block, Call, Edge, Function, Inst, SliceParam};
#[cfg(test)]
pub(crate) use function::{InterpretError, buffer_address};
pub use inst::{
    Binary, Lane, Op, RustEntry, RustFunction, Scale, Simd, Terminator, Ternary, Type, Unary,
    Value, VectorAddress,
};
pub(crate) use inst::{Effects, TrapKind};
