mod function;
pub mod inst;
#[cfg(test)]
mod tests;

pub(crate) use function::VerifiedBody;
pub use function::{Block, Call, Edge, Function, Inst};
pub use inst::{
    Binary, Lane, Op, Scale, Simd, Terminator, Ternary, Type, Unary, Value, VectorAddress,
};
