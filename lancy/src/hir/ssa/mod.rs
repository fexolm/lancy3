mod function;
mod instruction;
#[cfg(test)]
mod tests;

pub(crate) use function::VerifiedBody;
pub use function::{Block, Call, Edge, Function, Inst};
pub use instruction::{Binary, Op, Scale, Terminator, Type, Value};
