mod function;
pub mod inst;
#[cfg(test)]
mod tests;

pub(crate) use function::VerifiedBody;
pub use function::{Block, Call, Edge, Function, Inst, PacketCall};
pub use inst::{
    Binary, Lane, Op, RustContextImport, RustImport, RustPacketImport, Scale, Simd, Terminator,
    Ternary, Type, Unary, Value, VectorAddress,
};
