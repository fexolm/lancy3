//! Splice stages Rust kernels into an enabled code generation backend.

pub use splice_core::*;

#[cfg(feature = "cranelift")]
pub use splice_cranelift::Cranelift;
#[cfg(feature = "lancy")]
pub use splice_lancy::Lancy;
#[cfg(feature = "llvm")]
pub use splice_llvm::Llvm;
#[cfg(feature = "text")]
pub use splice_text::Text;
