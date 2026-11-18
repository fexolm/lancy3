//! Build the portable instruction stream at specialization time. The attribute
//! emits calls to the same builder available to handwritten frontends.

mod builder;
pub use builder::FunctionBuilder;
#[doc(hidden)]
pub mod scalar;
