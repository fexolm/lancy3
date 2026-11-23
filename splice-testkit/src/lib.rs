//! One place that lists the Splice backends, so every test covers all of
//! them and a new backend is added once.
pub use splice_cranelift::Cranelift;
pub use splice_lancy::Lancy;
pub use splice_llvm::{Llvm, LlvmManualSimd};

/// Run `body` once per backend with `$b` naming the backend type:
/// `each_backend!(B => check::<B>(&ir))`.
#[macro_export]
macro_rules! each_backend {
    ($b:ident => $body:expr) => {{
        {
            type $b = $crate::Lancy;
            $body
        };
        {
            type $b = $crate::Cranelift;
            $body
        };
        {
            type $b = $crate::Llvm;
            $body
        };
        {
            type $b = $crate::LlvmManualSimd;
            $body
        };
    }};
}
