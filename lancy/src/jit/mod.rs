//! Install machine code, invoke kernels, and manage background specialization.
mod memory;
mod scalar;
mod worker;

pub(crate) use memory::Executable;
pub use scalar::{
    AdaptiveBufferKernel, AdaptiveScalar, ScalarCaller, ScalarFn, ScalarFn3, ScalarKernel,
};

/// The currently available tier, or the state of background compilation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tier {
    Baseline,
    Compiling,
    Specialized,
    Generic,
    Failed,
}
