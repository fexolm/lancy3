//! The Splice language frontend. Staging a kernel constructs a verified function
//! from its `#[comptime]` arguments; a code generator compiles that function.
//! The same function may be compiled more than once with different generators.
#![forbid(unsafe_code)]

extern crate self as splice;

pub mod frontend;
pub mod hir;

pub use hir::{BufferArg, BufferKind, Function, validate_buffers};
pub use splice_macros::kernel;

/// Compile staged Splice IR into an owned, callable kernel.
pub trait Codegen {
    type Kernel: Executable;
    fn compile(function: &Function) -> Result<Self::Kernel, String>;
}

/// The portable calling convention. Scalars and slices have separate
/// argument lists, each following declaration order. An error stops execution;
/// completed stores remain visible to the caller.
pub trait Executable {
    fn call(&self, args: &[i32], buffers: &mut [BufferArg<'_>]) -> Result<i32, String>;
}
