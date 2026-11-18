//! Builders receive staged operations directly from a kernel. A code generator
//! may select its native IR or materialize the portable Splice function.
mod builder;
pub use builder::FunctionBuilder;
#[doc(hidden)]
pub mod scalar;

use crate::hir::{Binary, Buffer, BufferKind, Function, Label, Reg};

/// A function body is staged once against the builder selected by its backend.
/// Borrowed comptime parameters may live only for the duration of the build.
pub type Body<'a> = Box<dyn FnOnce(&mut dyn KernelBuilder) -> Result<(), String> + 'a>;

/// Names from a kernel declaration, in the runtime calling convention's order.
#[derive(Debug)]
pub struct KernelSource {
    pub name: &'static str,
    pub scalars: &'static [&'static str],
    pub buffers: &'static [&'static str],
}

/// Symbolic scalar and label handles are local to the function being staged.
/// A nested body receives a fresh builder for its own function and arguments.
pub trait KernelBuilder {
    fn set_source(&mut self, _source: &'static KernelSource) {}
    fn arg(&self, index: usize) -> Reg;
    fn local(&mut self) -> Reg;
    fn constant(&mut self, value: i32) -> Reg;
    fn buffer_len(&mut self, buffer: Buffer) -> Reg;
    fn load(&mut self, buffer: Buffer, index: Reg) -> Reg;
    fn store(&mut self, buffer: Buffer, index: Reg, value: Reg);
    fn immediate_condition(&self, reg: Reg) -> Option<bool>;
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> Reg;
    fn assign(&mut self, dst: Reg, src: Reg);
    fn label(&mut self) -> Label;
    fn mark(&mut self, label: Label);
    fn jump(&mut self, label: Label);
    fn branch(&mut self, cond: Reg, yes: Label, no: Label);
    fn ret(&mut self, value: Reg);
    fn call_kernel(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> Result<Reg, String>;
}

/// Explicitly materialize a verified portable IR for inspection or reuse.
pub fn materialize(arity: usize, kinds: &[BufferKind], body: Body<'_>) -> Result<Function, String> {
    let mut builder = FunctionBuilder::with_buffers(arity, kinds.to_vec());
    body(&mut builder)?;
    builder.finish()
}
