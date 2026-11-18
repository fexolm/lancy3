//! Builders receive staged operations directly from a kernel. A code generator
//! may select its native IR or materialize the portable Splice function.
pub use crate::hir::FunctionBuilder;
#[doc(hidden)]
pub mod scalar;
mod specialization;
pub use specialization::{ComptimeKey, KeyArgument, MaybeKey, Specialization};

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

    /// Backends that retain callee definitions can reuse a successful specialization.
    /// Custom streaming builders may continue to stage every call independently.
    #[doc(hidden)]
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> Result<Reg, String> {
        let (arity, kinds) = key.signature();
        self.call_kernel(arity, kinds, args, buffers, body)
    }
}

/// Compound operations used by staged kernel bodies.
impl dyn KernelBuilder + '_ {
    /// Snapshot a register before a later operation can overwrite it.
    pub fn copy(&mut self, value: Reg) -> Reg {
        let dst = self.local();
        self.assign(dst, value);
        dst
    }

    /// Continue in a fresh block when the condition holds.
    pub fn branch_if_false(&mut self, cond: Reg, otherwise: Label) {
        let next = self.label();
        self.branch(cond, next, otherwise);
        self.mark(next);
    }
}

/// Explicitly materialize a verified portable IR for inspection or reuse.
pub fn materialize(arity: usize, kinds: &[BufferKind], body: Body<'_>) -> Result<Function, String> {
    let mut builder = FunctionBuilder::with_buffers(arity, kinds.to_vec());
    body(&mut builder)?;
    builder.finish()
}
