//! Builders receive staged operations directly from a kernel. A code generator
//! may select its native IR or materialize the portable Splice function.
pub use crate::hir::FunctionBuilder;
#[doc(hidden)]
pub mod evaluation;
#[doc(hidden)]
pub mod native;
#[doc(hidden)]
pub mod scalar;
#[doc(hidden)]
pub mod source;
mod specialization;
pub use specialization::{ComptimeKey, KeyArgument, MaybeKey, Specialization};

use crate::hir::{Binary, Buffer, BufferKind, ContextSlice, Function, Label, Reg, Simd, ValueType};

/// A function body is staged once against the builder selected by its backend.
/// Borrowed comptime parameters may live only for the duration of the build.
pub type Body<'a> = Box<dyn FnOnce(&mut dyn KernelBuilder) -> crate::Result<()> + 'a>;

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
    fn debug_enabled(&self) -> bool {
        false
    }
    fn debug_point(
        &mut self,
        _location: crate::debug::Location,
        _variables: &[crate::debug::Variable],
    ) {
    }
    fn arg(&self, index: usize) -> Reg;
    fn local(&mut self) -> Reg;
    /// Scalar-only builders use the default type for every register.
    fn reg_type(&self, _r: Reg) -> Option<ValueType> {
        Some(ValueType::Scalar)
    }
    /// Builders with typed storage override this for vector and mask locals.
    fn typed_local(&mut self, _ty: ValueType) -> Reg {
        self.local()
    }
    fn simd(&mut self, _inst: Simd, _args: &[Reg]) -> crate::Result<Option<Reg>> {
        Err(crate::BuildError::unsupported(
            "this builder does not support SIMD",
        ))
    }
    fn constant(&mut self, value: i32) -> Reg;
    fn buffer_len(&mut self, buffer: Buffer) -> Reg;
    fn load(&mut self, buffer: Buffer, index: Reg) -> Reg;
    fn store(&mut self, buffer: Buffer, index: Reg, value: Reg);
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> Reg;
    fn assign(&mut self, dst: Reg, src: Reg);
    fn label(&mut self) -> Label;
    fn mark(&mut self, label: Label);
    fn jump(&mut self, label: Label);
    fn branch(&mut self, cond: Reg, yes: Label, no: Label);
    fn ret(&mut self, value: Reg);
    fn trap_bounds(&mut self) -> crate::Result<()> {
        Err(crate::BuildError::unsupported(
            "this builder does not support an explicit bounds trap",
        ))
    }
    fn rust_context_call(
        &mut self,
        context: Buffer,
        operation: u32,
        args: &[Reg],
    ) -> crate::Result<Reg> {
        self.rust_context_call_with_slices(context, operation, args, &[])
    }
    #[doc(hidden)]
    fn rust_context_call_with_slices(
        &mut self,
        _context: Buffer,
        _operation: u32,
        _args: &[Reg],
        _slices: &[ContextSlice],
    ) -> crate::Result<Reg> {
        Err(crate::BuildError::unsupported(
            "this builder does not support Rust context operations",
        ))
    }
    #[doc(hidden)]
    fn rust_static_call(
        &mut self,
        symbol: &'static crate::StaticRustSymbol,
        _args: &[Reg],
        _slices: &[ContextSlice],
    ) -> crate::Result<Reg> {
        Err(crate::BuildError::staging(format!(
            "backend cannot link Rust operation {}",
            symbol.name()
        )))
    }
    /// Emit a typed, effectful Rust operation at this point in the function.
    fn rust_call(&mut self, import: crate::hir::RustCall, _args: &[Reg]) -> crate::Result<Reg> {
        Err(crate::BuildError::staging(format!(
            "backend cannot link Rust import {}",
            import.symbol.name
        )))
    }
    #[doc(hidden)]
    fn rust_packet_call(
        &mut self,
        import: &'static crate::hir::RustPacketSymbol,
        _args: &[Reg],
    ) -> crate::Result<Reg> {
        Err(crate::BuildError::staging(format!(
            "backend cannot link Rust import {}",
            import.name
        )))
    }
    /// Reject mismatched call arguments before staging the callee body.
    fn call_kernel(
        &mut self,
        arity: usize,
        kinds: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> crate::Result<Reg>;

    /// Backends that retain callee definitions can reuse a successful specialization.
    /// Custom streaming builders may continue to stage every call independently.
    #[doc(hidden)]
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        buffers: &[Buffer],
        body: Body<'_>,
    ) -> crate::Result<Reg> {
        let (arity, kinds) = key.signature();
        self.call_kernel(arity, kinds, args, buffers, body)
    }
}

/// Compound operations used by staged kernel bodies.
impl dyn KernelBuilder + '_ {
    /// Snapshot a register before a later operation can overwrite it.
    pub fn copy(&mut self, value: Reg) -> Reg {
        let dst = self.typed_local(self.reg_type(value).unwrap_or(ValueType::Scalar));
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
pub fn materialize(arity: usize, kinds: &[BufferKind], body: Body<'_>) -> crate::Result<Function> {
    materialize_with_options(arity, kinds, body, crate::BuildOptions::default())
}

pub fn materialize_with_options(
    arity: usize,
    kinds: &[BufferKind],
    body: Body<'_>,
    options: crate::BuildOptions,
) -> crate::Result<Function> {
    let _span = tracing::debug_span!(target: "splice::compile", "materialize_ir").entered();
    let mut builder = FunctionBuilder::with_options(arity, kinds.to_vec(), options);
    body(&mut builder)?;
    builder.finish()
}
