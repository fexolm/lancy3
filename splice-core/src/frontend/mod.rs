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

use crate::hir::{
    Binary, Buffer, BufferKind, Constant, ContextSlice, Function, Label, Reg, ScalarType, Simd,
    ValueType,
};
use std::sync::Arc;

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

/// The scalar and buffer parameters of a kernel, in declaration order,
/// and the type of its result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Signature<'a> {
    pub params: &'a [ScalarType],
    pub result: ScalarType,
    pub buffers: &'a [BufferKind],
}

impl<'a> Signature<'a> {
    pub const fn new(
        params: &'a [ScalarType],
        result: ScalarType,
        buffers: &'a [BufferKind],
    ) -> Self {
        Self {
            params,
            result,
            buffers,
        }
    }
}

/// Receives the operations of a kernel body as it is staged.
///
/// Register and label handles are local to the function being staged: the
/// parameters are the first registers and every new handle takes the next
/// index. A nested body receives a fresh builder for its own function.
/// Kernels and backends never see an unchecked stream: [`Validating`]
/// checks each operation before it reaches a backend's builder, so an
/// implementation may assume that every operand is valid. An error stops
/// staging; the build reports it.
///
/// [`Validating`]: crate::validate::Validating
pub trait KernelBuilder {
    fn set_source(&mut self, _source: &'static KernelSource) {}
    fn debug_enabled(&self) -> bool {
        false
    }
    fn debug_point(
        &mut self,
        _location: crate::debug::Location,
        _variables: &[crate::debug::Variable],
    ) -> crate::Result<()> {
        Ok(())
    }
    /// The register of scalar parameter `index`.
    fn arg(&self, index: usize) -> crate::Result<Reg> {
        u32::try_from(index)
            .map(Reg)
            .map_err(|_| crate::BuildError::verify("invalid parameter index"))
    }
    fn reg_type(&self, r: Reg) -> Option<ValueType>;
    fn local(&mut self, ty: ValueType) -> crate::Result<Reg>;
    fn simd(&mut self, _inst: Simd, _args: &[Reg]) -> crate::Result<Option<Reg>> {
        Err(crate::BuildError::unsupported(
            "this builder does not support SIMD",
        ))
    }
    fn constant(&mut self, value: Constant) -> crate::Result<Reg>;
    /// Rust's `value as to`.
    fn cast(&mut self, value: Reg, to: ScalarType) -> crate::Result<Reg>;
    fn buffer_len(&mut self, buffer: Buffer) -> crate::Result<Reg>;
    fn load(&mut self, buffer: Buffer, index: Reg) -> crate::Result<Reg>;
    fn store(&mut self, buffer: Buffer, index: Reg, value: Reg) -> crate::Result<()>;
    fn binary(&mut self, op: Binary, a: Reg, b: Reg) -> crate::Result<Reg>;
    fn assign(&mut self, dst: Reg, src: Reg) -> crate::Result<()>;
    fn label(&mut self) -> crate::Result<Label>;
    fn mark(&mut self, label: Label) -> crate::Result<()>;
    fn jump(&mut self, label: Label) -> crate::Result<()>;
    fn branch(&mut self, cond: Reg, yes: Label, no: Label) -> crate::Result<()>;
    fn ret(&mut self, value: Reg) -> crate::Result<()>;
    fn trap_bounds(&mut self) -> crate::Result<()> {
        Err(crate::BuildError::unsupported(
            "this builder does not support an explicit bounds trap",
        ))
    }
    /// Run Rust code at this point of the kernel: a static function or an
    /// operation of a Rust context argument, on `i32` words and slices.
    #[doc(hidden)]
    fn rust_call(
        &mut self,
        target: crate::hir::RustTarget,
        _args: &[Reg],
        _slices: &[ContextSlice],
    ) -> crate::Result<Reg> {
        Err(crate::BuildError::staging(format!(
            "backend cannot link Rust code {target:?}"
        )))
    }
    /// Stage `body` as a nested kernel and call it.
    fn call_kernel(
        &mut self,
        signature: Signature<'_>,
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
        self.call_kernel(key.signature(), args, buffers, body)
    }

    /// Call a portable function. Builders that keep callee definitions
    /// lower each shared function once; by default it is staged again.
    fn call_function(
        &mut self,
        function: &Arc<Function>,
        args: &[Reg],
        buffers: &[Buffer],
    ) -> crate::Result<Reg> {
        let callee = Arc::clone(function);
        self.call_kernel(
            function.signature(),
            args,
            buffers,
            Box::new(move |b| crate::hir::replay(&callee, b)),
        )
    }
}

/// Compound operations used by staged kernel bodies.
impl dyn KernelBuilder + '_ {
    /// Snapshot a register before a later operation can overwrite it.
    pub fn copy(&mut self, value: Reg) -> crate::Result<Reg> {
        let ty = self
            .reg_type(value)
            .ok_or_else(|| crate::BuildError::verify("invalid register"))?;
        let dst = self.local(ty)?;
        self.assign(dst, value)?;
        Ok(dst)
    }

    /// Continue in a fresh block when the condition holds.
    pub fn branch_if_false(&mut self, cond: Reg, otherwise: Label) -> crate::Result<()> {
        let next = self.label()?;
        self.branch(cond, next, otherwise)?;
        self.mark(next)
    }
}

/// Explicitly materialize a verified portable IR for inspection or reuse.
pub fn materialize(
    signature: Signature<'_>,
    body: Body<'_>,
    options: crate::BuildOptions,
) -> crate::Result<Function> {
    record(
        signature,
        crate::validate::validated(signature, body),
        options,
    )
}

/// Record a body that validates itself (see [`crate::validate::validated`]).
pub(crate) fn record(
    signature: Signature<'_>,
    body: Body<'_>,
    options: crate::BuildOptions,
) -> crate::Result<Function> {
    let _span = tracing::debug_span!(target: "splice::compile", "materialize_ir").entered();
    let mut builder = FunctionBuilder::new(options, signature);
    body(&mut builder)?;
    Ok(builder.finish_validated())
}
