//! Builders receive staged operations directly from a kernel. A code generator
//! may select its native IR or materialize the portable Splice function.
pub use crate::hir::FunctionBuilder;
#[doc(hidden)]
pub mod evaluation;
#[doc(hidden)]
pub mod methods;
#[doc(hidden)]
pub mod native;
#[doc(hidden)]
pub mod scalar;
#[doc(hidden)]
pub mod source;
mod specialization;
pub use specialization::{ComptimeKey, KeyArgument, MaybeKey, Specialization};

use crate::hir::Intrinsic;
use crate::hir::{
    Binary, BufferKind, Constant, ContextSlice, Function, Label, Reg, ScalarType, Simd, Slice,
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
    /// The registers of buffer parameter `buffer`, after the scalars.
    pub const fn slice(&self, buffer: usize) -> Slice {
        Slice::param(self.params.len(), buffer)
    }
    /// The type of parameter register `r`, if it is one.
    pub fn param_type(&self, r: Reg) -> Option<ValueType> {
        let index = r.0 as usize;
        if let Some(&ty) = self.params.get(index) {
            return Some(ValueType::Scalar(ty));
        }
        let slices = index - self.params.len();
        match self.buffers.get(slices / 2) {
            Some(&kind) if slices.is_multiple_of(2) => Some(ValueType::Ptr(kind)),
            Some(_) => Some(ValueType::Scalar(ScalarType::Index)),
            None => None,
        }
    }
    /// The number of parameter registers.
    pub const fn registers(&self) -> usize {
        self.params.len() + 2 * self.buffers.len()
    }
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
    /// The overflow mode the build options impose on kernel arithmetic, or
    /// `None` when each kernel follows its crate (see
    /// [`BuildOptions::overflow_checks`](crate::BuildOptions::overflow_checks)).
    fn overflow_checks(&self) -> Option<bool> {
        None
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
    /// The slice of buffer parameter `index`: its two registers.
    fn buffer(&self, index: usize) -> crate::Result<Slice>;
    fn reg_type(&self, r: Reg) -> Option<ValueType>;
    fn local(&mut self, ty: ValueType) -> crate::Result<Reg>;
    fn simd(&mut self, inst: Simd, args: &[Reg]) -> crate::Result<Option<Reg>> {
        self.call_intrinsic(Intrinsic::Simd(inst), args)
    }
    fn constant(&mut self, value: Constant) -> crate::Result<Reg>;
    /// Rust's `value as to`.
    fn cast(&mut self, value: Reg, to: ScalarType) -> crate::Result<Reg>;
    /// `&slice[s..e]` for `e = min(end, len)` and `s = min(start, e)`, with
    /// no trap: the part of `slice` the range selects, clamped so that it
    /// stays within `slice` whatever `start` and `end` hold. This is the
    /// only way to derive a slice: its pointer and length come from one
    /// operation, and the verifier accepts a slice operand only as such a
    /// pair or a buffer parameter. The checked forms (`subslice` and the
    /// others on `dyn KernelBuilder`) trap first, so the minima are `start`
    /// and `end`; they show a backend that the length is a nonnegative
    /// difference within `len`.
    ///
    fn slice_range(&mut self, slice: Slice, start: Reg, end: Reg) -> crate::Result<Slice>;
    /// Element `index` of `slice`; traps unless `index` is below its length.
    fn load(&mut self, slice: Slice, index: Reg) -> crate::Result<Reg>;
    /// Write element `index` of `slice`; traps unless `index` is below its length.
    fn store(&mut self, slice: Slice, index: Reg, value: Reg) -> crate::Result<()>;
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
    /// operation of a Rust context argument, on scalar words and slices,
    /// returning a `result`.
    #[doc(hidden)]
    fn rust_call(
        &mut self,
        target: crate::hir::RustTarget,
        _args: &[Reg],
        _slices: &[ContextSlice],
        _result: crate::hir::ScalarType,
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
        buffers: &[Slice],
        body: Body<'_>,
    ) -> crate::Result<Reg>;

    /// Backends that retain callee definitions can reuse a successful specialization.
    /// Custom streaming builders may continue to stage every call independently.
    #[doc(hidden)]
    fn call_specialized(
        &mut self,
        key: Specialization,
        args: &[Reg],
        buffers: &[Slice],
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
        buffers: &[Slice],
    ) -> crate::Result<Reg> {
        let callee = Arc::clone(function);
        self.call_kernel(
            function.signature(),
            args,
            buffers,
            Box::new(move |b| crate::hir::replay(&callee, b)),
        )
    }

    /// Emit one built-in operation. Stores have no result register.
    fn call_intrinsic(
        &mut self,
        intrinsic: Intrinsic,
        _args: &[Reg],
    ) -> crate::Result<Option<Reg>> {
        Err(crate::BuildError::unsupported(format!(
            "intrinsic {intrinsic:?} is not supported by builder {}",
            std::any::type_name::<Self>()
        )))
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
    /// `&slice[start..end]`, or `&slice[start..=end]` when `inclusive`:
    /// traps with [`crate::Trap::Bounds`], as Rust's slice index panics,
    /// unless `start <= end` and the end is within the length.
    pub fn subslice(
        &mut self,
        slice: Slice,
        start: Reg,
        end: Reg,
        inclusive: bool,
    ) -> crate::Result<Slice> {
        let (inside, end) = if inclusive {
            let one = self.constant(Constant::from(1usize))?;
            let inside = self.binary(Binary::Lt, end, slice.len)?;
            (inside, self.binary(Binary::Add, end, one)?)
        } else {
            (self.binary(Binary::Le, end, slice.len)?, end)
        };
        let ordered = self.binary(Binary::Le, start, end)?;
        let valid = self.binary(Binary::And, inside, ordered)?;
        let (next, fail) = (self.label()?, self.label()?);
        self.branch(valid, next, fail)?;
        self.mark(fail)?;
        self.trap_bounds()?;
        self.mark(next)?;
        self.slice_range(slice, start, end)
    }

    /// `text.is_char_boundary(index)` for the UTF-8 bytes `text`: the end,
    /// or a byte inside that does not continue a character. It reads no
    /// byte at or past the end.
    pub fn is_char_boundary(&mut self, text: Slice, index: Reg) -> crate::Result<Reg> {
        let result = self.local(ValueType::Scalar(ScalarType::Bool))?;
        let (inside, outside, done) = (self.label()?, self.label()?, self.label()?);
        let within = self.binary(Binary::Lt, index, text.len)?;
        self.branch(within, inside, outside)?;
        self.mark(inside)?;
        let byte = self.load(text, index)?;
        let top = self.constant(Constant::from(0xc0u8))?;
        let continuation = self.constant(Constant::from(0x80u8))?;
        let top = self.binary(Binary::And, byte, top)?;
        let starts = self.binary(Binary::Ne, top, continuation)?;
        self.assign(result, starts)?;
        self.jump(done)?;
        self.mark(outside)?;
        let end = self.binary(Binary::Eq, index, text.len)?;
        self.assign(result, end)?;
        self.jump(done)?;
        self.mark(done)?;
        Ok(result)
    }

    /// `&text[start..end]` of a `&str` (or `..=` when `inclusive`): the
    /// range check of `subslice`, then a trap unless both
    /// ends are character boundaries, as Rust's `str` index panics.
    pub fn str_subslice(
        &mut self,
        text: Slice,
        start: Reg,
        end: Reg,
        inclusive: bool,
    ) -> crate::Result<Slice> {
        let part = self.subslice(text, start, end, inclusive)?;
        let end = self.binary(Binary::Add, start, part.len)?;
        let first = self.is_char_boundary(text, start)?;
        let last = self.is_char_boundary(text, end)?;
        let valid = self.binary(Binary::And, first, last)?;
        let (next, fail) = (self.label()?, self.label()?);
        self.branch(valid, next, fail)?;
        self.mark(fail)?;
        self.trap_bounds()?;
        self.mark(next)?;
        Ok(part)
    }

    /// `slice.split_at(mid)`: traps unless `mid` is within the length.
    pub fn split_at(&mut self, slice: Slice, mid: Reg) -> crate::Result<(Slice, Slice)> {
        let zero = self.constant(Constant::from(0usize))?;
        let head = self.subslice(slice, zero, mid, false)?;
        Ok((head, self.slice_range(slice, mid, slice.len)?))
    }

    /// `text.split_at(mid)` of a `&str`: the check of
    /// `split_at`, then a trap unless `mid` is a character
    /// boundary, as Rust's `str::split_at` panics.
    pub fn str_split_at(&mut self, text: Slice, mid: Reg) -> crate::Result<(Slice, Slice)> {
        let halves = self.split_at(text, mid)?;
        let boundary = self.is_char_boundary(text, mid)?;
        let (next, fail) = (self.label()?, self.label()?);
        self.branch(boundary, next, fail)?;
        self.mark(fail)?;
        self.trap_bounds()?;
        self.mark(next)?;
        Ok(halves)
    }

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
        crate::validate::validated(signature, body, options.overflow_checks),
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
