//! The Splice frontend stages a kernel directly into its chosen backend.
//! A portable function can also be materialized for inspection and reuse.
#![deny(unsafe_code)]

extern crate self as splice;

pub mod debug;
pub mod frontend;
pub mod hir;
mod imports;
pub mod simd;
pub mod validate;
#[doc(hidden)]
pub use imports::ImportInvocation;
pub use imports::{ImportArgs, ImportTarget, LinkedImport, RustImport};
#[allow(unsafe_code)]
mod runtime_context;
#[doc(hidden)]
pub use runtime_context::{
    ContextArguments, ContextFrame, ContextOperation, ContextPacketLayout, ContextSignature,
    RuntimeContext, StaticRustCall, StaticRustSymbol, invoke_packet_runtime_context,
    invoke_runtime_context,
};

/// Description of an actual Rust struct field for this monomorphization.
pub struct KernelField {
    pub name: &'static str,
    pub offset: usize,
    pub type_name: fn() -> &'static str,
}
pub trait KernelType {
    const FIELDS: &'static [KernelField];
}

mod error;
mod typed;
pub use error::{BuildError, CallError, SignatureError, TargetError, Trap};
#[doc(hidden)]
pub use error::{build_boundary, call_boundary};
pub use splice_target::{self as target, Target};
pub use typed::{CallArgs, CompileInput, TypedIr, TypedKernel};

pub use frontend::ComptimeKey;
pub use hir::{BufferArg, BufferElement, BufferKind, Function, validate_buffers};
use hir::{Constant, ScalarType};
pub use splice_macros::{ComptimeKey, KernelType, kernel};

/// Result of staging or compiling a kernel.
pub type Result<T> = std::result::Result<T, BuildError>;

/// Selects the optimization and source-level debug information for a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Preserve source statements, variables and calls for native debuggers.
    Debug,
    /// Use the backend's optimized pipeline.
    Release,
}

impl Mode {
    /// Select the mode from the caller's `cfg!(debug_assertions)`.
    pub const fn for_profile(debug_assertions: bool) -> Self {
        if debug_assertions {
            Self::Debug
        } else {
            Self::Release
        }
    }
}

/// How and for which machine a kernel is staged and compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildOptions {
    pub mode: Mode,
    pub target: Target,
}

impl BuildOptions {
    pub const fn new(mode: Mode, target: Target) -> Self {
        Self { mode, target }
    }

    /// The given mode for the target selected by `SPLICE_SIMD` on this machine.
    pub fn host(mode: Mode) -> std::result::Result<Self, TargetError> {
        Ok(Self::new(mode, Target::host()?))
    }

    pub fn with_mode(self, mode: Mode) -> Self {
        Self { mode, ..self }
    }

    pub fn with_target(self, target: Target) -> Self {
        Self { target, ..self }
    }

    pub fn debug(&self) -> bool {
        self.mode == Mode::Debug
    }
}

/// A code generator. A backend compiles a verified portable function, and
/// may stage kernel bodies directly into its own builder instead of
/// recording them first. Kernels reach both through [`Codegen`].
pub trait Backend: Sized {
    /// The value produced by a successful build.
    type Output;
    /// Compile a verified portable function. Source points are available
    /// when the function was staged in debug mode.
    fn compile_function(function: &Function, options: BuildOptions) -> Result<Self::Output>;
    /// Stage `body` with `options`. The body checks its own operations
    /// (see [`validate::Validating`]), so the backend's builder receives
    /// only valid ones. By default the body is recorded as a portable
    /// function and compiled.
    fn stage(
        signature: frontend::Signature<'_>,
        body: frontend::Body<'_>,
        options: BuildOptions,
    ) -> Result<Self::Output> {
        Self::compile_function(&frontend::record(signature, body, options)?, options)
    }
}

/// Build kernels with a [`Backend`]: compile a portable function, or stage
/// a kernel body. Every backend gets the same, validated, operations.
pub trait Codegen: Backend {
    fn compile<I: CompileInput<Self>>(input: &I) -> std::result::Result<I::Output, I::Error> {
        input.compile_with(None)
    }
    /// Override the code generation mode or target of a portable function.
    fn compile_with_options<I: CompileInput<Self>>(
        input: &I,
        options: BuildOptions,
    ) -> std::result::Result<I::Output, I::Error> {
        input.compile_with(Some(options))
    }
    /// Stage a kernel body directly.
    fn build_with_options(
        signature: frontend::Signature<'_>,
        body: frontend::Body<'_>,
        options: BuildOptions,
    ) -> Result<Self::Output> {
        let _span = tracing::debug_span!(target: "splice::compile", "build", backend = std::any::type_name::<Self>()).entered();
        Self::stage(signature, validate::validated(signature, body), options)
    }
}

impl<B: Backend> Codegen for B {}

/// [`Executable::call`] passes `i32` words: the entry's scalars must fit one.
/// The portable calling convention. Scalars and slices have separate
/// argument lists, each following declaration order. Each scalar argument
/// and the result is one word holding the value's [`Constant::bits`]: see
/// [`KernelScalar`] for the encoding. An error stops execution; completed
/// stores remain visible to the caller.
pub trait Executable {
    fn call(
        &self,
        args: &[u64],
        buffers: &mut [BufferArg<'_>],
    ) -> std::result::Result<u64, CallError>;
}

impl<E: Executable + ?Sized> Executable for &E {
    fn call(
        &self,
        args: &[u64],
        buffers: &mut [BufferArg<'_>],
    ) -> std::result::Result<u64, CallError> {
        (**self).call(args, buffers)
    }
}

impl<E: Executable + ?Sized> Executable for Box<E> {
    fn call(
        &self,
        args: &[u64],
        buffers: &mut [BufferArg<'_>],
    ) -> std::result::Result<u64, CallError> {
        (**self).call(args, buffers)
    }
}

/// A Rust type of a kernel scalar and its ABI word: the value's bits in the
/// low `TYPE.bits()` bits and zero above (`bool` is 0 or 1, a float its IEEE
/// bit pattern).
pub trait KernelScalar: Copy + Into<Constant> {
    const TYPE: ScalarType;
    fn to_word(self) -> u64 {
        self.into().bits()
    }
    /// The value of a word of this type; bits above the value are ignored.
    fn from_word(word: u64) -> Self;
}

macro_rules! kernel_scalar {
    ($($ty:ty => $scalar:ident, |$word:ident| $decode:expr;)*) => {$(
        impl KernelScalar for $ty {
            const TYPE: ScalarType = ScalarType::$scalar;
            fn from_word($word: u64) -> Self {
                $decode
            }
        }
    )*};
}
kernel_scalar! {
    bool => Bool, |word| word & 1 != 0;
    i32 => I32, |word| word as i32;
    u32 => U32, |word| word as u32;
    i64 => I64, |word| word as i64;
    u64 => U64, |word| word;
    usize => Index, |word| word as usize;
    f32 => F32, |word| f32::from_bits(word as u32);
    f64 => F64, |word| f64::from_bits(word);
}

/// Check scalar words against the parameter types before a call: every
/// backend reads a word at its type's width, so a word with bits above its
/// value would mean different values to different backends.
pub fn validate_scalars(
    params: &[ScalarType],
    args: &[u64],
) -> std::result::Result<(), SignatureError> {
    if params.len() != args.len() {
        return Err(SignatureError::Scalars {
            expected: params.len(),
            actual: args.len(),
        });
    }
    match params
        .iter()
        .zip(args)
        .position(|(&ty, &word)| Constant::new(ty, word).bits() != word)
    {
        Some(index) => Err(SignatureError::Word {
            index,
            ty: params[index],
        }),
        None => Ok(()),
    }
}
