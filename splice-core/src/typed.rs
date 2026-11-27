//! The declaration marker ties a kernel's Rust argument type to its specialized IR.
use crate::error::build_boundary;
use crate::{BuildError, BuildOptions, CallError, Codegen, Executable, Function};
use std::{fmt, marker::PhantomData, ops::Deref};

/// Argument record emitted by an individual kernel declaration.
pub trait CallArgs {
    type Signature;
    type Output;
    fn invoke<K: Executable>(self, executable: &K) -> Result<Self::Output, CallError>;
}

/// A verified portable function retaining its declaration's argument types.
pub struct TypedIr<S> {
    function: Function,
    signature: PhantomData<fn(S) -> S>,
}
impl<S> fmt::Debug for TypedIr<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.function.fmt(f)
    }
}
impl<S> TypedIr<S> {
    /// Give `function` the declaration's argument types. `S` is a kernel's
    /// `Signature` marker: only the kernel's own module can construct one,
    /// so the IR of one kernel cannot be retyped as another's.
    #[doc(hidden)]
    pub fn new(function: Function, _: S) -> Self {
        Self {
            function,
            signature: PhantomData,
        }
    }
    pub fn into_inner(self) -> Function {
        self.function
    }
    pub fn as_inner(&self) -> &Function {
        &self.function
    }
}
impl<S> Deref for TypedIr<S> {
    type Target = Function;
    fn deref(&self) -> &Function {
        &self.function
    }
}

/// Backend output with a call interface specific to the kernel declaration.
pub struct TypedKernel<S, K> {
    output: K,
    signature: PhantomData<fn(S) -> S>,
}
impl<S, K: fmt::Debug> fmt::Debug for TypedKernel<S, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.output.fmt(f)
    }
}
impl<S, K> TypedKernel<S, K> {
    /// Give `output` the declaration's call interface; `S` seals it as
    /// [`TypedIr::new`] does.
    #[doc(hidden)]
    pub fn new(output: K, _: S) -> Self {
        Self::retyped(output)
    }
    /// The output of compiling a [`TypedIr<S>`] keeps its declaration.
    fn retyped(output: K) -> Self {
        Self {
            output,
            signature: PhantomData,
        }
    }
    pub fn as_inner(&self) -> &K {
        &self.output
    }
    pub fn into_inner(self) -> K {
        self.output
    }
}
impl<S, K: Executable> TypedKernel<S, K> {
    #[inline]
    pub fn call<A: CallArgs<Signature = S>>(&self, args: A) -> Result<A::Output, CallError> {
        args.invoke(&self.output)
    }
}
impl<S, K> Deref for TypedKernel<S, K> {
    type Target = K;
    fn deref(&self) -> &K {
        &self.output
    }
}

pub trait CompileInput<B: Codegen> {
    type Output;
    type Error;
    #[doc(hidden)]
    fn compile_with(&self, options: Option<BuildOptions>) -> Result<Self::Output, Self::Error>;
}
impl<B: Codegen> CompileInput<B> for Function {
    type Output = B::Output;
    type Error = BuildError;
    fn compile_with(&self, options: Option<BuildOptions>) -> crate::Result<B::Output> {
        let options = options.unwrap_or(self.options());
        crate::intrinsic::check::<B>(self, &options.target)?;
        B::compile_function(self, options)
    }
}
impl<S, B: Codegen> CompileInput<B> for TypedIr<S> {
    type Output = TypedKernel<S, B::Output>;
    type Error = BuildError;
    fn compile_with(&self, options: Option<BuildOptions>) -> Result<Self::Output, BuildError> {
        build_boundary(|| {
            <Function as CompileInput<B>>::compile_with(&self.function, options)
                .map(TypedKernel::retyped)
        })
    }
}
