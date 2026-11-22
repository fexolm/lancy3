//! The declaration marker ties a kernel's Rust argument type to its specialized IR.
use crate::{BuildOptions, Codegen, Executable, Function};
use std::{
    fmt,
    marker::PhantomData,
    ops::Deref,
    panic::{AssertUnwindSafe, catch_unwind},
};

macro_rules! phase_error {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, PartialEq)]
        pub struct $name(String);
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
        impl std::error::Error for $name {}
        impl From<String> for $name {
            fn from(message: String) -> Self {
                Self(message)
            }
        }
        impl From<$name> for String {
            fn from(error: $name) -> Self {
                error.0
            }
        }
    };
}
phase_error!(BuildError);
phase_error!(CallError);

fn boundary<T, E: From<String>>(action: impl FnOnce() -> crate::Result<T>) -> Result<T, E> {
    // Returning an error abandons this build or invocation. Mutations to the
    // caller's arguments made before an unwind retain their Rust semantics.
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(value) => value.map_err(E::from),
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_else(|| "kernel panicked".to_owned());
            // A panic payload can itself own a panicking destructor. Keep that
            // second unwind inside Rust as well, including at an extern entry.
            if let Err(secondary) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
                std::mem::forget(secondary);
            }
            Err(E::from(message))
        }
    }
}

#[doc(hidden)]
pub fn build_boundary<T>(action: impl FnOnce() -> crate::Result<T>) -> Result<T, BuildError> {
    boundary(action)
}
#[doc(hidden)]
pub fn call_boundary<T>(action: impl FnOnce() -> crate::Result<T>) -> Result<T, CallError> {
    boundary(action)
}

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
    #[doc(hidden)]
    pub fn new(function: Function) -> Self {
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
    #[doc(hidden)]
    pub fn new(output: K) -> Self {
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
impl<S, K: fmt::Display> fmt::Display for TypedKernel<S, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.output.fmt(f)
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
    type Error = String;
    fn compile_with(&self, options: Option<BuildOptions>) -> crate::Result<B::Output> {
        match options {
            Some(options) => B::compile_raw_with_options(self, options),
            None => B::compile_raw(self),
        }
    }
}
impl<S, B: Codegen> CompileInput<B> for TypedIr<S> {
    type Output = TypedKernel<S, B::Output>;
    type Error = BuildError;
    fn compile_with(&self, options: Option<BuildOptions>) -> Result<Self::Output, BuildError> {
        build_boundary(|| {
            <Function as CompileInput<B>>::compile_with(&self.function, options)
                .map(TypedKernel::new)
        })
    }
}
