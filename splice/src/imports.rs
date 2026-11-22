//! A typed Rust import whose arguments, result and environment stay in Rust.
use crate::{
    BuildError, BuildOptions, CallError, Codegen, CompileInput, Executable, call_boundary,
};
use std::{marker::PhantomData, sync::Arc};

pub trait ImportArgs<E> {
    type Signature;
    type Output;
    fn invoke(self, environment: &E) -> Self::Output;
}

pub struct RustImport<S, E> {
    name: &'static str,
    environment: Arc<E>,
    signature: PhantomData<fn(S) -> S>,
}
impl<S, E> RustImport<S, E> {
    pub fn new(name: &'static str, environment: E) -> Self {
        Self {
            name,
            environment: Arc::new(environment),
            signature: PhantomData,
        }
    }
    pub fn name(&self) -> &'static str {
        self.name
    }
}

pub struct LinkedImport<S, E> {
    environment: Arc<E>,
    signature: PhantomData<fn(S) -> S>,
}
impl<S, E> LinkedImport<S, E> {
    pub fn call<A: ImportArgs<E, Signature = S>>(&self, args: A) -> Result<A::Output, CallError> {
        self.with_environment(|environment| args.invoke(environment))
    }

    #[doc(hidden)]
    pub fn with_environment<R>(&self, action: impl FnOnce(&E) -> R) -> Result<R, CallError> {
        call_boundary(|| Ok(action(&self.environment)))
    }
}

pub trait ImportTarget {
    type Linked<S, E>;
    fn link<S, E>(import: &RustImport<S, E>) -> Result<Self::Linked<S, E>, BuildError>;
}
impl<K: Executable> ImportTarget for K {
    type Linked<S, E> = LinkedImport<S, E>;
    fn link<S, E>(import: &RustImport<S, E>) -> Result<Self::Linked<S, E>, BuildError> {
        Ok(LinkedImport {
            environment: Arc::clone(&import.environment),
            signature: PhantomData,
        })
    }
}
impl ImportTarget for String {
    type Linked<S, E> = String;
    fn link<S, E>(import: &RustImport<S, E>) -> Result<String, BuildError> {
        Ok(format!("rust import {}\n", import.name))
    }
}
impl<S, E, B: Codegen> CompileInput<B> for RustImport<S, E>
where
    B::Output: ImportTarget,
{
    type Output = <B::Output as ImportTarget>::Linked<S, E>;
    type Error = BuildError;
    fn compile_with(&self, _options: Option<BuildOptions>) -> Result<Self::Output, BuildError> {
        B::Output::link(self)
    }
}
