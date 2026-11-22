//! Typed adapters for individually annotated Rust trait implementations.
//! The lookup key is the Rust function item type before pointer coercion.
use super::{KernelBuilder, scalar::Word};
use crate::hir::Reg;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Copy)]
pub struct ScalarMethod {
    pub evaluate: fn(&[Option<i32>]) -> crate::Result<i32>,
    pub stage: fn(&[Word], &mut dyn KernelBuilder) -> crate::Result<Reg>,
}

#[derive(Clone, Copy)]
pub enum Method {
    Scalar(ScalarMethod),
    // Promotion gives the pair of function pointers static storage without
    // imposing a static lifetime on the receiver they accept.
    Receiver(&'static (dyn Any + Send + Sync)),
}

impl Method {
    pub fn receiver_adapter<T: Any + Copy>(&self) -> Option<T> {
        match self {
            Self::Scalar(_) => None,
            Self::Receiver(adapter) => adapter.downcast_ref::<T>().copied(),
        }
    }
}

pub struct ReceiverArgument<T: ?Sized, Arg>(PhantomData<fn() -> *const T>, Option<Arg>);

impl<T: ?Sized, Arg> ReceiverArgument<T, Arg> {
    pub fn new(arg: Arg) -> Self {
        Self(PhantomData, Some(arg))
    }
}

pub trait InvokeReceiver<T: ?Sized> {
    fn evaluate_receiver(
        self,
        adapter: fn(T, &[Option<i32>]) -> crate::Result<i32>,
        words: &[Option<i32>],
    ) -> crate::Result<i32>;

    fn stage_receiver(
        self,
        adapter: fn(T, &[Word], &mut dyn KernelBuilder) -> crate::Result<Reg>,
        words: &[Word],
        builder: &mut dyn KernelBuilder,
    ) -> crate::Result<Reg>;
}

trait KnownReceiver<T> {
    fn require(self) -> crate::Result<T>;
}

impl<T> KnownReceiver<T> for Option<T> {
    fn require(self) -> crate::Result<T> {
        self.ok_or_else(|| "runtime value passed to a comptime receiver".into())
    }
}

impl<T> KnownReceiver<T> for super::scalar::Host<T> {
    fn require(self) -> crate::Result<T> {
        Ok(self.0)
    }
}

impl<T, Arg> ReceiverArgument<T, Arg> {
    fn take(&mut self) -> crate::Result<T>
    where
        Arg: KnownReceiver<T>,
    {
        self.1
            .take()
            .ok_or_else(|| String::from("the comptime trait receiver was consumed"))?
            .require()
    }
}

impl<T, Arg: KnownReceiver<T>> InvokeReceiver<T> for &mut ReceiverArgument<T, Arg> {
    fn evaluate_receiver(
        self,
        adapter: fn(T, &[Option<i32>]) -> crate::Result<i32>,
        words: &[Option<i32>],
    ) -> crate::Result<i32> {
        adapter(self.take()?, words)
    }

    fn stage_receiver(
        self,
        adapter: fn(T, &[Word], &mut dyn KernelBuilder) -> crate::Result<Reg>,
        words: &[Word],
        builder: &mut dyn KernelBuilder,
    ) -> crate::Result<Reg> {
        adapter(self.take()?, words, builder)
    }
}

impl<T: ?Sized, Arg> InvokeReceiver<T> for &&mut ReceiverArgument<T, Arg> {
    fn evaluate_receiver(
        self,
        _: fn(T, &[Option<i32>]) -> crate::Result<i32>,
        _: &[Option<i32>],
    ) -> crate::Result<i32> {
        Err("the comptime trait receiver cannot be materialized for specialization".into())
    }

    fn stage_receiver(
        self,
        _: fn(T, &[Word], &mut dyn KernelBuilder) -> crate::Result<Reg>,
        _: &[Word],
        _: &mut dyn KernelBuilder,
    ) -> crate::Result<Reg> {
        Err("the comptime trait receiver cannot be materialized for specialization".into())
    }
}

fn methods() -> &'static Mutex<HashMap<TypeId, Method>> {
    static METHODS: OnceLock<Mutex<HashMap<TypeId, Method>>> = OnceLock::new();
    METHODS.get_or_init(|| Mutex::new(HashMap::new()))
}

// The function item identifies an adapter. A qualified caller downcasts to its
// exact function pointer type before passing any receiver value or borrow.
pub fn register<F>(_: F, method: Method) {
    methods()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(typeid::of::<F>(), method);
}

pub fn lookup<F>(_: F) -> Option<Method> {
    methods()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&typeid::of::<F>())
        .copied()
}
