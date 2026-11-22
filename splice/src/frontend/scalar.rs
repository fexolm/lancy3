//! Staging support used by the scalar kernel attribute during HIR construction.
use super::KernelBuilder;
use crate::hir::{Binary, Reg};
use std::{cell::Cell, marker::PhantomData, rc::Rc};

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Host<T>(pub T);
#[doc(hidden)]
pub trait ScalarConstant {
    fn bits(self) -> i32;
}
impl ScalarConstant for i32 {
    fn bits(self) -> i32 {
        self
    }
}
impl ScalarConstant for bool {
    fn bits(self) -> i32 {
        -i32::from(self)
    }
}
impl<T: ScalarConstant + Copy> ScalarConstant for &T {
    fn bits(self) -> i32 {
        (*self).bits()
    }
}
#[doc(hidden)]
#[diagnostic::on_unimplemented(message = "a scalar kernel parameter requires an i32 value")]
pub trait RunArg {
    fn known(&self) -> Option<i32> {
        None
    }
    fn reg(self, builder: &mut dyn KernelBuilder) -> Reg;
}
impl RunArg for Reg {
    fn reg(self, _: &mut dyn KernelBuilder) -> Reg {
        self
    }
}
/// A word whose stage becomes known when the selected declaration is resolved.
#[derive(Clone, Copy)]
pub enum Word {
    Known(i32),
    Runtime(Reg),
}
impl Word {
    pub fn input(reg: Reg, inputs: &[(Reg, Option<i32>)]) -> Self {
        inputs
            .iter()
            .find(|(input, _)| *input == reg)
            .and_then(|(_, known)| *known)
            .map_or(Self::Runtime(reg), Self::Known)
    }
    pub fn binary(self, op: Binary, rhs: Self, b: &mut dyn KernelBuilder) -> Self {
        if let (Self::Known(a), Self::Known(z)) = (self, rhs)
            && let Some(value) = op.checked_eval(a, z)
        {
            return Self::Known(value);
        }
        let (a, z) = (self.reg(b), rhs.reg(b));
        Self::Runtime(b.binary(op, a, z))
    }
}

/// Each field of a partially dynamic Rust struct is a runtime value, even
/// when a later optimization can replace its register by a constant.
pub fn aggregate<T, const N: usize>(
    mut fields: [Word; N],
    _constructor: fn() -> T,
    builder: &mut dyn KernelBuilder,
) -> crate::Result<[Word; N]> {
    if std::mem::needs_drop::<T>() {
        return Err("a Rust aggregate with drop glue requires a Rust call".into());
    }
    if fields.iter().any(|field| matches!(field, Word::Runtime(_))) {
        for field in &mut fields {
            *field = Word::Runtime(field.reg(builder));
        }
    }
    Ok(fields)
}

pub fn has_destructor<T>(_: fn() -> T) -> bool {
    std::mem::needs_drop::<T>()
}

pub fn known_field<T, F: ImportWord>(
    _constructor: fn() -> T,
    _project: fn(&T) -> &F,
    word: Option<i32>,
) -> crate::Result<F> {
    word.map(F::decode)
        .ok_or_else(|| "runtime aggregate in comptime!".into())
}
impl RunArg for Word {
    fn known(&self) -> Option<i32> {
        match *self {
            Self::Known(x) => Some(x),
            Self::Runtime(_) => None,
        }
    }
    fn reg(self, b: &mut dyn KernelBuilder) -> Reg {
        match self {
            Self::Known(x) => b.constant(x),
            Self::Runtime(r) => r,
        }
    }
}

pub trait KnownArg<T> {
    fn require(self) -> crate::Result<T>;
}
impl<T> KnownArg<T> for Host<T> {
    fn require(self) -> crate::Result<T> {
        Ok(self.0)
    }
}
impl<T> KnownArg<T> for Reg {
    fn require(self) -> crate::Result<T> {
        Err("runtime value passed to a comptime parameter".into())
    }
}
impl<T> KnownArg<T> for String {
    fn require(self) -> crate::Result<T> {
        Err("runtime value passed to a comptime parameter".into())
    }
}
impl<T: ImportWord> KnownArg<T> for Word {
    fn require(self) -> crate::Result<T> {
        self.known()
            .map(T::decode)
            .ok_or_else(|| "runtime value passed to a comptime parameter".into())
    }
}

impl RunArg for Host<bool> {
    fn known(&self) -> Option<i32> {
        Some(-i32::from(self.0))
    }
    fn reg(self, b: &mut dyn KernelBuilder) -> Reg {
        b.constant(-i32::from(self.0))
    }
}
impl RunArg for Host<i32> {
    fn known(&self) -> Option<i32> {
        Some(self.0)
    }
    fn reg(self, b: &mut dyn KernelBuilder) -> Reg {
        b.constant(self.0)
    }
}

// Method lookup on &operands selects the i32 implementation (including
// borrowed operands). Other host types use their own Rust operator.
trait HostI32 {
    fn integer(&self) -> i32;
}
impl HostI32 for i32 {
    fn integer(&self) -> i32 {
        *self
    }
}
impl<T: HostI32> HostI32 for &T {
    fn integer(&self) -> i32 {
        (**self).integer()
    }
}
impl<T: HostI32> HostI32 for &mut T {
    fn integer(&self) -> i32 {
        (**self).integer()
    }
}

#[doc(hidden)]
pub trait ArithmeticGuard {
    fn faulting_i32_arithmetic(self, op: Binary) -> Option<(i32, i32)>;
}
impl<L: HostI32, R: HostI32> ArithmeticGuard for &(L, R) {
    fn faulting_i32_arithmetic(self, op: Binary) -> Option<(i32, i32)> {
        let (lhs, rhs) = (self.0.integer(), self.1.integer());
        op.checked_eval(lhs, rhs).is_none().then_some((lhs, rhs))
    }
}
impl<L, R> ArithmeticGuard for &&(L, R) {
    fn faulting_i32_arithmetic(self, _: Binary) -> Option<(i32, i32)> {
        None
    }
}

#[derive(Clone, Copy)]
struct Expansion {
    depth: u32,
    remaining: u32,
}

const SPECIALIZATION_STEPS: u32 = 65_536;

thread_local! {
    static EXPANSION: Cell<Expansion> = const { Cell::new(Expansion { depth: 0, remaining: 0 }) };
}

/// Bounds non-terminating specialization, including mutually recursive factories.
#[doc(hidden)]
pub struct ExpansionGuard(PhantomData<Rc<()>>);

#[doc(hidden)]
pub fn enter_expansion() -> Result<ExpansionGuard, String> {
    EXPANSION.with(|slot| {
        let mut state = slot.get();
        if state.depth >= 64 {
            return Err("kernel specialization exceeded 64 nested expansions".into());
        }
        if state.depth == 0 {
            state.remaining = SPECIALIZATION_STEPS;
        }
        state.depth += 1;
        slot.set(state);
        Ok(ExpansionGuard(PhantomData))
    })
}

#[doc(hidden)]
#[inline]
pub fn specialization_step() -> crate::Result<()> {
    EXPANSION.with(|slot| {
        let mut state = slot.get();
        let Some(remaining) = state.remaining.checked_sub(1) else {
            return Err("kernel specialization exceeded its iteration budget".into());
        };
        state.remaining = remaining;
        slot.set(state);
        Ok(())
    })
}

impl Drop for ExpansionGuard {
    fn drop(&mut self) {
        EXPANSION.with(|slot| {
            let mut state = slot.get();
            state.depth -= 1;
            slot.set(state);
        });
    }
}
/// Resolve a read-only column against the layout supplied at specialization.
#[doc(hidden)]
pub fn column(
    (kinds, base): (&[crate::BufferKind], crate::hir::Buffer),
    slot: usize,
    element: crate::simd::Element,
) -> crate::Result<crate::hir::Buffer> {
    match kinds.get(slot) {
        Some(kind) if kind.context_signature().is_none() && kind.element() == element => {
            let index = base
                .index()
                .checked_add(slot)
                .and_then(|i| u32::try_from(i).ok())
                .ok_or_else(|| format!("column {slot} is not representable"))?;
            Ok(crate::hir::Buffer(index))
        }
        Some(_) => Err(format!("column {slot} expects {} elements", element.name())),
        None => Err(format!("unknown column {slot}")),
    }
}

// Expand only the selected expression. The other branch may use the calling
// convention of a function with a different Rust signature.
#[macro_export]
#[doc(hidden)]
macro_rules! __splice_choose_kernel {
    ($kernel:expr, $rust:expr) => {
        $kernel
    };
}
#[macro_export]
#[doc(hidden)]
macro_rules! __splice_choose_rust {
    (@implementation $body:expr) => {
        ::core::compile_error!("a #[kernel] trait impl requires #[kernel] on the trait declaration")
    };
    (@expr $name:ident, $pick:ident, $args:tt) => {
        $pick! { ordinary $args }
    };
    ($kernel:expr, $rust:expr) => {
        $rust
    };
}

// The trait chooses the convention. Discard other branches before name
// resolution: they can reference entries that this declaration does not have.
#[doc(hidden)]
#[macro_export]
macro_rules! __splice_method_case {
    (word (call $s:expr, $b:expr, $o:expr, $r:expr)) => { $s };
    (boolean (call $s:expr, $b:expr, $o:expr, $r:expr)) => { $b };
    (ordinary (call $s:expr, $b:expr, $o:expr, $r:expr)) => { $r };
    ($kind:ident (call $s:expr, $b:expr, $o:expr, $r:expr)) => { $o };
    (ordinary (evaluate $e:expr, $o:expr)) => { $o };
    (opaque (evaluate $e:expr, $o:expr)) => { $o };
    ($kind:ident (evaluate $e:expr, $o:expr)) => { $e };
    (word (items stage $($items:item)*)) => { $($items)* };
    (boolean (items stage $($items:item)*)) => { $($items)* };
    (eval (items eval $($items:item)*)) => { $($items)* };
    ($kind:ident (items $provided:ident $($items:item)*)) => {};
}

pub trait ImportWord: Sized {
    fn decode(bits: i32) -> Self;
    fn encode(self) -> i32;
}
impl ImportWord for i32 {
    fn decode(bits: i32) -> Self {
        bits
    }
    fn encode(self) -> i32 {
        self
    }
}
impl ImportWord for bool {
    fn decode(bits: i32) -> Self {
        bits != 0
    }
    fn encode(self) -> i32 {
        -i32::from(self)
    }
}
impl ImportWord for () {
    fn decode(_: i32) -> Self {}
    fn encode(self) -> i32 {
        0
    }
}

/// The generated adapter calls this inside its C entry point. Unwinding
/// finishes inside Rust before control returns to the native kernel.
pub fn invoke_import<R: ImportWord>(call: impl FnOnce() -> R) -> i64 {
    match crate::call_boundary(|| Ok(call().encode())) {
        Ok(word) => i64::from(word as u32),
        Err(_) => 5_i64 << 32,
    }
}
