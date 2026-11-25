//! Staging support used by the scalar kernel attribute during HIR construction.
use super::KernelBuilder;
use crate::hir::{Binary, Constant, Reg, ScalarType, ValueType};
use std::{cell::Cell, marker::PhantomData, rc::Rc};

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Host<T>(pub T);
/// A host value that a kernel uses at run time.
#[doc(hidden)]
pub trait ScalarConstant {
    fn constant(self) -> Constant;
}
impl ScalarConstant for i32 {
    fn constant(self) -> Constant {
        self.into()
    }
}
impl ScalarConstant for bool {
    fn constant(self) -> Constant {
        self.into()
    }
}
impl<T: ScalarConstant + Copy> ScalarConstant for &T {
    fn constant(self) -> Constant {
        (*self).constant()
    }
}
#[doc(hidden)]
#[diagnostic::on_unimplemented(message = "a scalar kernel parameter requires an i32 value")]
pub trait RunArg {
    fn known(&self) -> Option<Constant> {
        None
    }
    fn reg(self, builder: &mut dyn KernelBuilder) -> crate::Result<Reg>;
    /// The value as a register of type `ty`. A Rust call returns an `i32`
    /// word: a `bool` result is a nonzero word.
    fn typed(self, ty: ScalarType, b: &mut dyn KernelBuilder) -> crate::Result<Reg>
    where
        Self: Sized,
    {
        let r = self.reg(b)?;
        match b.reg_type(r) {
            Some(ValueType::Scalar(ScalarType::I32)) if ty == ScalarType::Bool => {
                let zero = b.constant(Constant::from(0))?;
                b.binary(Binary::Ne, r, zero)
            }
            _ => Ok(r),
        }
    }
    /// The value as the `i32` word that passes it to a Rust call.
    fn word(self, b: &mut dyn KernelBuilder) -> crate::Result<Reg>
    where
        Self: Sized,
    {
        let r = self.reg(b)?;
        match b.reg_type(r) {
            Some(ValueType::Scalar(ScalarType::Bool)) => b.cast(r, ScalarType::I32),
            _ => Ok(r),
        }
    }
}
impl RunArg for Reg {
    fn reg(self, _: &mut dyn KernelBuilder) -> crate::Result<Reg> {
        Ok(self)
    }
}
/// A word whose stage becomes known when the selected declaration is resolved.
#[derive(Clone, Copy)]
pub enum Word {
    Known(Constant),
    Runtime(Reg),
}
impl Word {
    pub fn input(reg: Reg, inputs: &[(Reg, Option<Constant>)]) -> Self {
        inputs
            .iter()
            .find(|(input, _)| *input == reg)
            .and_then(|(_, known)| *known)
            .map_or(Self::Runtime(reg), Self::Known)
    }
    /// The result of a call evaluated at build time.
    pub fn known<T: ImportWord>(value: T) -> Self {
        Self::Known(value.constant())
    }
    pub fn binary(self, op: Binary, rhs: Self, b: &mut dyn KernelBuilder) -> crate::Result<Self> {
        if let (Self::Known(a), Self::Known(z)) = (self, rhs)
            && a.ty() == z.ty()
            && op.output(a.ty()).is_some()
            && let Ok(value) = op.checked_eval(a, z)
        {
            return Ok(Self::Known(value));
        }
        let (mut a, mut z) = (self.reg(b)?, rhs.reg(b)?);
        // A Rust call returns an i32 word: it meets a bool operand as a bool.
        let bool = Some(ValueType::Scalar(ScalarType::Bool));
        if b.reg_type(a) == bool {
            z = z.typed(ScalarType::Bool, b)?;
        } else if b.reg_type(z) == bool {
            a = a.typed(ScalarType::Bool, b)?;
        }
        Ok(Self::Runtime(b.binary(op, a, z)?))
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
        return Err(crate::BuildError::staging(
            "a Rust aggregate with drop glue requires a Rust call",
        ));
    }
    if fields.iter().any(|field| matches!(field, Word::Runtime(_))) {
        for field in &mut fields {
            *field = Word::Runtime(field.reg(builder)?);
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
    word: Option<Constant>,
) -> crate::Result<F> {
    word.map(F::from_constant)
        .ok_or_else(|| crate::BuildError::staging("runtime aggregate in comptime!"))
}
impl RunArg for Word {
    fn known(&self) -> Option<Constant> {
        match *self {
            Self::Known(x) => Some(x),
            Self::Runtime(_) => None,
        }
    }
    fn reg(self, b: &mut dyn KernelBuilder) -> crate::Result<Reg> {
        match self {
            Self::Known(x) => b.constant(x),
            Self::Runtime(r) => Ok(r),
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
        Err(crate::BuildError::staging(
            "runtime value passed to a comptime parameter",
        ))
    }
}
impl<T> KnownArg<T> for String {
    fn require(self) -> crate::Result<T> {
        Err(crate::BuildError::staging(
            "runtime value passed to a comptime parameter",
        ))
    }
}
impl<T: ImportWord> KnownArg<T> for Word {
    fn require(self) -> crate::Result<T> {
        self.known().map(T::from_constant).ok_or_else(|| {
            crate::BuildError::staging("runtime value passed to a comptime parameter")
        })
    }
}

impl RunArg for Host<bool> {
    fn known(&self) -> Option<Constant> {
        Some(self.0.into())
    }
    fn reg(self, b: &mut dyn KernelBuilder) -> crate::Result<Reg> {
        b.constant(self.0.into())
    }
}
impl RunArg for Host<i32> {
    fn known(&self) -> Option<Constant> {
        Some(self.0.into())
    }
    fn reg(self, b: &mut dyn KernelBuilder) -> crate::Result<Reg> {
        b.constant(self.0.into())
    }
}

/// A host integer operation on which Rust panics (or would panic with
/// overflow checks). Kernels defer it: staged code traps when the operation
/// is reached, and `comptime!` returns its trap as a build error.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostFault {
    /// `Div` or `Rem` by zero.
    DivisionByZero(Binary),
    /// `Div` or `Rem` of the minimum by `-1`.
    DivisionOverflow(Binary),
    Overflow,
}

impl HostFault {
    /// An i32 kernel operation, with the faulting operator, that traps with
    /// the same error.
    pub fn kernel_operation(self) -> (Binary, Constant, Constant) {
        let (op, a, b) = match self {
            Self::DivisionByZero(op) => (op, 0, 0),
            Self::DivisionOverflow(op) => (op, i32::MIN, -1),
            Self::Overflow => (Binary::CheckedAdd, i32::MAX, 1),
        };
        (op, a.into(), b.into())
    }

    /// The trap the staged operation reports.
    pub fn trap(self) -> crate::Trap {
        match self {
            Self::DivisionByZero(_) => crate::Trap::DivisionByZero,
            Self::DivisionOverflow(_) => crate::Trap::DivisionOverflow,
            Self::Overflow => crate::Trap::Overflow,
        }
    }
}

/// A primitive integer. Generated host code borrows its operands, so
/// references forward to the referenced integer.
#[doc(hidden)]
pub trait HostInteger {
    type Int: Copy;
    fn int(&self) -> Self::Int;
    /// The Rust shift amount: Rust rejects a negative or too large one.
    fn shift_amount(&self) -> Option<u32>;
    fn arithmetic_fault(lhs: Self::Int, op: Binary, rhs: Self::Int) -> Option<HostFault>;
    fn shift_fault(lhs: Self::Int, op: Binary, amount: Option<u32>) -> Option<HostFault>;
    fn negation_fault(value: Self::Int) -> Option<HostFault>;
}

macro_rules! host_integer {
    ($($ty:ty),*) => {$(
        impl HostInteger for $ty {
            type Int = $ty;
            fn int(&self) -> $ty {
                *self
            }
            fn shift_amount(&self) -> Option<u32> {
                u32::try_from(*self).ok()
            }
            fn arithmetic_fault(lhs: $ty, op: Binary, rhs: $ty) -> Option<HostFault> {
                let ok = match op {
                    Binary::CheckedAdd => lhs.checked_add(rhs).is_some(),
                    Binary::CheckedSub => lhs.checked_sub(rhs).is_some(),
                    Binary::CheckedMul => lhs.checked_mul(rhs).is_some(),
                    Binary::Div | Binary::Rem if rhs == 0 => {
                        return Some(HostFault::DivisionByZero(op));
                    }
                    Binary::Div | Binary::Rem => {
                        return lhs.checked_div(rhs).is_none().then_some(HostFault::DivisionOverflow(op));
                    }
                    _ => true,
                };
                (!ok).then_some(HostFault::Overflow)
            }
            fn shift_fault(lhs: $ty, op: Binary, amount: Option<u32>) -> Option<HostFault> {
                let shifted = amount.and_then(|n| match op {
                    Binary::CheckedShl => lhs.checked_shl(n),
                    _ => lhs.checked_shr(n),
                });
                shifted.is_none().then_some(HostFault::Overflow)
            }
            fn negation_fault(value: $ty) -> Option<HostFault> {
                (0 as $ty).checked_sub(value).is_none().then_some(HostFault::Overflow)
            }
        }
    )*};
}
host_integer!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

impl<T: HostInteger> HostInteger for &T {
    type Int = T::Int;
    fn int(&self) -> T::Int {
        (**self).int()
    }
    fn shift_amount(&self) -> Option<u32> {
        (**self).shift_amount()
    }
    fn arithmetic_fault(lhs: T::Int, op: Binary, rhs: T::Int) -> Option<HostFault> {
        T::arithmetic_fault(lhs, op, rhs)
    }
    fn shift_fault(lhs: T::Int, op: Binary, amount: Option<u32>) -> Option<HostFault> {
        T::shift_fault(lhs, op, amount)
    }
    fn negation_fault(value: T::Int) -> Option<HostFault> {
        T::negation_fault(value)
    }
}
impl<T: HostInteger> HostInteger for &mut T {
    type Int = T::Int;
    fn int(&self) -> T::Int {
        (**self).int()
    }
    fn shift_amount(&self) -> Option<u32> {
        (**self).shift_amount()
    }
    fn arithmetic_fault(lhs: T::Int, op: Binary, rhs: T::Int) -> Option<HostFault> {
        T::arithmetic_fault(lhs, op, rhs)
    }
    fn shift_fault(lhs: T::Int, op: Binary, amount: Option<u32>) -> Option<HostFault> {
        T::shift_fault(lhs, op, amount)
    }
    fn negation_fault(value: T::Int) -> Option<HostFault> {
        T::negation_fault(value)
    }
}

// Generated code calls these on `&(lhs, rhs)` or `&(value,)`. Method lookup picks
// the integer implementation when one applies and otherwise autorefs once
// more to the catch-all, which leaves floats and user types to their own
// operators.

#[doc(hidden)]
pub trait HostArithmetic {
    fn arithmetic_fault(self, op: Binary) -> Option<HostFault>;
}
impl<L: HostInteger, R: HostInteger<Int = L::Int>> HostArithmetic for &(L, R) {
    fn arithmetic_fault(self, op: Binary) -> Option<HostFault> {
        L::arithmetic_fault(self.0.int(), op, self.1.int())
    }
}
impl<L, R> HostArithmetic for &&(L, R) {
    fn arithmetic_fault(self, _: Binary) -> Option<HostFault> {
        None
    }
}

#[doc(hidden)]
pub trait HostShift {
    fn shift_fault(self, op: Binary) -> Option<HostFault>;
}
impl<L: HostInteger, R: HostInteger> HostShift for &(L, R) {
    fn shift_fault(self, op: Binary) -> Option<HostFault> {
        L::shift_fault(self.0.int(), op, self.1.shift_amount())
    }
}
impl<L, R> HostShift for &&(L, R) {
    fn shift_fault(self, _: Binary) -> Option<HostFault> {
        None
    }
}

#[doc(hidden)]
pub trait HostNegation {
    fn negation_fault(self) -> Option<HostFault>;
}
impl<T: HostInteger> HostNegation for &(T,) {
    fn negation_fault(self) -> Option<HostFault> {
        T::negation_fault(self.0.int())
    }
}
impl<T> HostNegation for &&(T,) {
    fn negation_fault(self) -> Option<HostFault> {
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
pub fn enter_expansion() -> crate::Result<ExpansionGuard> {
    EXPANSION.with(|slot| {
        let mut state = slot.get();
        if state.depth >= 64 {
            return Err(crate::BuildError::staging(
                "kernel specialization exceeded 64 nested expansions",
            ));
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
            return Err(crate::BuildError::staging(
                "kernel specialization exceeded its iteration budget",
            ));
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
                .ok_or_else(|| {
                    crate::BuildError::staging(format!("column {slot} is not representable"))
                })?;
            Ok(crate::hir::Buffer(index))
        }
        Some(_) => Err(crate::BuildError::staging(format!(
            "column {slot} expects {} elements",
            element.name()
        ))),
        None => Err(crate::BuildError::staging(format!("unknown column {slot}"))),
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

/// A value that a Rust call passes or returns as one `i32` word.
pub trait ImportWord: Sized {
    fn decode(bits: i32) -> Self;
    fn encode(self) -> i32;
    /// The value in a kernel register.
    fn constant(self) -> Constant;
    fn from_constant(value: Constant) -> Self {
        Self::decode(value.bits() as u32 as i32)
    }
}
impl ImportWord for i32 {
    fn decode(bits: i32) -> Self {
        bits
    }
    fn encode(self) -> i32 {
        self
    }
    fn constant(self) -> Constant {
        self.into()
    }
}
impl ImportWord for bool {
    fn decode(bits: i32) -> Self {
        bits != 0
    }
    fn encode(self) -> i32 {
        i32::from(self)
    }
    fn constant(self) -> Constant {
        self.into()
    }
}
impl ImportWord for () {
    fn decode(_: i32) -> Self {}
    fn encode(self) -> i32 {
        0
    }
    fn constant(self) -> Constant {
        0i32.into()
    }
}
