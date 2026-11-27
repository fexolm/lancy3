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
macro_rules! scalar_constant {
    ($($ty:ty)*) => {$(
        impl ScalarConstant for $ty {
            fn constant(self) -> Constant {
                self.into()
            }
        }
    )*};
}
scalar_constant!(bool i32 u32 i64 u64 usize f32 f64);
impl<T: ScalarConstant + Copy> ScalarConstant for &T {
    fn constant(self) -> Constant {
        (*self).constant()
    }
}
#[doc(hidden)]
#[diagnostic::on_unimplemented(
    message = "a scalar kernel parameter requires a kernel scalar value"
)]
pub trait RunArg {
    fn known(&self) -> Option<Constant> {
        None
    }
    fn reg(self, builder: &mut dyn KernelBuilder) -> crate::Result<Reg>;
    /// The known value as an argument of a parameter of type `ty`: rustc
    /// gave an unsuffixed literal argument the parameter's type.
    fn known_as(&self, ty: ScalarType) -> Option<Constant> {
        self.known().and_then(|value| value.cast(ty))
    }
    /// The value as the register of a parameter of type `ty`.
    fn param(self, ty: ScalarType, b: &mut dyn KernelBuilder) -> crate::Result<Reg>
    where
        Self: Sized,
    {
        match self.known_as(ty) {
            Some(value) if self.known().is_some_and(|known| known.ty() != ty) => b.constant(value),
            _ => self.reg(b),
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
        // rustc gave both operands of an operator other than a shift one
        // type, so a known operand of another type is an unsuffixed literal.
        let shift = matches!(
            op,
            Binary::Shl | Binary::Shr | Binary::CheckedShl | Binary::CheckedShr
        );
        let retyped = |known: Self, runtime: Self, b: &dyn KernelBuilder| match (known, runtime) {
            (Self::Known(value), Self::Runtime(r)) if !shift && value.ty() != ScalarType::Bool => {
                match b.reg_type(r) {
                    Some(ValueType::Scalar(ty)) if ty != ScalarType::Bool => {
                        value.cast(ty).map_or(known, Self::Known)
                    }
                    _ => known,
                }
            }
            _ => known,
        };
        let (lhs, rhs) = (retyped(self, rhs, b), retyped(rhs, self, b));
        let (a, z) = (lhs.reg(b)?, rhs.reg(b)?);
        Ok(Self::Runtime(b.binary(op, a, z)?))
    }
}

/// A scalar operand of a Rust object operation, which reads it as `ty`.
pub fn context_operand(r: Reg, ty: ScalarType, b: &dyn KernelBuilder) -> crate::Result<Reg> {
    match b.reg_type(r) {
        Some(ValueType::Scalar(actual)) if actual == ty => Ok(r),
        _ => Err(crate::BuildError::staging(format!(
            "a Rust object operation reads this operand as {ty}"
        ))),
    }
}

/// The kernel type of the result of `operation`, which is never called:
/// the Rust type of a Rust object operation.
pub fn result_type<R: ImportWord>(_operation: impl FnOnce() -> R) -> ScalarType {
    R::TYPE
}

/// Assign an arm's value to the result of a branch whose type only the
/// builder knows: the first arm creates the result with its type.
pub fn assign_inferred(
    slot: &mut Option<Reg>,
    value: Reg,
    b: &mut dyn KernelBuilder,
) -> crate::Result<()> {
    let result = match *slot {
        Some(result) => result,
        None => {
            let ty = b
                .reg_type(value)
                .ok_or_else(|| crate::BuildError::staging("a branch value has no type"))?;
            *slot.insert(b.local(ty)?)
        }
    };
    b.assign(result, value)
}

/// The result of a branch of [`assign_inferred`] arms; without a reachable
/// arm, an unused `i32`.
pub fn inferred_result(slot: Option<Reg>, b: &mut dyn KernelBuilder) -> crate::Result<Reg> {
    match slot {
        Some(result) => Ok(result),
        None => b.local(ValueType::Scalar(ScalarType::I32)),
    }
}

/// Each field of a partially dynamic Rust struct is a runtime value, even
/// when a later optimization can replace its register by a constant.
pub fn aggregate<T, const N: usize>(
    mut fields: [Word; N],
    _constructor: fn() -> T,
    builder: &mut dyn KernelBuilder,
) -> crate::Result<[Word; N]> {
    if fields.iter().any(|field| matches!(field, Word::Runtime(_))) {
        for field in &mut fields {
            *field = Word::Runtime(field.reg(builder)?);
        }
    }
    Ok(fields)
}

/// An aggregate local of a kernel: the IR cannot run its destructor, so a
/// type with drop glue does not compile.
pub fn dropless_aggregate<T>(_: fn() -> T) {
    const {
        assert!(
            !std::mem::needs_drop::<T>(),
            "a kernel cannot build an aggregate with drop glue; build it in a Rust function and call that"
        );
    }
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

impl<T: crate::KernelScalar> RunArg for Host<T> {
    fn known(&self) -> Option<Constant> {
        Some(self.0.into())
    }
    fn reg(self, b: &mut dyn KernelBuilder) -> crate::Result<Reg> {
        b.constant(self.0.into())
    }
}

fn scalar_type(b: &dyn KernelBuilder, r: Reg, operation: &str) -> crate::Result<ScalarType> {
    match b.reg_type(r) {
        Some(ValueType::Scalar(ty)) => Ok(ty),
        _ => Err(crate::BuildError::staging(format!(
            "{operation} requires a runtime scalar"
        ))),
    }
}

/// An unsuffixed integer literal beside the register `like`: rustc gave it
/// `like`'s type, which only the builder knows.
pub fn int_like(b: &mut dyn KernelBuilder, like: Reg, value: u64) -> crate::Result<Reg> {
    let ty = scalar_type(b, like, "an integer literal")?;
    let constant = Constant::from(value)
        .cast(ty)
        .ok_or_else(|| crate::BuildError::staging("an integer literal beside a non-number"))?;
    b.constant(constant)
}

/// An unsuffixed float literal beside the register `like`.
pub fn float_like(b: &mut dyn KernelBuilder, like: Reg, value: f64) -> crate::Result<Reg> {
    let ty = scalar_type(b, like, "a float literal")?;
    if !ty.is_float() {
        return Err(crate::BuildError::staging(
            "a float literal beside an integer",
        ));
    }
    b.constant(
        Constant::from(value)
            .cast(ty)
            .expect("a float converts to a float"),
    )
}

/// Rust's binary operator `op` on two operands of one type: integer `+`,
/// `-`, `*`, `<<` and `>>` check overflow when `checks` is set.
pub fn arithmetic(
    b: &mut dyn KernelBuilder,
    op: Binary,
    checks: bool,
    lhs: Reg,
    rhs: Reg,
) -> crate::Result<Reg> {
    let checked = match op {
        Binary::Add => Binary::CheckedAdd,
        Binary::Sub => Binary::CheckedSub,
        Binary::Mul => Binary::CheckedMul,
        Binary::Shl => Binary::CheckedShl,
        Binary::Shr => Binary::CheckedShr,
        op => op,
    };
    let op = if checks && scalar_type(b, lhs, "arithmetic")?.is_integer() {
        checked
    } else {
        op
    };
    b.binary(op, lhs, rhs)
}

/// Rust's `-x`. A float is multiplied by `-1.0`: `0.0 - x` loses the sign of
/// a zero `x`.
pub fn negate(b: &mut dyn KernelBuilder, x: Reg, checks: bool) -> crate::Result<Reg> {
    let ty = scalar_type(b, x, "negation")?;
    if ty.is_float() {
        let minus_one = float_like(b, x, -1.0)?;
        return b.binary(Binary::Mul, x, minus_one);
    }
    let zero = b.constant(Constant::new(ty, 0))?;
    arithmetic(b, Binary::Sub, checks, zero, x)
}

/// Rust's `!x` on a `bool` or an integer.
pub fn not(b: &mut dyn KernelBuilder, x: Reg) -> crate::Result<Reg> {
    let ty = scalar_type(b, x, "`!`")?;
    let ones = if ty == ScalarType::Bool {
        Constant::from(true)
    } else {
        Constant::from(u64::MAX)
            .cast(ty)
            .filter(|_| ty.is_integer())
            .ok_or_else(|| crate::BuildError::staging("`!` on a float"))?
    };
    let ones = b.constant(ones)?;
    b.binary(Binary::Xor, x, ones)
}

/// Whether the crate of `module` (a `module_path!()`) compiles Rust
/// arithmetic with overflow checks.
/// `probe` is `|value| value + 1` compiled in that crate, so it panics on
/// `i32::MAX` exactly when the crate checks overflow. `unwind` and `profile`
/// are the crate's `cfg!(panic = "unwind")` and `cfg!(debug_assertions)`:
/// under `panic = "abort"` the probe cannot run and the profile decides
/// (`BuildOptions::with_overflow_checks` overrides it for kernel code).
/// The answer is computed once per crate name.
#[doc(hidden)]
pub fn overflow_checks(
    module: &'static str,
    probe: fn(i32) -> i32,
    unwind: bool,
    profile: bool,
) -> bool {
    use std::sync::Mutex;
    static CRATES: Mutex<Vec<(&'static str, bool)>> = Mutex::new(Vec::new());
    let krate = module.split("::").next().unwrap_or(module);
    let mut crates = CRATES.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some(&(_, checks)) = crates.iter().find(|(name, _)| *name == krate) {
        return checks;
    }
    let checks = if unwind { panics(probe) } else { profile };
    crates.push((krate, checks));
    checks
}

/// Run `probe(i32::MAX)` and report whether it panicked. While it runs, this
/// thread's panic is hidden from the panic hook; other threads still reach
/// the hook that was installed, which is restored afterwards.
fn panics(probe: fn(i32) -> i32) -> bool {
    use std::panic;
    use std::sync::Arc;
    thread_local! {
        static PROBING: Cell<bool> = const { Cell::new(false) };
    }
    let previous: Arc<dyn Fn(&panic::PanicHookInfo<'_>) + Send + Sync> =
        Arc::from(panic::take_hook());
    let delegate = Arc::clone(&previous);
    panic::set_hook(Box::new(move |info| {
        if !PROBING.get() {
            delegate(info);
        }
    }));
    PROBING.set(true);
    let panicked = panic::catch_unwind(|| probe(std::hint::black_box(i32::MAX))).is_err();
    PROBING.set(false);
    drop(panic::take_hook());
    panic::set_hook(Box::new(move |info| previous(info)));
    panicked
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

thread_local! {
    static INLINED: std::cell::RefCell<Vec<super::Specialization>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Inlines in progress of self-recursive kernels, keyed by specialization
/// and known runtime arguments.
#[doc(hidden)]
pub struct InlineGuard(bool);

/// Enter the inline expansion of a self-recursive kernel. Expanding the
/// same key inside itself would never end: report it as recursion at once.
/// A kernel whose comptime arguments have no key (`None`) is bounded by
/// [`enter_expansion`] only.
#[doc(hidden)]
pub fn enter_inline(key: Option<super::Specialization>) -> crate::Result<InlineGuard> {
    let Some(key) = key else {
        return Ok(InlineGuard(false));
    };
    INLINED.with_borrow_mut(|inlined| {
        if inlined.contains(&key) {
            return Err(crate::BuildError::staging(
                "a kernel calls itself with the same specialization arguments; \
                 recursion must change a #[comptime] argument and end at specialization",
            ));
        }
        inlined.push(key);
        Ok(InlineGuard(true))
    })
}

impl Drop for InlineGuard {
    fn drop(&mut self) {
        if self.0 {
            INLINED.with_borrow_mut(Vec::pop);
        }
    }
}

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
    element: crate::hir::ScalarType,
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
    (word (call $s:expr, $o:expr, $r:expr)) => { $s };
    (ordinary (call $s:expr, $o:expr, $r:expr)) => { $r };
    ($kind:ident (call $s:expr, $o:expr, $r:expr)) => { $o };
    (ordinary (evaluate $e:expr, $o:expr)) => { $o };
    (opaque (evaluate $e:expr, $o:expr)) => { $o };
    ($kind:ident (evaluate $e:expr, $o:expr)) => { $e };
    (word (items stage $($items:item)*)) => { $($items)* };
    (eval (items eval $($items:item)*)) => { $($items)* };
    ($kind:ident (items $provided:ident $($items:item)*)) => {};
}

/// A value that a Rust call passes or returns as one word of its kernel
/// type (see [`crate::KernelScalar`]); `()` is an `i32` zero.
pub trait ImportWord: Sized {
    const TYPE: ScalarType;
    fn decode(word: u64) -> Self;
    fn encode(self) -> u64;
    /// The value in a kernel register.
    fn constant(self) -> Constant {
        Constant::new(Self::TYPE, self.encode())
    }
    fn from_constant(value: Constant) -> Self {
        Self::decode(value.bits())
    }
}
impl<T: crate::KernelScalar> ImportWord for T {
    const TYPE: ScalarType = T::TYPE;
    fn decode(word: u64) -> Self {
        T::from_word(word)
    }
    fn encode(self) -> u64 {
        self.to_word()
    }
}
impl ImportWord for () {
    const TYPE: ScalarType = ScalarType::I32;
    fn decode(_: u64) -> Self {}
    fn encode(self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::overflow_checks;
    use std::cell::Cell;

    #[test]
    fn the_overflow_probe_hides_only_its_own_panic_and_keeps_the_hook() {
        thread_local! {
            static HOOKED: Cell<usize> = const { Cell::new(0) };
        }
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            HOOKED.set(HOOKED.get() + 1);
            previous(info);
        }));
        let checked: fn(i32) -> i32 = |value| value.checked_add(1).expect("overflow");
        let wrapping: fn(i32) -> i32 = |value| value.wrapping_add(1);
        assert!(overflow_checks("checked::kernel", checked, true, false));
        assert!(!overflow_checks("wrapping::kernel", wrapping, true, true));
        // Without unwinding the profile decides; each crate is probed once.
        assert!(overflow_checks("aborting::kernel", checked, false, true));
        assert!(overflow_checks("checked::other", wrapping, true, false));
        assert_eq!(HOOKED.get(), 0);
        let _ = std::panic::catch_unwind(|| panic!("after the probe"));
        assert_eq!(HOOKED.get(), 1);
    }
}
