//! Type-directed checks for operations supported by the build-time evaluator.
//! Autoref selects the rejection for other Rust types without invoking user code.
use std::marker::PhantomData;

/// In `Dropless::check()?.bind(expr)`, Rust checks the type before evaluating
/// the argument. Even constructing a named constant could schedule a user Drop.
pub struct Dropless<T>(PhantomData<fn() -> T>);

impl<T> Dropless<T> {
    pub fn check() -> crate::Result<Self> {
        if std::mem::needs_drop::<T>() {
            return Err(crate::BuildError::staging(
                "this value would run a destructor during specialization",
            ));
        }
        Ok(Self(PhantomData))
    }

    pub fn bind(self, value: T) -> T {
        value
    }
}

pub struct Operand<T: ?Sized>(PhantomData<fn(&T)>);
impl<T: ?Sized> Operand<T> {
    pub fn of(_: &T) -> Self {
        Self(PhantomData)
    }
}

// The type witness preserves Rust's result type without invoking a custom Index.
pub trait ReadIndex<T: ?Sized, I, V: ?Sized> {
    fn read_index(self, values: &T, index: I, infer: fn(&T, I) -> &V) -> crate::Result<Option<&V>>;
}

trait BuiltinSlice {
    type Item;
    fn elements(&self) -> &[Self::Item];
}
impl<T> BuiltinSlice for [T] {
    type Item = T;
    fn elements(&self) -> &[T] {
        self
    }
}
impl<T, const N: usize> BuiltinSlice for [T; N] {
    type Item = T;
    fn elements(&self) -> &[T] {
        self
    }
}
impl<T> BuiltinSlice for Vec<T> {
    type Item = T;
    fn elements(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T: BuiltinSlice + ?Sized> BuiltinSlice for &T {
    type Item = T::Item;
    fn elements(&self) -> &[Self::Item] {
        T::elements(self)
    }
}
impl<T: BuiltinSlice + ?Sized> BuiltinSlice for &mut T {
    type Item = T::Item;
    fn elements(&self) -> &[Self::Item] {
        T::elements(self)
    }
}

impl<T: BuiltinSlice + ?Sized, I, V: ?Sized> ReadIndex<T, I, V> for &Operand<T>
where
    I: std::slice::SliceIndex<[T::Item], Output = V>,
{
    fn read_index(self, values: &T, index: I, _: fn(&T, I) -> &V) -> crate::Result<Option<&V>> {
        Ok(values.elements().get(index))
    }
}
impl<T: ?Sized, I, V: ?Sized> ReadIndex<T, I, V> for &&Operand<T> {
    fn read_index(self, _: &T, _: I, _: fn(&T, I) -> &V) -> crate::Result<Option<&V>> {
        Err(crate::BuildError::staging(
            "opaque indexing in specialization",
        ))
    }
}
trait Primitive {}
macro_rules! primitives {
    ($($ty:ty),*) => {$(impl Primitive for $ty {})*}
}
primitives!(
    bool, char, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64
);
impl<T: Primitive> Primitive for &T {}

// A write during specialization may only replace primitive data owned by the
// specialization. In particular an exclusive borrow still belongs to the caller.
trait OwnedBuffer: BuiltinSlice {}
impl<T: Primitive> OwnedBuffer for Vec<T> {}
impl<T: Primitive, const N: usize> OwnedBuffer for [T; N] {}

pub trait CheckOwnedIndex<I> {
    fn check_owned_index(self, index: &I) -> crate::Result<()>;
}
impl<T: OwnedBuffer, I: std::slice::SliceIndex<[T::Item]>> CheckOwnedIndex<I> for &Operand<T> {
    fn check_owned_index(self, _: &I) -> crate::Result<()> {
        Ok(())
    }
}
impl<T: ?Sized, I> CheckOwnedIndex<I> for &&Operand<T> {
    fn check_owned_index(self, _: &I) -> crate::Result<()> {
        Err(crate::BuildError::staging(
            "specialization cannot write through a captured or opaque index",
        ))
    }
}

pub trait CheckReference {
    fn check_reference(self) -> crate::Result<()>;
}
impl<T: ?Sized> CheckReference for &Operand<&T> {
    fn check_reference(self) -> crate::Result<()> {
        Ok(())
    }
}
impl<T: ?Sized> CheckReference for &Operand<&mut T> {
    fn check_reference(self) -> crate::Result<()> {
        Ok(())
    }
}
impl<T: ?Sized> CheckReference for &&Operand<T> {
    fn check_reference(self) -> crate::Result<()> {
        Err(crate::BuildError::staging(
            "user-defined dereference during specialization",
        ))
    }
}

pub trait CheckPrimitive {
    fn check(self) -> crate::Result<()>;
}
impl<T: Primitive> CheckPrimitive for &Operand<T> {
    fn check(self) -> crate::Result<()> {
        Ok(())
    }
}
impl<T: ?Sized> CheckPrimitive for &&Operand<T> {
    fn check(self) -> crate::Result<()> {
        Err(crate::BuildError::staging(
            "user-defined operation in comptime!",
        ))
    }
}
pub trait ReadLength<T: ?Sized> {
    fn length(self, value: &T) -> crate::Result<usize>;
}
trait Length {
    fn count(&self) -> usize;
}
impl<T> Length for [T] {
    fn count(&self) -> usize {
        self.len()
    }
}
impl<T, const N: usize> Length for [T; N] {
    fn count(&self) -> usize {
        N
    }
}
impl Length for str {
    fn count(&self) -> usize {
        self.len()
    }
}
impl<T: Length + ?Sized> Length for &T {
    fn count(&self) -> usize {
        T::count(self)
    }
}
impl<T: Length + ?Sized> ReadLength<T> for &Operand<T> {
    fn length(self, value: &T) -> crate::Result<usize> {
        Ok(value.count())
    }
}
impl<T: ?Sized> ReadLength<T> for &&Operand<T> {
    fn length(self, _: &T) -> crate::Result<usize> {
        Err(crate::BuildError::staging("opaque length in comptime!"))
    }
}

pub trait OwnField {
    fn own_field(self, name: &str) -> bool;
}
impl<T: crate::KernelType> OwnField for &Operand<T> {
    fn own_field(self, name: &str) -> bool {
        T::FIELDS.iter().any(|f| f.name == name)
    }
}
impl<T: ?Sized> OwnField for &&Operand<T> {
    fn own_field(self, _: &str) -> bool {
        false
    }
}
pub trait ReferentField {
    fn referent_field(self, name: &str) -> bool;
}
impl<T: crate::KernelType> ReferentField for &Operand<&T> {
    fn referent_field(self, name: &str) -> bool {
        T::FIELDS.iter().any(|f| f.name == name)
    }
}
impl<T: crate::KernelType> ReferentField for &Operand<&mut T> {
    fn referent_field(self, name: &str) -> bool {
        T::FIELDS.iter().any(|f| f.name == name)
    }
}
impl<T: ?Sized> ReferentField for &&Operand<T> {
    fn referent_field(self, _: &str) -> bool {
        false
    }
}

// Only these concrete iterables can advance during specialization. Calling
// an arbitrary IntoIterator or Iterator implementation may run user code.
trait PlanSource {}
impl<T> PlanSource for [T] {}
impl<T, const N: usize> PlanSource for [T; N] {}
impl<T> PlanSource for Vec<T> {}
impl<T: PlanSource + ?Sized> PlanSource for &T {}
impl<T> PlanSource for std::slice::Iter<'_, T> {}
impl<I: PlanSource> PlanSource for std::iter::Enumerate<I> {}
impl<T: Primitive> PlanSource for std::ops::Range<T> {}
impl<T: Primitive> PlanSource for std::ops::RangeInclusive<T> {}

pub trait CheckPlanSource {
    fn check_source(self) -> crate::Result<()>;
}
impl<T: PlanSource + ?Sized> CheckPlanSource for &Operand<T> {
    fn check_source(self) -> crate::Result<()> {
        Ok(())
    }
}
impl<T: ?Sized> CheckPlanSource for &&Operand<T> {
    fn check_source(self) -> crate::Result<()> {
        Err(crate::BuildError::staging(
            "opaque iterator cannot run during specialization",
        ))
    }
}
