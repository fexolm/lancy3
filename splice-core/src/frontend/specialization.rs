use std::{any::TypeId, marker::PhantomData};

/// An owned, prefix-free encoding of the parts of a host value that determine
/// its staged code. Deriving this trait includes the enum variant and all fields.
pub trait ComptimeKey {
    /// Identifies the Rust type with its lifetimes erased, for generic kernels.
    type Family: 'static;
    fn write_key(&self, bytes: &mut Vec<u8>);
}

/// A cache key lives within one kernel build. Equality checks the full argument
/// sequence even if two keys land in the same hash bucket.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Specialization {
    declaration: TypeId,
    signature: super::Signature<'static>,
    types: Vec<TypeId>,
    values: Vec<u8>,
}

impl Specialization {
    #[doc(hidden)]
    pub fn new<K: 'static>(signature: super::Signature<'static>) -> Self {
        Self {
            declaration: TypeId::of::<K>(),
            signature,
            types: Vec::new(),
            values: Vec::new(),
        }
    }
    pub fn signature(&self) -> super::Signature<'static> {
        self.signature
    }

    /// Also key on which runtime arguments are known constants, and their
    /// values: an inlined body folds on them.
    #[doc(hidden)]
    pub fn with_known(mut self, known: &[Option<crate::hir::Constant>]) -> Self {
        for constant in known {
            match constant {
                Some(constant) => {
                    self.values.push(1);
                    self.values.push(constant.ty() as u8);
                    self.values.extend(constant.bits().to_le_bytes());
                }
                None => self.values.push(0),
            }
        }
        self
    }
}

// Autoref fallback keeps opaque host values usable without assuming identity
// from their address or requiring all existing kernel inputs to implement a trait.
#[doc(hidden)]
pub struct KeyArgument<T: ?Sized>(PhantomData<fn(&T)>);
impl<T: ?Sized> KeyArgument<T> {
    pub fn for_value(_: &T) -> Self {
        Self::for_type()
    }
    pub fn for_type() -> Self {
        Self(PhantomData)
    }
}
#[doc(hidden)]
pub trait MaybeKey<T: ?Sized> {
    fn add_value(self, value: &T, key: &mut Option<Specialization>);
    fn add_type(self, key: &mut Option<Specialization>);
}
impl<T: ComptimeKey + ?Sized> MaybeKey<T> for &KeyArgument<T> {
    fn add_value(self, value: &T, key: &mut Option<Specialization>) {
        if let Some(key) = key {
            value.write_key(&mut key.values);
        }
    }
    fn add_type(self, key: &mut Option<Specialization>) {
        if let Some(key) = key {
            key.types.push(TypeId::of::<T::Family>());
        }
    }
}
impl<T: ?Sized> MaybeKey<T> for &&KeyArgument<T> {
    fn add_value(self, _: &T, key: &mut Option<Specialization>) {
        *key = None;
    }
    fn add_type(self, key: &mut Option<Specialization>) {
        *key = None;
    }
}

macro_rules! integers {
    ($($t:ty),*) => {$(
        impl ComptimeKey for $t {
            type Family = Self;
            fn write_key(&self, bytes: &mut Vec<u8>) { bytes.extend(self.to_le_bytes()); }
        }
    )*};
}
integers!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64
);
impl ComptimeKey for bool {
    type Family = Self;
    fn write_key(&self, bytes: &mut Vec<u8>) {
        bytes.push(u8::from(*self));
    }
}
impl ComptimeKey for char {
    type Family = Self;
    fn write_key(&self, bytes: &mut Vec<u8>) {
        (*self as u32).write_key(bytes);
    }
}
impl<T: ComptimeKey + ?Sized> ComptimeKey for &T {
    type Family = &'static T::Family;
    fn write_key(&self, bytes: &mut Vec<u8>) {
        (**self).write_key(bytes);
    }
}
impl<T: ComptimeKey + ?Sized> ComptimeKey for Box<T> {
    type Family = Box<T::Family>;
    fn write_key(&self, bytes: &mut Vec<u8>) {
        (**self).write_key(bytes);
    }
}
impl<T: ComptimeKey> ComptimeKey for [T] {
    type Family = Box<[T::Family]>;
    fn write_key(&self, bytes: &mut Vec<u8>) {
        self.len().write_key(bytes);
        for value in self {
            value.write_key(bytes);
        }
    }
}
impl<T: ComptimeKey> ComptimeKey for Vec<T> {
    type Family = Vec<T::Family>;
    fn write_key(&self, bytes: &mut Vec<u8>) {
        self.as_slice().write_key(bytes);
    }
}
impl<T: ComptimeKey, const N: usize> ComptimeKey for [T; N] {
    type Family = [T::Family; N];
    fn write_key(&self, bytes: &mut Vec<u8>) {
        self.as_slice().write_key(bytes);
    }
}
impl ComptimeKey for str {
    type Family = Box<str>;
    fn write_key(&self, bytes: &mut Vec<u8>) {
        self.len().write_key(bytes);
        bytes.extend(self.as_bytes());
    }
}
impl ComptimeKey for String {
    type Family = String;
    fn write_key(&self, bytes: &mut Vec<u8>) {
        self.as_str().write_key(bytes);
    }
}
impl<T: ComptimeKey> ComptimeKey for Option<T> {
    type Family = Option<T::Family>;
    fn write_key(&self, bytes: &mut Vec<u8>) {
        self.is_some().write_key(bytes);
        if let Some(value) = self {
            value.write_key(bytes);
        }
    }
}
macro_rules! tuple {
    ($($t:ident:$i:tt),+) => {
        impl<$($t: ComptimeKey),+> ComptimeKey for ($($t,)+) {
            type Family = ($($t::Family,)+);
            fn write_key(&self, bytes: &mut Vec<u8>) { $(self.$i.write_key(bytes);)+ }
        }
    };
}
impl ComptimeKey for () {
    type Family = ();
    fn write_key(&self, _: &mut Vec<u8>) {}
}
tuple!(A:0);
tuple!(A:0, B:1);
tuple!(A:0, B:1, C:2);
tuple!(A:0, B:1, C:2, D:3);
