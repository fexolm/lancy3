use crate::KernelScalar;
use crate::hir::{Reg, ScalarType, Slice, ValueType};
use crate::runtime_context::{ContextSignature, RuntimeContext};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferKind {
    element: ScalarType,
    writable: bool,
    /// The bytes are UTF-8: a `&str`, a read-only slice of `u8`.
    utf8: bool,
    context: Option<&'static ContextSignature>,
}

impl BufferKind {
    /// A shared `i32` slice.
    pub const READ_I32: Self = Self::new(ScalarType::I32, false);
    /// A writable `i32` slice.
    pub const WRITE_I32: Self = Self::new(ScalarType::I32, true);
    /// A `&str`: a shared slice of `u8` that holds UTF-8.
    pub const STR: Self = Self {
        utf8: true,
        ..Self::new(ScalarType::U8, false)
    };
    /// # Panics
    /// If `element` is `Bool`: a kernel cannot borrow a `bool` slice.
    pub const fn new(element: ScalarType, writable: bool) -> Self {
        assert!(
            !matches!(element, ScalarType::Bool),
            "a bool slice is not a kernel buffer"
        );
        Self {
            element,
            writable,
            utf8: false,
            context: None,
        }
    }
    #[doc(hidden)]
    pub const fn context(signature: &'static ContextSignature) -> Self {
        Self {
            element: ScalarType::I32,
            writable: true,
            utf8: false,
            context: Some(signature),
        }
    }
    #[doc(hidden)]
    pub const fn context_signature(self) -> Option<&'static ContextSignature> {
        self.context
    }
    pub const fn element(self) -> ScalarType {
        self.element
    }
    /// Whether the slice is a `&str`, whose bytes are UTF-8.
    pub const fn utf8(self) -> bool {
        self.utf8
    }
    /// This kind without the UTF-8 guarantee: `str::as_bytes`.
    pub const fn bytes(self) -> Self {
        Self {
            utf8: false,
            ..self
        }
    }
    pub const fn writable(self) -> bool {
        self.writable
    }
    /// Whether two distinct parameters with these kinds may overlap in a
    /// valid kernel call. A writable slice has exclusive access to its bytes.
    pub const fn may_alias(self, other: Self) -> bool {
        !self.writable() && !other.writable()
    }
    /// A `&str` is also its bytes, but bytes need not be a `&str`.
    fn accepts(self, actual: Self) -> bool {
        self.same_type(actual) && (!self.writable || actual.writable) && (!self.utf8 || actual.utf8)
    }
    fn same_type(self, actual: Self) -> bool {
        self.element == actual.element && self.context == actual.context
    }
}

/// A borrowed typed slice, or the Rust context of a kernel call. Writable
/// slices also support loads.
pub struct BufferArg<'a> {
    data: NonNull<u8>,
    len: usize,
    kind: BufferKind,
    /// The borrow the argument was made from: shared for read-only slices,
    /// exclusive for writable slices and contexts.
    borrow: PhantomData<&'a mut [u8]>,
}

impl fmt::Debug for BufferArg<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferArg")
            .field("kind", &self.kind)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

mod sealed {
    pub trait Sealed {}
}

/// An element type that can be borrowed from a [`BufferArg`]: every kernel
/// scalar but `bool`, whose bytes Rust requires to be 0 or 1.
#[diagnostic::on_unimplemented(
    message = "`[{Self}]` is not a kernel slice",
    label = "not a slice element",
    note = "slice elements are `char`, `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `usize`, `f32` and `f64`"
)]
pub trait BufferElement: KernelScalar + sealed::Sealed {}
macro_rules! buffer_element {
    ($($ty:ty),*) => {$(
        impl sealed::Sealed for $ty {}
        impl BufferElement for $ty {}
    )*};
}
buffer_element!(char, i8, u8, i16, u16, i32, u32, i64, u64, usize, f32, f64);

impl<'a, T: BufferElement> From<&'a [T]> for BufferArg<'a> {
    fn from(slice: &'a [T]) -> Self {
        Self::read(slice)
    }
}
impl<'a, T: BufferElement> From<&'a mut [T]> for BufferArg<'a> {
    fn from(slice: &'a mut [T]) -> Self {
        Self::write(slice)
    }
}

impl<'a> BufferArg<'a> {
    /// A read-only slice. Native code never writes through it: the verifier
    /// rejects stores to read-only buffers.
    pub fn read<T: BufferElement>(slice: &'a [T]) -> Self {
        Self {
            data: NonNull::from(slice).cast(),
            len: slice.len(),
            kind: BufferKind::new(T::TYPE, false),
            borrow: PhantomData,
        }
    }

    /// A `&str`, as its UTF-8 bytes. Native code never writes through it.
    pub fn str(text: &'a str) -> Self {
        Self {
            kind: BufferKind::STR,
            ..Self::read(text.as_bytes())
        }
    }

    pub fn write<T: BufferElement>(slice: &'a mut [T]) -> Self {
        let len = slice.len();
        Self {
            data: NonNull::from(slice).cast(),
            len,
            kind: BufferKind::new(T::TYPE, true),
            borrow: PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn context(context: &'a mut RuntimeContext<'a>) -> Self {
        let kind = BufferKind::context(context.signature());
        Self {
            data: NonNull::from(context).cast(),
            len: 0,
            kind,
            borrow: PhantomData,
        }
    }
}

impl BufferArg<'_> {
    /// Reborrow a slice through a shared reference. A Rust context has no
    /// read-only form.
    pub fn as_read_only(&self) -> Option<BufferArg<'_>> {
        self.kind.context.is_none().then_some(BufferArg {
            kind: BufferKind {
                writable: false,
                ..self.kind
            },
            ..*self
        })
    }

    pub fn kind(&self) -> BufferKind {
        self.kind
    }
    /// The number of elements. A Rust context has none.
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Raw address for validated native backends. Only writable arguments
    /// and contexts permit writes.
    pub fn data(&mut self) -> *mut u8 {
        self.data.as_ptr()
    }
    /// Borrow the slice as `T`.
    ///
    /// # Panics
    /// If this argument holds a different element type or a Rust context.
    #[allow(unsafe_code)]
    pub fn as_slice<T: BufferElement>(&self) -> &[T] {
        assert!(
            self.kind.context.is_none() && self.kind.element == T::TYPE,
            "expected a {} buffer",
            T::TYPE.name()
        );
        // SAFETY: `read` and `write` made `data` and `len` from a live
        // `&[T]` or `&mut [T]` that this argument borrows, and the element
        // type matches. `&self` permits only shared access to it.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast(), self.len) }
    }
}

/// Check arity, mutability and slice lengths before entering native code:
/// a slice longer than `longest` elements, the most the backend handles, is
/// rejected.
#[inline]
pub fn validate_buffers(
    kinds: &[BufferKind],
    args: &[BufferArg<'_>],
    longest: usize,
) -> Result<(), crate::SignatureError> {
    use crate::SignatureError::*;
    if kinds.len() != args.len() {
        return Err(Buffers {
            expected: kinds.len(),
            actual: args.len(),
        });
    }
    for (buffer, (&kind, arg)) in kinds.iter().zip(args).enumerate() {
        if !kind.same_type(arg.kind()) {
            if kind.context_signature().is_some() {
                return Err(Context { buffer });
            }
            return Err(Element {
                buffer,
                expected: kind.element(),
            });
        }
        if kind.utf8() && !arg.kind().utf8() {
            return Err(Utf8 { buffer });
        }
        if !kind.accepts(arg.kind()) {
            return Err(ReadOnly { buffer });
        }
        if arg.len() > longest {
            return Err(Length {
                buffer,
                limit: longest,
            });
        }
    }
    Ok(())
}

/// Check the borrow contract at a static call site, where the caller's
/// registers have `types`. Multiple immutable views may name the same
/// slice. A mutable view needs exclusive access: a slice value passed
/// twice is rejected, and distinct values are distinct borrows, which the
/// Rust source that produced them guarantees.
pub fn check_forwarding(
    types: Types<'_>,
    callee: &[BufferKind],
    actual: &[Slice],
) -> crate::Result<()> {
    if callee.len() != actual.len() {
        return Err(crate::BuildError::verify(format!(
            "expected {} buffer arguments, got {}",
            callee.len(),
            actual.len()
        )));
    }
    let mut seen: HashMap<_, BufferKind> = HashMap::new();
    for (i, (&required, &slice)) in callee.iter().zip(actual).enumerate() {
        let kind = slice.kind(types).ok_or_else(|| {
            crate::BuildError::verify(format!("buffer argument {i} is not a slice"))
        })?;
        if !required.same_type(kind) {
            return Err(crate::BuildError::verify(format!(
                "buffer argument {i} has wrong element type"
            )));
        }
        if required.utf8() && !kind.utf8() {
            return Err(crate::BuildError::verify(format!(
                "buffer argument {i} is not a &str"
            )));
        }
        if !required.accepts(kind) {
            return Err(crate::BuildError::verify(format!(
                "buffer argument {i} is read-only"
            )));
        }
        if seen
            .insert(slice.ptr, required)
            .is_some_and(|prior| !prior.may_alias(required))
        {
            return Err(crate::BuildError::verify(format!(
                "writable buffer argument {i} aliases another call argument"
            )));
        }
    }
    Ok(())
}

/// What the registers of a function under construction hold.
pub type Types<'a> = &'a dyn Fn(Reg) -> Option<ValueType>;

impl Slice {
    /// The kind of this slice when its registers have `types`: `None`
    /// unless the pointer is a [`ValueType::Ptr`] and the length an `Index`.
    pub fn kind(self, types: Types<'_>) -> Option<BufferKind> {
        match (types(self.ptr), types(self.len)) {
            (Some(ValueType::Ptr(kind)), Some(ValueType::Scalar(ScalarType::Index))) => Some(kind),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_argument_keeps_its_borrow_kind_and_length() {
        let mut data = [1.5f32, 2.5, 3.5];
        let mut arg = BufferArg::write(&mut data);
        assert_eq!(arg.kind(), BufferKind::new(ScalarType::F32, true));
        assert_eq!(arg.len(), 3);
        let address = arg.data();
        let shared = arg.as_read_only().expect("a slice has a read-only form");
        assert_eq!(shared.kind(), BufferKind::new(ScalarType::F32, false));
        assert_eq!(shared.as_slice::<f32>(), &[1.5, 2.5, 3.5]);
        assert_eq!(address, data.as_mut_ptr().cast());
    }

    #[test]
    fn a_str_parameter_needs_a_str() {
        use crate::SignatureError;
        let text = BufferArg::str("ab");
        assert_eq!(text.kind(), BufferKind::STR);
        assert!(validate_buffers(&[BufferKind::STR], &[BufferArg::str("ab")], usize::MAX).is_ok());
        assert!(matches!(
            validate_buffers(&[BufferKind::STR], &[BufferArg::read(b"ab")], usize::MAX),
            Err(SignatureError::Utf8 { buffer: 0 })
        ));
        // A `&str` is also its bytes.
        let bytes = BufferKind::new(ScalarType::U8, false);
        assert!(validate_buffers(&[bytes], &[BufferArg::str("ab")], usize::MAX).is_ok());
    }

    /// A backend's limit rejects only longer slices; without one, a slice
    /// past `i32::MAX` elements is accepted.
    #[test]
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    fn only_a_slice_past_the_backend_limit_is_rejected() {
        // Zeroed pages are mapped lazily: the 4 GiB are never touched.
        let long = vec![0u16; i32::MAX as usize + 1];
        let kinds = [BufferKind::new(ScalarType::U16, false)];
        let limit = i32::MAX as usize;
        assert!(matches!(
            validate_buffers(&kinds, &[BufferArg::read(&long)], limit),
            Err(crate::SignatureError::Length {
                buffer: 0,
                limit: _
            })
        ));
        let fits = &long[..limit];
        assert!(validate_buffers(&kinds, &[BufferArg::read(fits)], limit).is_ok());
        assert!(validate_buffers(&kinds, &[BufferArg::read(&long)], usize::MAX).is_ok());
    }

    #[test]
    #[should_panic(expected = "expected a i64 buffer")]
    fn borrowing_another_element_type_panics() {
        BufferArg::read(&[1i32]).as_slice::<i64>();
    }
}
