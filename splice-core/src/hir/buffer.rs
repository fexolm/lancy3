use crate::KernelScalar;
use crate::hir::{DataType, Reg, ScalarType, Slice, StructType, ValueType};
use crate::runtime_context::{ContextSignature, RuntimeContext};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// What one element of a slice holds. Its [`bytes`](Self::bytes) are the
/// one layout fact a backend addresses elements by; an element that is
/// not a scalar is read whole by its own operation (a slice element by
/// [`KernelBuilder::load_slice`]).
///
/// [`KernelBuilder::load_slice`]: crate::frontend::KernelBuilder::load_slice
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Element {
    Scalar(ScalarType),
    /// A shared slice of `kind`, `&[T]` or `&str`: a pointer and a length,
    /// laid out as Rust's slice reference and [`SliceDescriptor`].
    ///
    /// [`SliceDescriptor`]: crate::call::SliceDescriptor
    Slice(&'static BufferKind),
    /// A struct, read and written one scalar field at a time
    /// ([`KernelBuilder::load_field`]).
    ///
    /// [`KernelBuilder::load_field`]: crate::frontend::KernelBuilder::load_field
    Struct(&'static StructType),
}

impl Element {
    /// The distance between consecutive elements in memory.
    pub const fn bytes(self) -> usize {
        match self {
            Self::Scalar(ty) => ty.bytes(),
            Self::Slice(_) => size_of::<crate::call::SliceDescriptor>(),
            Self::Struct(ty) => ty.size,
        }
    }
    /// The byte offset of the length in a [`Slice`](Self::Slice) element.
    pub const SLICE_LEN_OFFSET: usize = std::mem::offset_of!(crate::call::SliceDescriptor, len);
    pub const fn scalar(self) -> Option<ScalarType> {
        match self {
            Self::Scalar(ty) => Some(ty),
            Self::Slice(_) | Self::Struct(_) => None,
        }
    }
    /// Field `field` of a struct element: its byte offset in the element
    /// and its type. A scalar element is its own field 0.
    pub fn field(self, field: usize) -> Option<(usize, ScalarType)> {
        match self {
            Self::Scalar(ty) => DataType::Scalar(ty).leaf(field),
            Self::Struct(ty) => DataType::Struct(ty).leaf(field),
            Self::Slice(_) => None,
        }
    }
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(ty) => f.write_str(ty.name()),
            Self::Slice(kind) if kind.utf8 => f.write_str("&str"),
            Self::Slice(kind) => write!(f, "&[{}]", kind.element),
            Self::Struct(ty) => f.write_str(ty.name),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferKind {
    element: Element,
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
            element: Element::Scalar(element),
            writable,
            utf8: false,
            context: None,
        }
    }
    /// A shared slice whose elements are slices of `inner`: `&[&[T]]` or
    /// `&[&str]`. Inner slices are shared (`&[&mut [T]]` is not supported).
    pub const fn slices(inner: &'static BufferKind) -> Self {
        assert!(
            !inner.writable && inner.context.is_none(),
            "a slice element is a shared slice"
        );
        Self {
            element: Element::Slice(inner),
            writable: false,
            utf8: false,
            context: None,
        }
    }
    /// A slice of values of `ty`: a scalar slice, or one whose elements are
    /// structs.
    ///
    /// # Panics
    /// If `ty` is `bool` (see [`new`](Self::new)).
    pub const fn data(ty: DataType, writable: bool) -> Self {
        match ty {
            DataType::Scalar(scalar) => Self::new(scalar, writable),
            DataType::Struct(layout) => Self {
                element: Element::Struct(layout),
                writable,
                utf8: false,
                context: None,
            },
        }
    }
    #[doc(hidden)]
    pub const fn context(signature: &'static ContextSignature) -> Self {
        Self {
            element: Element::Scalar(ScalarType::I32),
            writable: true,
            utf8: false,
            context: Some(signature),
        }
    }
    #[doc(hidden)]
    pub const fn context_signature(self) -> Option<&'static ContextSignature> {
        self.context
    }
    pub const fn element(self) -> Element {
        self.element
    }
    /// The element type of a slice of scalars.
    pub const fn scalar(self) -> Option<ScalarType> {
        self.element.scalar()
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

    /// A slice of slices, `&[&[T]]`: each element is Rust's slice
    /// reference, which [`Element::Slice`] reads in place.
    pub fn slices<T: BufferElement>(slices: &'a [&'a [T]]) -> Self {
        Self {
            data: NonNull::from(slices).cast(),
            len: slices.len(),
            kind: BufferKind::slices(const { &BufferKind::new(T::TYPE, false) }),
            borrow: PhantomData,
        }
    }

    /// A shared slice of kernel data: structs that derive `KernelType`
    /// (or the scalar of a `transparent` newtype).
    pub fn records<T: crate::KernelData>(slice: &'a [T]) -> Self {
        Self {
            data: NonNull::from(slice).cast(),
            len: slice.len(),
            kind: const { BufferKind::data(crate::data_type::<T>(), false) },
            borrow: PhantomData,
        }
    }

    /// A writable slice of kernel data.
    pub fn records_mut<T: crate::KernelData>(slice: &'a mut [T]) -> Self {
        let len = slice.len();
        Self {
            data: NonNull::from(slice).cast(),
            len,
            kind: const { BufferKind::data(crate::data_type::<T>(), true) },
            borrow: PhantomData,
        }
    }

    /// A slice of `&str`s, `&[&str]`.
    pub fn strs(texts: &'a [&'a str]) -> Self {
        Self {
            data: NonNull::from(texts).cast(),
            len: texts.len(),
            kind: BufferKind::slices(&BufferKind::STR),
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
            self.kind.context.is_none() && self.kind.element == Element::Scalar(T::TYPE),
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
        if arg.len() > longest
            || !structs_within(kind.element(), arg.len, longest)
            || !nested_within(kind.element(), arg.data, arg.len, longest)
        {
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

/// Whether the bytes of a slice of structs fit the backend's limit: a
/// backend that addresses at most `longest` elements of a scalar slice
/// (Lancy, `i32`) computes byte offsets into a struct slice in the same
/// width, so a limit below every Rust slice length bounds its bytes.
fn structs_within(element: Element, len: usize, longest: usize) -> bool {
    let Element::Struct(ty) = element else {
        return true;
    };
    longest >= isize::MAX as usize || len.checked_mul(ty.size).is_some_and(|b| b <= longest)
}

/// Whether every slice inside the `len` elements at `data` is at most
/// `longest` long: a backend's limit holds for each slice a kernel reads.
/// Only a backend with a limit below every Rust slice length pays the walk.
#[allow(unsafe_code)]
fn nested_within(element: Element, data: NonNull<u8>, len: usize, longest: usize) -> bool {
    let Element::Slice(inner) = element else {
        return true;
    };
    if longest >= isize::MAX as usize && inner.element.scalar().is_some() {
        return true;
    }
    // SAFETY: a `BufferArg` of slice elements borrows `len` Rust slice
    // references (`BufferArg::slices`/`strs`), whose layout is the
    // descriptor's (asserted next to `SliceDescriptor`).
    let slices = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<crate::call::SliceDescriptor>(), len)
    };
    slices.iter().all(|s| {
        let data = NonNull::new(s.data).unwrap_or(NonNull::dangling());
        s.len as usize <= longest && nested_within(inner.element, data, s.len as usize, longest)
    })
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

    /// A backend's length limit holds for every slice a kernel reads, the
    /// elements of a slice of slices included.
    #[test]
    fn a_slice_of_slices_checks_each_length() {
        use crate::SignatureError;
        let (short, long) = ([1u16], [1u16, 2, 3]);
        let rows: [&[u16]; 2] = [&short, &long];
        let arg = BufferArg::slices(&rows);
        let kind = BufferKind::slices(const { &BufferKind::new(ScalarType::U16, false) });
        assert_eq!(arg.kind(), kind);
        assert_eq!(kind.element().bytes(), 16);
        assert!(validate_buffers(&[kind], &[BufferArg::slices(&rows)], 3).is_ok());
        assert!(matches!(
            validate_buffers(&[kind], &[BufferArg::slices(&rows)], 2),
            Err(SignatureError::Length {
                buffer: 0,
                limit: 2
            })
        ));
        let texts = ["ab", "€"];
        let strs = BufferKind::slices(&BufferKind::STR);
        assert!(validate_buffers(&[strs], &[BufferArg::strs(&texts)], 3).is_ok());
        assert!(validate_buffers(&[strs], &[BufferArg::strs(&texts)], 2).is_err());
        // A `&[&[u8]]` is not a `&[&str]`, nor the other way round.
        let bytes: [&[u8]; 1] = [b"ab"];
        assert!(matches!(
            validate_buffers(&[strs], &[BufferArg::slices(&bytes)], usize::MAX),
            Err(SignatureError::Element { buffer: 0, .. })
        ));
        assert_eq!(
            validate_buffers(&[kind], &[BufferArg::strs(&texts)], usize::MAX)
                .unwrap_err()
                .to_string(),
            "buffer 0 expects &[u16] elements"
        );
    }

    #[test]
    #[should_panic(expected = "expected a i64 buffer")]
    fn borrowing_another_element_type_panics() {
        BufferArg::read(&[1i32]).as_slice::<i64>();
    }
}
