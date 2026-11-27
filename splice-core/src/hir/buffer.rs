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
    context: Option<&'static ContextSignature>,
}

impl BufferKind {
    /// A shared `i32` slice.
    pub const READ_I32: Self = Self::new(ScalarType::I32, false);
    /// A writable `i32` slice.
    pub const WRITE_I32: Self = Self::new(ScalarType::I32, true);
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
            context: None,
        }
    }
    #[doc(hidden)]
    pub const fn context(signature: &'static ContextSignature) -> Self {
        Self {
            element: ScalarType::I32,
            writable: true,
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
    pub const fn writable(self) -> bool {
        self.writable
    }
    /// Whether two distinct parameters with these kinds may overlap in a
    /// valid kernel call. A writable slice has exclusive access to its bytes.
    pub const fn may_alias(self, other: Self) -> bool {
        !self.writable() && !other.writable()
    }
    fn accepts(self, actual: Self) -> bool {
        self.same_type(actual) && (!self.writable || actual.writable)
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

/// Check arity, mutability and representable slice lengths before entering native code.
///
/// Every slice is at most `i32::MAX` elements long: backends rely on it, for
/// example LLVM sign-extends 32-bit gather offsets, which is the index itself
/// only below `2^31`.
#[inline]
pub fn validate_buffers(
    kinds: &[BufferKind],
    args: &[BufferArg<'_>],
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
        if !kind.accepts(arg.kind()) {
            return Err(ReadOnly { buffer });
        }
        if arg.len() > i32::MAX as usize {
            return Err(Length { buffer });
        }
    }
    Ok(())
}

/// Check the slices passed for the buffer parameters `callee` of a call.
/// Multiple immutable views may share a pointer register; a mutable view
/// needs exclusive access. Distinct registers come from borrows that Rust
/// checked in the kernel's source, such as the halves of `split_at_mut`.
pub fn check_forwarding(
    callee: &[BufferKind],
    actual: &[Slice],
    reg_type: impl Fn(Reg) -> Option<ValueType>,
) -> crate::Result<()> {
    if callee.len() != actual.len() {
        return Err(crate::BuildError::verify(format!(
            "expected {} buffer arguments, got {}",
            callee.len(),
            actual.len()
        )));
    }
    let mut seen: HashMap<Reg, BufferKind> = HashMap::new();
    for (i, (&required, &slice)) in callee.iter().zip(actual).enumerate() {
        let Some(ValueType::Ptr(kind)) = reg_type(slice.ptr) else {
            return Err(crate::BuildError::verify(format!(
                "buffer argument {i} is not a slice pointer"
            )));
        };
        if reg_type(slice.len) != Some(ValueType::Scalar(ScalarType::Index)) {
            return Err(crate::BuildError::verify(format!(
                "buffer argument {i} has no usize length"
            )));
        }
        if !required.same_type(kind) {
            return Err(crate::BuildError::verify(format!(
                "buffer argument {i} has wrong element type"
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

    /// Backends address slices with signed 32-bit offsets (BE-19).
    #[test]
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    fn a_slice_longer_than_i32_max_is_rejected() {
        // Zeroed pages are mapped lazily: the 4 GiB are never touched.
        let long = vec![0u16; i32::MAX as usize + 1];
        let kinds = [BufferKind::new(ScalarType::U16, false)];
        assert!(matches!(
            validate_buffers(&kinds, &[BufferArg::read(&long)]),
            Err(crate::SignatureError::Length { buffer: 0 })
        ));
        let fits = &long[..i32::MAX as usize];
        assert!(validate_buffers(&kinds, &[BufferArg::read(fits)]).is_ok());
    }

    #[test]
    #[should_panic(expected = "expected a i64 buffer")]
    fn borrowing_another_element_type_panics() {
        BufferArg::read(&[1i32]).as_slice::<i64>();
    }
}
