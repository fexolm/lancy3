use crate::simd::{Element, LaneElement};
use crate::{ContextSignature, RuntimeContext};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// A buffer parameter in a function's separate buffer argument list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Buffer(pub u32);

impl Buffer {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferKind {
    element: Element,
    writable: bool,
    context: Option<&'static ContextSignature>,
}

impl BufferKind {
    #[allow(non_upper_case_globals)]
    pub const Read: Self = Self::new(Element::I32, false);
    #[allow(non_upper_case_globals)]
    pub const Write: Self = Self::new(Element::I32, true);
    pub const fn new(element: Element, writable: bool) -> Self {
        Self {
            element,
            writable,
            context: None,
        }
    }
    #[doc(hidden)]
    pub const fn context(signature: &'static ContextSignature) -> Self {
        Self {
            element: Element::I32,
            writable: true,
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

/// An element type that can be borrowed from a [`BufferArg`].
pub trait BufferElement: LaneElement {}
impl BufferElement for i32 {}
impl BufferElement for u16 {}
impl BufferElement for u32 {}
impl BufferElement for i64 {}
impl BufferElement for f32 {}
impl BufferElement for f64 {}

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
            kind: BufferKind::new(T::ELEMENT, false),
            borrow: PhantomData,
        }
    }

    pub fn write<T: BufferElement>(slice: &'a mut [T]) -> Self {
        let len = slice.len();
        Self {
            data: NonNull::from(slice).cast(),
            len,
            kind: BufferKind::new(T::ELEMENT, true),
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
            self.kind.context.is_none() && self.kind.element == T::ELEMENT,
            "expected a {} buffer",
            T::ELEMENT.name()
        );
        // SAFETY: `read` and `write` made `data` and `len` from a live
        // `&[T]` or `&mut [T]` that this argument borrows, and the element
        // type matches. `&self` permits only shared access to it.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr().cast(), self.len) }
    }
}

/// Check arity, mutability and representable slice lengths before entering native code.
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

/// Check the borrow contract at a static call site. Multiple immutable
/// views may name the same buffer. A mutable view needs exclusive access.
pub fn check_forwarding(
    caller: &[BufferKind],
    callee: &[BufferKind],
    actual: &[Buffer],
) -> crate::Result<()> {
    if callee.len() != actual.len() {
        return Err(crate::BuildError::verify(format!(
            "expected {} buffer arguments, got {}",
            callee.len(),
            actual.len()
        )));
    }
    if actual.is_empty() {
        return Ok(());
    }
    let mut seen: HashMap<_, BufferKind> = HashMap::new();
    for (&required, &buffer) in callee.iter().zip(actual) {
        let kind = caller
            .get(buffer.index())
            .ok_or_else(|| crate::BuildError::verify(format!("unknown buffer {}", buffer.0)))?;
        if !required.same_type(*kind) {
            return Err(crate::BuildError::verify(format!(
                "buffer {} has wrong element type",
                buffer.0
            )));
        }
        if !required.accepts(*kind) {
            return Err(crate::BuildError::verify(format!(
                "buffer {} is read-only",
                buffer.0
            )));
        }
        if actual.len() > 1
            && seen
                .insert(buffer, required)
                .is_some_and(|prior| !prior.may_alias(required))
        {
            return Err(crate::BuildError::verify(format!(
                "writable buffer {} aliases another call argument",
                buffer.0
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
        assert_eq!(arg.kind(), BufferKind::new(Element::F32, true));
        assert_eq!(arg.len(), 3);
        let address = arg.data();
        let shared = arg.as_read_only().expect("a slice has a read-only form");
        assert_eq!(shared.kind(), BufferKind::new(Element::F32, false));
        assert_eq!(shared.as_slice::<f32>(), &[1.5, 2.5, 3.5]);
        assert_eq!(address, data.as_mut_ptr().cast());
    }

    #[test]
    #[should_panic(expected = "expected a i64 buffer")]
    fn borrowing_another_element_type_panics() {
        BufferArg::read(&[1i32]).as_slice::<i64>();
    }
}
