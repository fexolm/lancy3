use crate::simd::{Element, LaneElement};
use crate::{ContextSignature, RuntimeContext};
use std::collections::HashMap;

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

/// A borrowed typed slice. Writable slices also support loads.
#[derive(Debug)]
pub enum BufferArg<'a> {
    #[doc(hidden)]
    Context(&'a mut RuntimeContext<'a>),
    Read(&'a [i32]),
    Write(&'a mut [i32]),
    ReadU16(&'a [u16]),
    WriteU16(&'a mut [u16]),
    ReadU32(&'a [u32]),
    WriteU32(&'a mut [u32]),
    ReadI64(&'a [i64]),
    WriteI64(&'a mut [i64]),
    ReadF32(&'a [f32]),
    WriteF32(&'a mut [f32]),
    ReadF64(&'a [f64]),
    WriteF64(&'a mut [f64]),
}

/// An element type that can be borrowed from a [`BufferArg`].
pub trait BufferElement: LaneElement {
    fn read_arg(slice: &[Self]) -> BufferArg<'_>;
    fn write_arg(slice: &mut [Self]) -> BufferArg<'_>;
    #[doc(hidden)]
    fn borrow<'a>(buffer: &'a BufferArg<'_>) -> Option<&'a [Self]>;
}

macro_rules! buffer_from {
    ($ty:ty,$read:ident,$write:ident) => {
        impl BufferElement for $ty {
            fn read_arg(slice: &[Self]) -> BufferArg<'_> {
                BufferArg::$read(slice)
            }
            fn write_arg(slice: &mut [Self]) -> BufferArg<'_> {
                BufferArg::$write(slice)
            }
            fn borrow<'a>(buffer: &'a BufferArg<'_>) -> Option<&'a [Self]> {
                match buffer {
                    BufferArg::$read(slice) => Some(slice),
                    BufferArg::$write(slice) => Some(slice),
                    _ => None,
                }
            }
        }
        impl<'a> From<&'a [$ty]> for BufferArg<'a> {
            fn from(s: &'a [$ty]) -> Self {
                Self::$read(s)
            }
        }
        impl<'a> From<&'a mut [$ty]> for BufferArg<'a> {
            fn from(s: &'a mut [$ty]) -> Self {
                Self::$write(s)
            }
        }
    };
}
buffer_from!(i32, Read, Write);
buffer_from!(u16, ReadU16, WriteU16);
buffer_from!(u32, ReadU32, WriteU32);
buffer_from!(i64, ReadI64, WriteI64);
buffer_from!(f32, ReadF32, WriteF32);
buffer_from!(f64, ReadF64, WriteF64);

impl BufferArg<'_> {
    /// Reborrow a dynamic column through a shared reference.
    pub fn as_read_only(&self) -> BufferArg<'_> {
        match self {
            Self::Context(_) => panic!("a Rust context cannot be a read-only slice"),
            Self::Read(s) => BufferArg::Read(s),
            Self::Write(s) => BufferArg::Read(s),
            Self::ReadU16(s) => BufferArg::ReadU16(s),
            Self::WriteU16(s) => BufferArg::ReadU16(s),
            Self::ReadU32(s) => BufferArg::ReadU32(s),
            Self::WriteU32(s) => BufferArg::ReadU32(s),
            Self::ReadI64(s) => BufferArg::ReadI64(s),
            Self::WriteI64(s) => BufferArg::ReadI64(s),
            Self::ReadF32(s) => BufferArg::ReadF32(s),
            Self::WriteF32(s) => BufferArg::ReadF32(s),
            Self::ReadF64(s) => BufferArg::ReadF64(s),
            Self::WriteF64(s) => BufferArg::ReadF64(s),
        }
    }

    pub fn kind(&self) -> BufferKind {
        match self {
            Self::Context(context) => BufferKind::context(context.signature()),
            Self::Read(_) => BufferKind::new(Element::I32, false),
            Self::Write(_) => BufferKind::new(Element::I32, true),
            Self::ReadU16(_) => BufferKind::new(Element::U16, false),
            Self::WriteU16(_) => BufferKind::new(Element::U16, true),
            Self::ReadU32(_) => BufferKind::new(Element::U32, false),
            Self::WriteU32(_) => BufferKind::new(Element::U32, true),
            Self::ReadI64(_) => BufferKind::new(Element::I64, false),
            Self::WriteI64(_) => BufferKind::new(Element::I64, true),
            Self::ReadF32(_) => BufferKind::new(Element::F32, false),
            Self::WriteF32(_) => BufferKind::new(Element::F32, true),
            Self::ReadF64(_) => BufferKind::new(Element::F64, false),
            Self::WriteF64(_) => BufferKind::new(Element::F64, true),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Context(_) => 0,
            Self::Read(s) => s.len(),
            Self::Write(s) => s.len(),
            Self::ReadU16(s) => s.len(),
            Self::WriteU16(s) => s.len(),
            Self::ReadU32(s) => s.len(),
            Self::WriteU32(s) => s.len(),
            Self::ReadI64(s) => s.len(),
            Self::WriteI64(s) => s.len(),
            Self::ReadF32(s) => s.len(),
            Self::WriteF32(s) => s.len(),
            Self::ReadF64(s) => s.len(),
            Self::WriteF64(s) => s.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Raw slice address for validated native backends. Only writable arguments permit writes.
    pub fn data(&mut self) -> *mut u8 {
        match self {
            Self::Context(context) => std::ptr::from_mut(&mut **context).cast(),
            Self::Read(s) => s.as_ptr().cast_mut().cast(),
            Self::Write(s) => s.as_mut_ptr().cast(),
            Self::ReadU16(s) => s.as_ptr().cast_mut().cast(),
            Self::WriteU16(s) => s.as_mut_ptr().cast(),
            Self::ReadU32(s) => s.as_ptr().cast_mut().cast(),
            Self::WriteU32(s) => s.as_mut_ptr().cast(),
            Self::ReadI64(s) => s.as_ptr().cast_mut().cast(),
            Self::WriteI64(s) => s.as_mut_ptr().cast(),
            Self::ReadF32(s) => s.as_ptr().cast_mut().cast(),
            Self::WriteF32(s) => s.as_mut_ptr().cast(),
            Self::ReadF64(s) => s.as_ptr().cast_mut().cast(),
            Self::WriteF64(s) => s.as_mut_ptr().cast(),
        }
    }
    /// Borrow the slice as `T`. Panics if this argument has a different element type.
    pub fn as_slice<T: BufferElement>(&self) -> &[T] {
        T::borrow(self).unwrap_or_else(|| panic!("expected a {} buffer", T::ELEMENT.name()))
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
