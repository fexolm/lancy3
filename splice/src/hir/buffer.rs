use crate::simd::Element;
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
}

impl BufferKind {
    #[allow(non_upper_case_globals)]
    pub const Read: Self = Self::new(Element::I32, false);
    #[allow(non_upper_case_globals)]
    pub const Write: Self = Self::new(Element::I32, true);
    pub const fn new(element: Element, writable: bool) -> Self {
        Self { element, writable }
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
        self.element == actual.element && (!self.writable || actual.writable)
    }
}

/// A borrowed typed slice. Writable slices also support loads.
#[derive(Debug)]
pub enum BufferArg<'a> {
    Read(&'a [i32]),
    Write(&'a mut [i32]),
    ReadU16(&'a [u16]),
    WriteU16(&'a mut [u16]),
    ReadI64(&'a [i64]),
    WriteI64(&'a mut [i64]),
    ReadF32(&'a [f32]),
    WriteF32(&'a mut [f32]),
    ReadF64(&'a [f64]),
    WriteF64(&'a mut [f64]),
}

macro_rules! buffer_from {
    ($ty:ty,$read:ident,$write:ident) => {
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
buffer_from!(i64, ReadI64, WriteI64);
buffer_from!(f32, ReadF32, WriteF32);
buffer_from!(f64, ReadF64, WriteF64);

impl BufferArg<'_> {
    pub fn kind(&self) -> BufferKind {
        match self {
            Self::Read(_) => BufferKind::new(Element::I32, false),
            Self::Write(_) => BufferKind::new(Element::I32, true),
            Self::ReadU16(_) => BufferKind::new(Element::U16, false),
            Self::WriteU16(_) => BufferKind::new(Element::U16, true),
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
            Self::Read(s) => s.len(),
            Self::Write(s) => s.len(),
            Self::ReadU16(s) => s.len(),
            Self::WriteU16(s) => s.len(),
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
            Self::Read(s) => s.as_ptr().cast_mut().cast(),
            Self::Write(s) => s.as_mut_ptr().cast(),
            Self::ReadU16(s) => s.as_ptr().cast_mut().cast(),
            Self::WriteU16(s) => s.as_mut_ptr().cast(),
            Self::ReadI64(s) => s.as_ptr().cast_mut().cast(),
            Self::WriteI64(s) => s.as_mut_ptr().cast(),
            Self::ReadF32(s) => s.as_ptr().cast_mut().cast(),
            Self::WriteF32(s) => s.as_mut_ptr().cast(),
            Self::ReadF64(s) => s.as_ptr().cast_mut().cast(),
            Self::WriteF64(s) => s.as_mut_ptr().cast(),
        }
    }
    /// Access an i32 buffer. Panics if this argument has a different element type.
    pub fn as_slice(&self) -> &[i32] {
        match self {
            Self::Read(s) => s,
            Self::Write(s) => s,
            _ => panic!("expected an i32 buffer"),
        }
    }
}

/// Check arity, mutability and representable slice lengths before entering native code.
pub fn validate_buffers(kinds: &[BufferKind], args: &[BufferArg<'_>]) -> Result<(), String> {
    if kinds.len() != args.len() {
        return Err(format!(
            "expected {} buffers, got {}",
            kinds.len(),
            args.len()
        ));
    }
    for (i, (&kind, arg)) in kinds.iter().zip(args).enumerate() {
        if kind.element() != arg.kind().element() {
            return Err(format!(
                "buffer {i} expects {} elements",
                kind.element().name()
            ));
        }
        if !kind.accepts(arg.kind()) {
            return Err(format!("buffer {i} requires a writable slice"));
        }
        if arg.len() > i32::MAX as usize {
            return Err(format!("buffer {i} length exceeds i32::MAX"));
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
) -> Result<(), String> {
    if callee.len() != actual.len() {
        return Err(format!(
            "expected {} buffer arguments, got {}",
            callee.len(),
            actual.len()
        ));
    }
    if actual.is_empty() {
        return Ok(());
    }
    let mut seen: HashMap<_, BufferKind> = HashMap::new();
    for (&required, &buffer) in callee.iter().zip(actual) {
        let kind = caller
            .get(buffer.index())
            .ok_or_else(|| format!("unknown buffer {}", buffer.0))?;
        if required.element() != kind.element() {
            return Err(format!("buffer {} has wrong element type", buffer.0));
        }
        if !required.accepts(*kind) {
            return Err(format!("buffer {} is read-only", buffer.0));
        }
        if actual.len() > 1
            && seen
                .insert(buffer, required)
                .is_some_and(|prior| !prior.may_alias(required))
        {
            return Err(format!(
                "writable buffer {} aliases another call argument",
                buffer.0
            ));
        }
    }
    Ok(())
}
