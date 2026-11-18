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
pub enum BufferKind {
    Read,
    Write,
}

impl BufferKind {
    fn accepts(self, actual: Self) -> bool {
        self == Self::Read || actual == Self::Write
    }
}

/// A borrowed i32 slice. Writable slices also support loads.
#[derive(Debug)]
pub enum BufferArg<'a> {
    Read(&'a [i32]),
    Write(&'a mut [i32]),
}

impl BufferArg<'_> {
    pub fn kind(&self) -> BufferKind {
        match self {
            Self::Read(_) => BufferKind::Read,
            Self::Write(_) => BufferKind::Write,
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    pub fn as_slice(&self) -> &[i32] {
        match self {
            Self::Read(slice) => slice,
            Self::Write(slice) => slice,
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
    let mut seen = HashMap::new();
    for (&required, &buffer) in callee.iter().zip(actual) {
        let kind = caller
            .get(buffer.index())
            .ok_or_else(|| format!("unknown buffer {}", buffer.0))?;
        if !required.accepts(*kind) {
            return Err(format!("buffer {} is read-only", buffer.0));
        }
        if actual.len() > 1
            && seen
                .insert(buffer, required)
                .is_some_and(|prior| prior == BufferKind::Write || required == BufferKind::Write)
        {
            return Err(format!(
                "writable buffer {} aliases another call argument",
                buffer.0
            ));
        }
    }
    Ok(())
}
