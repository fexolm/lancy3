//! A synchronous Rust dispatcher borrowed for one native invocation.
use splice_target::abi::{Trap, rust_entry};
use std::{
    fmt,
    hash::{Hash, Hasher},
};

/// The operations of a kernel and of methods that use the same runtime object.
/// The generated kernel owns one static root, whose identity is part of linking.
pub struct ContextSignature {
    local: &'static [u32],
    borrows: &'static [u32],
    children: &'static [&'static Self],
    size: u32,
    depth: u32,
}

impl fmt::Debug for ContextSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextSignature")
            .field("local_operations", &self.local.len())
            .field("children", &self.children.len())
            .field("size", &self.size)
            .finish()
    }
}

/// An operation relative to one descriptor. Child operations belong to a
/// contiguous range after the local operations and the earlier children.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextOperation {
    Local(u32),
    Child { index: u32, operation: u32 },
}

/// A range of Rust operations in an active kernel specialization.
#[derive(Clone, Copy)]
pub struct ContextFrame {
    signature: &'static ContextSignature,
    base: u32,
}

const MAX_CONTEXT_DEPTH: u32 = 64;

impl ContextSignature {
    pub const fn new(arities: &'static [u32]) -> Self {
        Self::with_children(arities, &[])
    }

    pub const fn with_children(local: &'static [u32], children: &'static [&'static Self]) -> Self {
        Self::with_borrows(local, &[], children)
    }

    /// # Panics
    ///
    /// On an invalid layout, which is a compile error when this initializes
    /// a `static`: `borrows` must be empty or name one slice count per local
    /// operation, every packet must fit `i32`, and nesting is limited to 64
    /// levels and `u32::MAX` operations.
    ///
    /// ```compile_fail,E0080
    /// # use splice_core::ContextSignature;
    /// // Four billion words do not fit an i32 packet.
    /// static HUGE: ContextSignature = ContextSignature::new(&[u32::MAX]);
    /// ```
    /// ```compile_fail,E0080
    /// # use splice_core::ContextSignature;
    /// // Two operations, but slice counts for one.
    /// static MISMATCH: ContextSignature = ContextSignature::with_borrows(&[1, 1], &[0], &[]);
    /// ```
    /// ```compile_fail,E0080
    /// # use splice_core::ContextSignature;
    /// macro_rules! double {
    ///     ($previous:ident;) => {};
    ///     ($previous:ident; $next:ident $(, $rest:ident)*) => {
    ///         static $next: ContextSignature = ContextSignature::with_children(&[], &[&$previous; 2]);
    ///         double!($next; $($rest),*);
    ///     };
    /// }
    /// static LEAF: ContextSignature = ContextSignature::new(&[2]);
    /// // 2^32 operations.
    /// double!(LEAF; N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16,
    ///     N17, N18, N19, N20, N21, N22, N23, N24, N25, N26, N27, N28, N29, N30, N31, N32);
    /// ```
    /// ```
    /// # use splice_core::ContextSignature;
    /// static VALID: ContextSignature = ContextSignature::with_borrows(&[1, 2], &[0, 1], &[]);
    /// assert_eq!(VALID.arity(1), Some(2));
    /// ```
    pub const fn with_borrows(
        local: &'static [u32],
        borrows: &'static [u32],
        children: &'static [&'static Self],
    ) -> Self {
        assert!(
            local.len() <= u32::MAX as usize && children.len() <= u32::MAX as usize,
            "too many Rust context operations"
        );
        assert!(
            borrows.is_empty() || borrows.len() == local.len(),
            "Rust context slice counts must match its operations"
        );
        let mut i = 0;
        while i < local.len() {
            let slices = if borrows.is_empty() { 0 } else { borrows[i] };
            assert!(
                ContextArguments::packet_layout(local[i], slices).is_some(),
                "Rust context operation packet exceeds i32::MAX bytes"
            );
            i += 1;
        }
        let mut size = local.len() as u32;
        let mut depth = 1;
        i = 0;
        while i < children.len() {
            let child = children[i];
            let Some(next) = size.checked_add(child.size) else {
                panic!("Rust context has more than u32::MAX operations");
            };
            assert!(
                child.depth < MAX_CONTEXT_DEPTH,
                "Rust contexts nest more than 64 levels"
            );
            if depth <= child.depth {
                depth = child.depth + 1;
            }
            size = next;
            i += 1;
        }
        Self {
            local,
            borrows,
            children,
            size,
            depth,
        }
    }

    pub fn operation_at(&self, mut operation: u32) -> Option<ContextOperation> {
        if operation >= self.size {
            return None;
        }
        let local = self.local.len() as u32;
        if operation < local {
            return Some(ContextOperation::Local(operation));
        }
        operation -= local;
        for (index, child) in self.children.iter().enumerate() {
            let size = child.size;
            if operation < size {
                return Some(ContextOperation::Child {
                    index: index as u32,
                    operation,
                });
            }
            operation -= size;
        }
        None
    }

    pub fn arity(&self, operation: u32) -> Option<usize> {
        self.shape(operation).map(|(words, _)| words)
    }

    pub fn shape(&self, mut operation: u32) -> Option<(usize, u32)> {
        let mut signature = self;
        loop {
            match signature.operation_at(operation)? {
                ContextOperation::Local(index) => {
                    let index = index as usize;
                    return Some((
                        signature.local[index] as usize,
                        signature.borrows.get(index).copied().unwrap_or(0),
                    ));
                }
                ContextOperation::Child {
                    index,
                    operation: local,
                } => {
                    signature = signature.children[index as usize];
                    operation = local;
                }
            }
        }
    }
}

impl ContextFrame {
    pub fn root(signature: &'static ContextSignature) -> Self {
        Self { signature, base: 0 }
    }

    pub fn child(self, index: u32) -> crate::Result<Self> {
        let (previous, remaining) = self
            .signature
            .children
            .split_at_checked(index as usize)
            .ok_or_else(|| crate::BuildError::staging("unknown child Rust context"))?;
        let signature = *remaining
            .first()
            .ok_or_else(|| crate::BuildError::staging("unknown child Rust context"))?;
        let mut base = self.base + self.signature.local.len() as u32;
        for child in previous {
            // Root construction validated all counts, and their sum fits u32.
            base += child.size;
        }
        Ok(Self { signature, base })
    }

    pub fn operation(self, local: u32) -> crate::Result<u32> {
        if local as usize >= self.signature.local.len() {
            return Err(crate::BuildError::staging(
                "unknown local Rust context operation",
            ));
        }
        Ok(self.base + local)
    }
}

impl PartialEq for ContextSignature {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Eq for ContextSignature {}

impl Hash for ContextSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::from_ref(self).hash(state);
    }
}

pub struct RuntimeContext<'a> {
    signature: &'static ContextSignature,
    invoke: Dispatcher<'a>,
}

enum Dispatcher<'a> {
    Scalars(&'a mut dyn FnMut(u32, i32, i32) -> i32),
    Packet(&'a mut dyn FnMut(u32, ContextArguments<'_>) -> i32),
}

impl<'a> RuntimeContext<'a> {
    pub fn new(
        signature: &'static ContextSignature,
        invoke: &'a mut dyn FnMut(u32, i32, i32) -> i32,
    ) -> Self {
        Self {
            signature,
            invoke: Dispatcher::Scalars(invoke),
        }
    }

    pub fn new_packet(
        signature: &'static ContextSignature,
        invoke: &'a mut dyn FnMut(u32, ContextArguments<'_>) -> i32,
    ) -> Self {
        Self {
            signature,
            invoke: Dispatcher::Packet(invoke),
        }
    }

    pub fn signature(&self) -> &'static ContextSignature {
        self.signature
    }
}

#[repr(C)]
struct SliceView {
    data: *const u8,
    len: i32,
    element: u32,
}

/// Scalars and slice reborrows valid during one opaque Rust operation.
pub struct ContextArguments<'a> {
    words: &'a [i32],
    slices: std::slice::Iter<'a, SliceView>,
}

/// Layout of a native callback packet. Words start at byte zero.
pub struct ContextPacketLayout {
    pub slices_at: u32,
    pub bytes: u32,
}

/// A Rust operation whose only operands are the words and slices in its packet.
pub trait StaticRustCall {
    const WORDS: u32;
    const SLICES: u32;

    fn invoke(arguments: ContextArguments<'_>) -> i32;
}

/// The packet shape and entry point for one statically selected Rust operation.
#[derive(Debug)]
pub struct StaticRustSymbol {
    name: &'static str,
    words: u32,
    slices: u32,
    entry: unsafe extern "C" fn(*mut u8, u32, *const u8, i32) -> i64,
}

impl StaticRustSymbol {
    pub const fn of<F: StaticRustCall>(name: &'static str) -> Self {
        Self {
            name,
            words: F::WORDS,
            slices: F::SLICES,
            entry: invoke_static_rust_packet::<F>,
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn shape(&self) -> (u32, u32) {
        (self.words, self.slices)
    }

    /// The entry requires live, aligned packet storage initialized according to
    /// this symbol's shape. Each view must refer to an initialized, live slice.
    /// Writable views must be exclusive and disjoint from every other view.
    /// The context and operation arguments of the native ABI are unused.
    pub fn entry(&self) -> unsafe extern "C" fn(*mut u8, u32, *const u8, i32) -> i64 {
        self.entry
    }
}

impl<'a> ContextArguments<'a> {
    pub const SLICE_BYTES: u32 = 16;
    pub const LENGTH_OFFSET: u32 = 8;
    pub const ELEMENT_OFFSET: u32 = 12;
    const WRITABLE: u32 = 1 << 31;

    pub const fn view_tag(element: crate::simd::Element, writable: bool) -> u32 {
        element as u32 | if writable { Self::WRITABLE } else { 0 }
    }

    pub const fn packet_layout(words: u32, slices: u32) -> Option<ContextPacketLayout> {
        let slices_at = (words as u64 * 4).next_multiple_of(8);
        let bytes = slices_at + slices as u64 * Self::SLICE_BYTES as u64;
        if bytes > i32::MAX as u64 {
            return None;
        }
        Some(ContextPacketLayout {
            slices_at: slices_at as u32,
            bytes: bytes as u32,
        })
    }

    pub fn word(&self, index: usize) -> i32 {
        self.words[index]
    }

    fn next<T: crate::simd::LaneElement>(&mut self) -> &'a SliceView {
        let view = self.slices.next().expect("missing Rust slice operand");
        assert_eq!(
            view.element & !Self::WRITABLE,
            T::ELEMENT as u32,
            "wrong Rust slice element type"
        );
        view
    }

    pub fn read<T: crate::simd::LaneElement>(&mut self) -> &'a [T] {
        let view = self.next::<T>();
        // SAFETY: the caller constructed each view from a verified slice
        // descriptor. Native execution is paused during this callback. The
        // shared views can overlap each other and are disjoint from writable
        // views. The returned borrow cannot outlive this packet.
        unsafe { std::slice::from_raw_parts(view.data.cast(), view.len as usize) }
    }

    pub fn write<T: crate::simd::LaneElement>(&mut self) -> &'a mut [T] {
        let view = self.next::<T>();
        assert_ne!(
            view.element & Self::WRITABLE,
            0,
            "Rust slice operand is read-only"
        );
        // SAFETY: a writable view is derived from a validated exclusive
        // buffer and disjoint from every other view in this packet. The
        // iterator consumes the descriptor and has no cloning interface.
        // Native code does not resume until the returned borrow ends.
        unsafe { std::slice::from_raw_parts_mut(view.data.cast_mut().cast(), view.len as usize) }
    }
}

const _: () = {
    assert!(std::mem::size_of::<SliceView>() == ContextArguments::SLICE_BYTES as usize);
    assert!(std::mem::offset_of!(SliceView, len) == ContextArguments::LENGTH_OFFSET as usize);
    assert!(std::mem::offset_of!(SliceView, element) == ContextArguments::ELEMENT_OFFSET as usize);
};

impl fmt::Debug for RuntimeContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeContext")
            .field("signature", &self.signature)
            .finish_non_exhaustive()
    }
}

/// # Safety
/// `raw` must hold the live, exclusive [`RuntimeContext`] borrow provided by
/// the verified invocation's `BufferArg::context`. It may not be retained.
pub unsafe extern "C" fn invoke_runtime_context(
    raw: *mut u8,
    operation: u32,
    a: i32,
    b: i32,
) -> i64 {
    // SAFETY: The backend verifies the context descriptor and borrows it for
    // this synchronous call. Every context, regardless of the captured Rust
    // types, contains the same erased Rust dispatcher representation.
    let context = unsafe { &mut *raw.cast::<RuntimeContext<'_>>() };
    let Some((words @ 0..=2, 0)) = context.signature.shape(operation) else {
        return Trap::Abi.status();
    };
    let scalars = [a, b];
    invoke(
        context,
        operation,
        ContextArguments {
            words: &scalars[..words],
            slices: [].iter(),
        },
    )
}

/// # Safety
/// `raw` must hold the live, exclusive Rust context token. `packet` points
/// to aligned, initialized native call workspace for this operation. Its
/// descriptors must point to validated live slices. Every descriptor tagged
/// writable must grant exclusive access and be disjoint from the other views.
/// Neither pointer nor the temporary borrows may be retained.
pub unsafe extern "C" fn invoke_packet_runtime_context(
    raw: *mut u8,
    operation: u32,
    packet: *const u8,
    _unused: i32,
) -> i64 {
    // SAFETY: the native caller supplies both the context and the correctly
    // sized, aligned packet under the contract above.
    let context = unsafe { &mut *raw.cast::<RuntimeContext<'_>>() };
    let Some((words, count)) = context.signature.shape(operation) else {
        return Trap::Abi.status();
    };
    if (words <= 2 && count == 0) || !matches!(context.invoke, Dispatcher::Packet(_)) {
        return Trap::Abi.status();
    }
    // SAFETY: the native caller supplies initialized, disjoint views and the
    // validated context signature gives the exact packet shape.
    unsafe {
        with_packet(packet, words as u32, count, |args| {
            invoke(context, operation, args)
        })
    }
}

unsafe extern "C" fn invoke_static_rust_packet<F: StaticRustCall>(
    _raw: *mut u8,
    _operation: u32,
    packet: *const u8,
    _unused: i32,
) -> i64 {
    // SAFETY: the symbol pairs this entry with F's exact shape. The checked
    // native caller supplies live, aligned, initialized packet storage and
    // the verified slice views for the duration of this call.
    unsafe {
        with_packet(packet, F::WORDS, F::SLICES, |args| {
            rust_entry(|| F::invoke(args))
        })
    }
}

// The native caller must provide a live, aligned packet of the given shape.
// Its buffers must satisfy the aliasing contract of ContextArguments.
unsafe fn with_packet(
    packet: *const u8,
    words: u32,
    count: u32,
    call: impl FnOnce(ContextArguments<'_>) -> i64,
) -> i64 {
    let Some(layout) = ContextArguments::packet_layout(words, count) else {
        return Trap::Abi.status();
    };
    // SAFETY: the verified packet contains initialized words and exactly
    // count descriptors at the checked aligned byte offset.
    let args = unsafe {
        let descriptors = packet.add(layout.slices_at as usize).cast();
        ContextArguments {
            words: std::slice::from_raw_parts(packet.cast::<i32>(), words as usize),
            slices: std::slice::from_raw_parts(descriptors, count as usize).iter(),
        }
    };
    call(args)
}

fn invoke(context: &mut RuntimeContext<'_>, operation: u32, args: ContextArguments<'_>) -> i64 {
    rust_entry(|| match &mut context.invoke {
        Dispatcher::Scalars(invoke) => invoke(
            operation,
            args.words.first().copied().unwrap_or(0),
            args.words.get(1).copied().unwrap_or(0),
        ),
        Dispatcher::Packet(invoke) => invoke(operation, args),
    })
}

#[cfg(test)]
mod tests {
    use super::{ContextFrame, ContextOperation, ContextSignature};

    #[test]
    fn routes_each_child_and_preserves_repeated_and_empty_ranges() {
        static LEAF: ContextSignature = ContextSignature::new(&[2, 1]);
        static EMPTY: ContextSignature = ContextSignature::new(&[]);
        static CHILD: ContextSignature = ContextSignature::with_children(&[0], &[&LEAF]);
        static ROOT: ContextSignature =
            ContextSignature::with_children(&[1], &[&CHILD, &EMPTY, &CHILD]);

        let frame = ContextFrame::root(&ROOT);
        assert_eq!(frame.operation(0).unwrap(), 0);
        assert!(frame.operation(1).is_err());
        assert_eq!(frame.child(0).unwrap().operation(0).unwrap(), 1);
        assert_eq!(
            frame
                .child(0)
                .unwrap()
                .child(0)
                .unwrap()
                .operation(1)
                .unwrap(),
            3
        );
        assert!(frame.child(1).unwrap().operation(0).is_err());
        assert_eq!(
            frame
                .child(2)
                .unwrap()
                .child(0)
                .unwrap()
                .operation(0)
                .unwrap(),
            5
        );
        assert!(frame.child(3).is_err());

        assert_eq!(ROOT.operation_at(0), Some(ContextOperation::Local(0)));
        assert_eq!(
            ROOT.operation_at(3),
            Some(ContextOperation::Child {
                index: 0,
                operation: 2
            })
        );
        assert_eq!(
            ROOT.operation_at(4),
            Some(ContextOperation::Child {
                index: 2,
                operation: 0
            })
        );
        assert_eq!(
            (0..8).map(|i| ROOT.arity(i)).collect::<Vec<_>>(),
            [
                Some(1),
                Some(0),
                Some(2),
                Some(1),
                Some(0),
                Some(2),
                Some(1),
                None
            ]
        );
    }

    #[test]
    fn expanded_operation_count_must_fit_the_native_operand() {
        macro_rules! double {
            ($previous:ident;) => {};
            ($previous:ident; $next:ident $(, $rest:ident)*) => {
                static $next: ContextSignature =
                    ContextSignature::with_children(&[], &[&$previous; 2]);
                double!($next; $($rest),*);
            };
        }
        static LEAF: ContextSignature = ContextSignature::new(&[2]);
        double!(LEAF;
            N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16,
            N17, N18, N19, N20, N21, N22, N23, N24, N25, N26, N27, N28, N29, N30, N31
        );
        assert_eq!(N31.arity((1 << 31) - 1), Some(2));
        assert_eq!(N31.arity(1 << 31), None);
    }
}
