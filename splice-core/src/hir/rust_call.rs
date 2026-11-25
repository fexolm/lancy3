//! Rust calls: one instruction, one native entry and one packet layout,
//! which every backend materializes from [`RustTarget::plan`].
use super::{Buffer, BufferKind, ContextSlice, ScalarType, ValueType};
use crate::{ContextArguments, RustEntry, StaticRustSymbol};
use std::collections::HashMap;

/// The Rust code a [`super::Instr::RustCall`] runs.
#[derive(Clone, Copy, Debug)]
pub enum RustTarget {
    /// A Rust function selected when the kernel is staged.
    Static(&'static StaticRustSymbol),
    /// An operation of the Rust context passed as `buffer`.
    Context { buffer: Buffer, operation: u32 },
}

impl PartialEq for RustTarget {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Static(a), Self::Static(b)) => std::ptr::eq(a, b),
            (
                Self::Context { buffer, operation },
                Self::Context {
                    buffer: b,
                    operation: o,
                },
            ) => buffer == b && operation == o,
            _ => false,
        }
    }
}

/// The operands of a Rust call are `i32` ABI words and it returns one.
pub(crate) const WORD: ValueType = ValueType::Scalar(ScalarType::I32);

/// A value a backend stores into the packet at [`PacketStore::at`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketValue {
    /// Operand `n` of the call: an `i32`.
    Word(usize),
    /// The data pointer of a slice: a pointer.
    Data(Buffer),
    /// The length of a slice: an `i32`.
    Length(Buffer),
    /// A constant `u32`.
    Tag(u32),
    /// The address of a static symbol: a 64-bit integer.
    Symbol(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PacketStore {
    /// Byte offset in the packet.
    pub at: u32,
    pub value: PacketValue,
}

/// How a backend emits a Rust call: store `stores` into an 8-byte aligned
/// packet of `bytes`, call `entry(receiver, operation, packet, 0)` and
/// forward a nonzero trap code of the returned status; the low word of the
/// status is the result. The receiver is the data pointer of the context
/// buffer, or the packet itself when there is none.
#[derive(Clone, Debug)]
pub struct RustCallPlan {
    pub entry: RustEntry,
    pub context: Option<Buffer>,
    pub operation: u32,
    pub bytes: u32,
    pub stores: Vec<PacketStore>,
}

impl RustTarget {
    /// The words and slices the target takes.
    fn shape(self, kinds: &[BufferKind]) -> crate::Result<(usize, u32)> {
        match self {
            Self::Context { buffer, operation } => kinds
                .get(buffer.index())
                .and_then(|kind| kind.context_signature())
                .ok_or_else(|| crate::BuildError::verify("expected a Rust context descriptor"))?
                .shape(operation)
                .ok_or_else(|| crate::BuildError::verify("unknown Rust context operation")),
            Self::Static(symbol) => {
                let (words, slices) = symbol.shape();
                Ok((words as usize, slices))
            }
        }
    }

    fn name(self) -> String {
        match self {
            Self::Static(symbol) => format!("Rust function {}", symbol.name()),
            Self::Context { operation, .. } => format!("Rust context operation {operation}"),
        }
    }

    /// Check a call's operands: `i32` words, and slices that the call may
    /// borrow without aliasing a writable one.
    pub fn check(
        self,
        kinds: &[BufferKind],
        args: impl ExactSizeIterator<Item = Option<ValueType>>,
        slices: &[ContextSlice],
    ) -> crate::Result<()> {
        let (words, borrows) = self.shape(kinds)?;
        if args.len() != words || slices.len() != borrows as usize {
            return Err(crate::BuildError::verify(format!(
                "{} takes {words} words and {borrows} slices, got {} and {}",
                self.name(),
                args.len(),
                slices.len()
            )));
        }
        if args.into_iter().any(|ty| ty != Some(WORD)) {
            return Err(crate::BuildError::verify(format!(
                "expected i32 operands for {}",
                self.name()
            )));
        }
        let mut seen = HashMap::new();
        for &ContextSlice { buffer, writable } in slices {
            let kind = kinds
                .get(buffer.index())
                .filter(|kind| kind.context_signature().is_none())
                .ok_or_else(|| {
                    crate::BuildError::verify(format!(
                        "expected slice operands for {}",
                        self.name()
                    ))
                })?;
            if writable && !kind.writable() {
                return Err(crate::BuildError::verify(format!(
                    "{} requires a writable slice",
                    self.name()
                )));
            }
            if seen
                .insert(buffer, writable)
                .is_some_and(|prior| prior || writable)
            {
                return Err(crate::BuildError::verify(
                    "writable Rust slice operand aliases another operand",
                ));
            }
        }
        Ok(())
    }

    /// The native call of a checked Rust call with `words` operands and
    /// `slices`. The operands follow the layout of [`ContextArguments`]; a
    /// static call's packet starts with the symbol's address (see
    /// [`crate::invoke_static`]), so no backend embeds a pointer constant.
    pub fn plan(self, kinds: &[BufferKind], words: usize, slices: &[ContextSlice]) -> RustCallPlan {
        let layout = ContextArguments::packet_layout(words as u32, slices.len() as u32)
            .expect("a checked Rust call has a valid packet");
        let (entry, context, operation, header): (RustEntry, _, _, _) = match self {
            Self::Context { buffer, operation } => {
                (crate::invoke_context, Some(buffer), operation, 0)
            }
            Self::Static(_) => (crate::invoke_static, None, 0, crate::STATIC_HEADER),
        };
        let mut stores = Vec::new();
        if let Self::Static(symbol) = self {
            stores.push(PacketStore {
                at: 0,
                value: PacketValue::Symbol(std::ptr::from_ref(symbol) as usize),
            });
        }
        stores.extend((0..words).map(|n| PacketStore {
            at: header + n as u32 * 4,
            value: PacketValue::Word(n),
        }));
        for (index, &ContextSlice { buffer, writable }) in slices.iter().enumerate() {
            let at = header + layout.slices_at + index as u32 * ContextArguments::SLICE_BYTES;
            let tag = ContextArguments::view_tag(kinds[buffer.index()].element(), writable);
            stores.extend([
                PacketStore {
                    at,
                    value: PacketValue::Data(buffer),
                },
                PacketStore {
                    at: at + ContextArguments::LENGTH_OFFSET,
                    value: PacketValue::Length(buffer),
                },
                PacketStore {
                    at: at + ContextArguments::ELEMENT_OFFSET,
                    value: PacketValue::Tag(tag),
                },
            ]);
        }
        RustCallPlan {
            entry,
            context,
            operation,
            // A packet without operands is still a valid pointer.
            bytes: (header + layout.bytes).max(8),
            stores,
        }
    }
}
