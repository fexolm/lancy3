//! Rust calls: one instruction, one native entry and one packet layout,
//! which every backend materializes from [`RustTarget::plan`].
use super::{BufferKind, ContextSlice, Reg, ValueType};
use crate::runtime_context::{CONTEXT_CALL, STATIC_CALL};
use crate::runtime_context::{ContextArguments, StaticRustSymbol};
use splice_target::abi::RustFunction;
use std::collections::HashMap;

/// The Rust code a [`super::Instr::RustCall`] runs.
#[derive(Clone, Copy, Debug)]
pub enum RustTarget {
    /// A Rust function selected when the kernel is staged.
    Static(&'static StaticRustSymbol),
    /// An operation of the Rust context whose pointer is `context`.
    Context { context: Reg, operation: u32 },
}

impl PartialEq for RustTarget {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Static(a), Self::Static(b)) => std::ptr::eq(a, b),
            (
                Self::Context { context, operation },
                Self::Context {
                    context: c,
                    operation: o,
                },
            ) => context == c && operation == o,
            _ => false,
        }
    }
}

/// A value a backend stores into the packet at [`PacketStore::at`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketValue {
    /// Operand `n` of the call, a scalar of any type, at the low end of its
    /// 8-byte word.
    Word(usize),
    /// The data pointer of a slice, a pointer register.
    Data(Reg),
    /// The length of a slice, a `usize` register, stored as an `i32`.
    Length(Reg),
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
/// packet of `bytes`, call `entry(receiver, operation, packet, 0)`, forward
/// a nonzero trap code of the returned status and otherwise load the
/// call's result, a scalar of its type, from the start of the packet. The
/// receiver is the context pointer, or the packet itself when there is none.
#[derive(Clone, Debug)]
pub struct RustCallPlan {
    pub entry: &'static RustFunction,
    pub context: Option<Reg>,
    pub operation: u32,
    pub bytes: u32,
    pub stores: Vec<PacketStore>,
}

impl RustTarget {
    /// The words and slices the target takes.
    fn shape(self, reg_type: &impl Fn(Reg) -> Option<ValueType>) -> crate::Result<(usize, u32)> {
        match self {
            Self::Context { context, operation } => pointee(reg_type(context))
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

    /// Check a call's operands: scalar words, and slices that the call may
    /// borrow without aliasing a writable one.
    pub fn check(
        self,
        args: &[Reg],
        slices: &[ContextSlice],
        reg_type: impl Fn(Reg) -> Option<ValueType>,
    ) -> crate::Result<()> {
        let (words, borrows) = self.shape(&reg_type)?;
        if args.len() != words || slices.len() != borrows as usize {
            return Err(crate::BuildError::verify(format!(
                "{} takes {words} words and {borrows} slices, got {} and {}",
                self.name(),
                args.len(),
                slices.len()
            )));
        }
        if args
            .iter()
            .any(|&r| !matches!(reg_type(r), Some(ValueType::Scalar(_))))
        {
            return Err(crate::BuildError::verify(format!(
                "expected scalar operands for {}",
                self.name()
            )));
        }
        let mut seen = HashMap::new();
        for &ContextSlice { slice, writable } in slices {
            let kind = pointee(reg_type(slice.ptr))
                .filter(|kind| kind.context_signature().is_none())
                .filter(|_| reg_type(slice.len) == Some(ValueType::Scalar(super::ScalarType::Index)))
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
            // Distinct pointer registers come from distinct borrows, which
            // Rust already checked; the same register twice is checked here.
            if seen
                .insert(slice.ptr, writable)
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
    /// `invoke_static`), so no backend embeds a pointer constant.
    pub fn plan(
        self,
        reg_type: impl Fn(Reg) -> ValueType,
        words: usize,
        slices: &[ContextSlice],
    ) -> RustCallPlan {
        let layout = ContextArguments::packet_layout(words as u32, slices.len() as u32)
            .expect("a checked Rust call has a valid packet");
        let (entry, context, operation, header) = match self {
            Self::Context { context, operation } => (&CONTEXT_CALL, Some(context), operation, 0),
            Self::Static(_) => (&STATIC_CALL, None, 0, crate::runtime_context::STATIC_HEADER),
        };
        let mut stores = Vec::new();
        if let Self::Static(symbol) = self {
            stores.push(PacketStore {
                at: 0,
                value: PacketValue::Symbol(std::ptr::from_ref(symbol) as usize),
            });
        }
        stores.extend((0..words).map(|n| PacketStore {
            at: header + n as u32 * ContextArguments::WORD_BYTES,
            value: PacketValue::Word(n),
        }));
        for (index, &ContextSlice { slice, writable }) in slices.iter().enumerate() {
            let at = header + layout.slices_at + index as u32 * ContextArguments::SLICE_BYTES;
            let element = pointee(Some(reg_type(slice.ptr)))
                .expect("a checked slice operand")
                .element();
            let tag = ContextArguments::view_tag(element, writable);
            stores.extend([
                PacketStore {
                    at,
                    value: PacketValue::Data(slice.ptr),
                },
                PacketStore {
                    at: at + ContextArguments::LENGTH_OFFSET,
                    value: PacketValue::Length(slice.len),
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

/// What a pointer register points to.
pub(crate) fn pointee(ty: Option<ValueType>) -> Option<BufferKind> {
    match ty? {
        ValueType::Ptr(kind) => Some(kind),
        _ => None,
    }
}
