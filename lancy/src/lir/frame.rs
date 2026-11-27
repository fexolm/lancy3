//! What both backends derive from selected code alike: the frame-local
//! regions instructions address, the word layout of call arguments, and the
//! registers a function writes. Each backend places the regions itself.
use super::{Function, TargetInst};
use crate::{CompileError, Limit, hir::Type};
use regalloc2::{Edit, Function as _, OperandKind, Output, PRegSet};

/// A frame-local region an instruction addresses, in bytes from the region
/// start.
pub(super) enum FrameUse {
    /// Debug slots up to this end.
    Debug(usize),
    /// `CallScratch` memory up to this end.
    Scratch(usize),
    /// Outgoing call arguments up to this end.
    Arguments(usize),
    /// The numeric fallback's operand area (`fallback::AREA_BYTES`).
    Fallback,
}

/// The extent of each frame-local region of one function.
#[derive(Clone, Copy, Default)]
pub(super) struct FrameRequirements {
    pub debug: usize,
    pub scratch: usize,
    pub arguments: usize,
    pub fallback: bool,
}

impl FrameRequirements {
    pub(super) fn scan<I: TargetInst>(body: &Function<I>) -> Self {
        let mut needs = Self::default();
        for inst in body.insts() {
            match inst.frame_use(body.machine()) {
                Some(FrameUse::Debug(end)) => needs.debug = needs.debug.max(end),
                Some(FrameUse::Scratch(end)) => needs.scratch = needs.scratch.max(end),
                Some(FrameUse::Arguments(end)) => needs.arguments = needs.arguments.max(end),
                Some(FrameUse::Fallback) => needs.fallback = true,
                None => {}
            }
        }
        needs
    }
}

/// Add frame sizes, failing with [`Limit::Frame`] on overflow.
pub(super) fn sum(parts: impl IntoIterator<Item = usize>) -> Result<usize, CompileError> {
    parts
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(CompileError::Unsupported(Limit::Frame))
}

/// Arguments travel in consecutive 8-byte words; each starts a new word.
/// Yields each argument's first word.
pub(super) fn argument_words(
    types: impl IntoIterator<Item = Type>,
    width: usize,
) -> impl Iterator<Item = (Type, usize)> {
    types.into_iter().scan(0, move |next, ty| {
        let at = *next;
        *next += ty.bytes(width).div_ceil(8);
        Some((ty, at))
    })
}

/// Words all arguments of `types` occupy.
pub(super) fn argument_size(types: impl IntoIterator<Item = Type>, width: usize) -> usize {
    types
        .into_iter()
        .map(|ty| ty.bytes(width).div_ceil(8))
        .sum()
}

/// Physical registers the allocated body defines, including the moves the
/// allocator inserted.
pub(super) fn defined_registers<I: TargetInst>(body: &Function<I>, allocation: &Output) -> PRegSet {
    let mut defined = PRegSet::empty();
    for index in 0..body.num_insts() {
        let inst = regalloc2::Inst::new(index);
        for (operand, alloc) in body
            .inst_operands(inst)
            .iter()
            .zip(allocation.inst_allocs(inst))
        {
            if operand.kind() == OperandKind::Def
                && let Some(reg) = alloc.as_reg()
            {
                defined.add(reg);
            }
        }
    }
    for (_, Edit::Move { to, .. }) in &allocation.edits {
        if let Some(reg) = to.as_reg() {
            defined.add(reg);
        }
    }
    defined
}
