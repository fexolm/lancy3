use super::{
    Result,
    inst::{Inst, nonvolatile},
    isel,
};
use crate::{
    CompileError,
    hir::{Function, ssa::VerifiedBody},
    lir::{self, regalloc},
    module::{FunctionIds, Module},
};
use regalloc2::{Ctx, Edit, Function as _, OperandKind, Output, PRegSet};
use splice_target::Simd;

pub(super) const TEMP_WORD_BYTES: usize = 64;
const TEMP_BYTES: usize = 4 * TEMP_WORD_BYTES;

pub(super) struct Frame {
    pub spills: usize,
    pub outgoing: usize,
    pub temporary: usize,
    pub saves: usize,
    pub bytes: usize,
    pub debug_base: usize,
    pub scratch_base: usize,
}

pub(super) struct Compiled {
    pub body: lir::Function<Inst>,
    pub allocation: Output,
    pub frame: Frame,
    pub written: PRegSet,
    pub peak: usize,
}

pub(super) fn compile(
    source: Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
    config: Simd,
) -> Result<Module<Compiled>> {
    let root = source.call_graph.root();
    let mut ctx = Ctx::default();
    source.try_map_functions::<Compiled>(|id, source, calls, compiled| {
        let mut body = isel::select(source, config, id == root, |func| {
            let id = ids[&std::ptr::from_ref(func)];
            (id, compiled[id].written)
        })?;
        let allocation = regalloc::allocate(&mut body, &mut ctx)?;
        let mut written = written_nonvolatile(&body, &allocation);
        let mut child_peak = 0;
        for &child in calls {
            written.union_from(compiled[child].written);
            child_peak = child_peak.max(compiled[child].peak);
        }
        let mut arg_bytes = 0;
        let mut scratch_bytes = 0;
        let mut needs_temp = false;
        for inst in body.insts() {
            match *inst {
                Inst::StoreArg(ty, off, _) => {
                    arg_bytes = arg_bytes.max(off + ty.bytes(config.bytes).next_multiple_of(8));
                }
                Inst::Libcall(..) | Inst::LaneAccess { .. } => needs_temp = true,
                Inst::CallScratch(offset, bytes, _) => {
                    scratch_bytes = scratch_bytes.max(offset as usize + bytes as usize);
                }
                Inst::RustPacketCall(import, offset, _) => {
                    let bytes = import.packet_bytes().expect("verified Rust packet");
                    scratch_bytes = scratch_bytes.max(offset as usize + bytes as usize);
                }
                _ => {}
            }
        }
        let spills = if id == root {
            0
        } else {
            super::CHILD_SAVES.len() * 8
        };
        let outgoing = allocation
            .num_spillslots
            .checked_mul(8)
            .and_then(|n| n.checked_add(spills))
            .ok_or(CompileError::FrameOverflow)?;
        let temporary = outgoing
            .checked_add(arg_bytes)
            .ok_or(CompileError::FrameOverflow)?;
        let saves = temporary
            .checked_add(if needs_temp { TEMP_BYTES } else { 0 })
            .ok_or(CompileError::FrameOverflow)?;
        let debug_base = saves
            .checked_add(if id == root {
                written.into_iter().count() * 8
            } else {
                0
            })
            .ok_or(CompileError::FrameOverflow)?;
        let debug_size = body
            .insts()
            .iter()
            .filter_map(|inst| match *inst {
                Inst::DebugStore(ty, offset, _) => Some(offset as usize + ty.bytes(config.bytes)),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let scratch_base = debug_base
            .checked_add(debug_size.next_multiple_of(8))
            .ok_or(CompileError::FrameOverflow)?;
        let bytes = scratch_base
            .checked_add(scratch_bytes.next_multiple_of(8))
            .ok_or(CompileError::FrameOverflow)?;
        let peak = bytes
            .checked_add(child_peak)
            .ok_or(CompileError::FrameOverflow)?;
        Ok(Compiled {
            body,
            allocation,
            frame: Frame {
                spills,
                outgoing,
                temporary,
                saves,
                bytes,
                debug_base,
                scratch_base,
            },
            written,
            peak,
        })
    })
}

fn written_nonvolatile(body: &lir::Function<Inst>, allocation: &Output) -> PRegSet {
    let mut written = PRegSet::empty();
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
                written.add(reg);
            }
        }
    }
    for (_, Edit::Move { to, .. }) in &allocation.edits {
        if let Some(reg) = to.as_reg() {
            written.add(reg);
        }
    }
    written & nonvolatile()
}
