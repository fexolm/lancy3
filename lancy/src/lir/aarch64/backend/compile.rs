use super::Config;
use super::{
    Result,
    inst::{Inst, nonvolatile},
    isel,
};
use crate::{
    CompileError,
    hir::ssa::VerifiedBody,
    lir::{
        self,
        fallback::AREA_BYTES,
        frame::{self, FrameRequirements, defined_registers},
        regalloc,
    },
    module::Module,
};
use regalloc2::{Ctx, Output, PRegSet};

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
    config: Config,
) -> Result<Module<Compiled>> {
    let root = source.call_graph.root();
    let mut ctx = Ctx::default();
    source.try_map_functions::<Compiled>(|id, source, calls, compiled| {
        let mut body = isel::select(source, config, id == root, |func| {
            let id = source.callee(func);
            (id, compiled[id].written)
        })?;
        let allocation = regalloc::allocate(&mut body, &mut ctx)?;
        let mut written = written_nonvolatile(&body, &allocation);
        let mut child_peak = 0;
        for &child in calls {
            written.union_from(compiled[child].written);
            child_peak = child_peak.max(compiled[child].peak);
        }
        let needs = FrameRequirements::scan(&body);
        let spills = if id == root {
            0
        } else {
            super::CHILD_SAVES.len() * 8
        };
        let spill_bytes = allocation
            .num_spillslots
            .checked_mul(8)
            .ok_or(CompileError::Unsupported(crate::Limit::Frame))?;
        let outgoing = frame::sum([spills, spill_bytes])?;
        let temporary = frame::sum([outgoing, needs.arguments])?;
        let fallback = if needs.fallback { AREA_BYTES } else { 0 };
        let saves = frame::sum([temporary, fallback])?;
        let root_saves = if id == root {
            written.into_iter().count() * 8
        } else {
            0
        };
        let debug_base = frame::sum([saves, root_saves])?;
        let scratch_base = frame::sum([debug_base, needs.debug.next_multiple_of(8)])?;
        let bytes = frame::sum([scratch_base, needs.scratch.next_multiple_of(8)])?;
        let peak = frame::sum([bytes, child_peak])?;
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
    defined_registers(body, allocation) & nonvolatile()
}
