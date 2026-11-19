use super::{
    frame::Frame,
    inst::{Inst, NONVOLATILE},
    isel,
};
use crate::{
    CompileError, hir,
    lir::{self, TargetInst, regalloc},
    module::{FunctionIds, Module},
};
use regalloc2::{Ctx, Edit, Function as _, OperandKind, Output, PRegSet};

pub(super) struct CompiledFunction {
    pub body: lir::Function<Inst>,
    pub allocation: Output,
    pub frame: Frame,
    /// Workspace for the longest path starting at this function.
    pub workspace: usize,
    /// Includes implicit writes and transitive callees on a normal return.
    pub written: PRegSet,
}

pub(super) fn compile(
    source: Module<hir::ssa::VerifiedBody<'_>>,
    ids: &FunctionIds<hir::Function>,
) -> Result<Module<CompiledFunction>, CompileError> {
    let entry = source.call_graph.root();
    let arity = source.functions[entry].function.arity();
    let mut ctx = Ctx::default();
    let module = source.try_map_functions::<CompiledFunction>(|id, src, calls, compiled| {
        let body = isel::select(src.function, &src.cfg, id == entry, |callee| {
            let id = ids[&std::ptr::from_ref(callee)];
            (id, compiled[id].written)
        })?;
        let allocation = regalloc::allocate(&body, &mut ctx)?;
        let written = written_regs(&body, &allocation);
        let mut frame = Frame::new(id == entry, src.function.arity(), allocation.num_spillslots);
        if id == entry {
            // Save the host's nonvolatile registers once, including those
            // written by descendants. Normal and trapping exits restore them.
            frame.slots += (written & NONVOLATILE).len() as usize;
        }
        let child_peak = calls
            .iter()
            .map(|&c| compiled[c].workspace)
            .max()
            .unwrap_or(0);
        let workspace = frame
            .slots
            .checked_add(child_peak)
            .ok_or(CompileError::FrameOverflow)?;
        Ok(CompiledFunction {
            body,
            allocation,
            frame,
            workspace,
            written,
        })
    })?;
    let slots = module.functions[entry].workspace.max(arity);
    if slots > i32::MAX as usize / 8 {
        return Err(CompileError::WorkspaceTooLarge { slots });
    }
    Ok(module)
}

pub(super) fn written_regs(f: &lir::Function<Inst>, allocation: &Output) -> PRegSet {
    let mut written = PRegSet::empty();
    for (i, op) in f.insts().iter().enumerate() {
        let i = regalloc2::Inst::new(i);
        written.union_from(op.clobbers());
        for (operand, loc) in f.inst_operands(i).iter().zip(allocation.inst_allocs(i)) {
            if operand.kind() == OperandKind::Def
                && let Some(r) = loc.as_reg()
            {
                written.add(r);
            }
        }
    }
    for (_, Edit::Move { to, .. }) in &allocation.edits {
        if let Some(r) = to.as_reg() {
            written.add(r);
        }
    }
    // Move resolution may also write R10, which is reserved throughout
    // the internal ABI and cannot hold a caller value across a call.
    written.intersect_from(Inst::machine_env().preferred_regs_by_class[0] | NONVOLATILE);
    written
}
