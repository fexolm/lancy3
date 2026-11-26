use super::{
    Machine,
    frame::{Frame, save_words},
    inst::Inst,
    isel,
};
use crate::{
    CompileError, hir,
    lir::{
        self, TargetInst,
        frame::{self, FrameRequirements, defined_registers},
        regalloc,
    },
    module::Module,
};
use regalloc2::{Ctx, Output, PRegSet};

pub(super) struct CompiledFunction {
    pub body: lir::Function<Inst>,
    pub allocation: Output,
    pub frame: Frame,
    pub result: hir::Type,
    /// Workspace for the longest path starting at this function.
    pub workspace: usize,
    /// Includes implicit writes and transitive callees on a normal return.
    pub written: PRegSet,
    pub saved: PRegSet,
}

pub(super) fn compile(
    machine: Machine,
    source: Module<hir::ssa::VerifiedBody<'_>>,
) -> Result<Module<CompiledFunction>, CompileError> {
    let entry = source.call_graph.root();
    let width = machine.cpu.bytes;
    let arg_words =
        |f: &hir::Function| frame::argument_size(f.param_types().iter().copied(), width);
    let arity = arg_words(source.functions[entry].function);
    let mut ctx = Ctx::default();
    let module = source.try_map_functions::<CompiledFunction>(|id, src, calls, compiled| {
        let mut body = isel::select(machine, src, id == entry, |callee| {
            let id = src.callee(callee);
            (id, compiled[id].written)
        })?;
        let allocation = regalloc::allocate(&mut body, &mut ctx)?;
        let (written, defined) = written_regs(&body, &allocation);
        let mut saved = defined & machine.abi.saved;
        for &callee in calls {
            saved.union_from(compiled[callee].saved);
        }
        let result = src.function.return_type();
        let mut frame = Frame::new(
            id == entry,
            arg_words(src.function),
            allocation.num_spillslots,
        );
        if id == entry && result.bytes(width) > 8 {
            let words = result.bytes(width) / 8;
            frame.spill_base += words;
            frame.slots += words;
        }
        if machine.abi.is_foreign() && body.insts().iter().any(|i| matches!(i, Inst::RustCall(..)))
        {
            return Err(foreign_calls(machine));
        }
        let needs = FrameRequirements::scan(&body);
        frame.debug_base = frame.slots;
        frame.scratch_base = frame::sum([frame.debug_base, needs.debug.div_ceil(8)])?;
        // The root saves the host's nonvolatile registers once, including
        // those written by descendants. Normal and trapping exits restore them.
        let saves = if id == entry {
            saved.into_iter().map(save_words).sum()
        } else {
            0
        };
        frame.slots = frame::sum([frame.scratch_base, needs.scratch.div_ceil(8), saves])?;
        let child_peak = calls
            .iter()
            .map(|&c| compiled[c].workspace)
            .max()
            .unwrap_or(0);
        let workspace = frame::sum([frame.slots, child_peak])?;
        Ok(CompiledFunction {
            body,
            allocation,
            frame,
            result,
            workspace,
            written,
            saved,
        })
    })?;
    let slots = module.functions[entry].workspace.max(arity);
    if slots > i32::MAX as usize / 8 {
        return Err(CompileError::Unsupported(crate::Limit::Workspace { slots }));
    }
    Ok(module)
}

/// Rust imports are `extern "C"` functions of this process.
fn foreign_calls(machine: Machine) -> CompileError {
    let convention = if machine.abi.is_win64() {
        splice_target::CallingConvention::Win64
    } else {
        splice_target::CallingConvention::SysV
    };
    CompileError::Target(splice_target::TargetError::HostCalls(convention))
}

pub(super) fn written_regs(f: &lir::Function<Inst>, allocation: &Output) -> (PRegSet, PRegSet) {
    let defined = defined_registers(f, allocation);
    let mut written = defined;
    for op in f.insts() {
        written.union_from(op.clobbers(f.machine()));
    }
    // Move resolution may also write R10, which is reserved throughout
    // the internal ABI and cannot hold a caller value across a call.
    let machine = f.machine();
    let regs = &Inst::machine_env(machine).preferred_regs_by_class;
    written.intersect_from(regs[0] | regs[1] | regs[2] | machine.abi.saved);
    (written, defined)
}
