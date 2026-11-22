use super::{
    frame::Frame,
    inst::{Inst, nonvolatile},
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
    pub result: hir::Type,
    /// Workspace for the longest path starting at this function.
    pub workspace: usize,
    /// Includes implicit writes and transitive callees on a normal return.
    pub written: PRegSet,
    pub saved: PRegSet,
}

pub(super) fn compile(
    source: Module<hir::ssa::VerifiedBody<'_>>,
    ids: &FunctionIds<hir::Function>,
) -> Result<Module<CompiledFunction>, CompileError> {
    let entry = source.call_graph.root();
    let width = super::Features::host().bytes;
    let arg_words = |f: &hir::Function| {
        f.param_types()
            .iter()
            .map(|ty| ty.bytes(width).div_ceil(8))
            .sum()
    };
    let arity = arg_words(source.functions[entry].function);
    let mut ctx = Ctx::default();
    let module = source.try_map_functions::<CompiledFunction>(|id, src, calls, compiled| {
        let mut body = isel::select(src, id == entry, |callee| {
            let id = ids[&std::ptr::from_ref(callee)];
            (id, compiled[id].written)
        })?;
        let allocation = regalloc::allocate(&mut body, &mut ctx)?;
        let (written, defined) = written_regs(&body, &allocation);
        let mut saved = defined & nonvolatile();
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
        frame.debug_base = frame.slots;
        let mut debug_words = 0;
        let mut scratch_words = 0;
        for inst in body.insts() {
            match *inst {
                Inst::DebugStore(ty, offset, _) => {
                    debug_words = debug_words.max((offset as usize + ty.bytes(width)).div_ceil(8));
                }
                Inst::CallScratch(bytes, _) => {
                    scratch_words = scratch_words.max((bytes as usize).div_ceil(8))
                }
                Inst::RustPacketCall(import, _) => {
                    let bytes = import.packet_bytes().expect("verified Rust packet");
                    scratch_words = scratch_words.max((bytes as usize).div_ceil(8));
                }
                _ => {}
            }
        }
        frame.slots += debug_words;
        frame.scratch_base = frame.slots;
        frame.slots = frame
            .slots
            .checked_add(scratch_words)
            .ok_or(CompileError::FrameOverflow)?;
        if id == entry {
            // Save the host's nonvolatile registers once, including those
            // written by descendants. Normal and trapping exits restore them.
            frame.slots += saved
                .into_iter()
                .map(super::frame::save_words)
                .sum::<usize>();
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
            result,
            workspace,
            written,
            saved,
        })
    })?;
    let slots = module.functions[entry].workspace.max(arity);
    if slots > i32::MAX as usize / 8 {
        return Err(CompileError::WorkspaceTooLarge { slots });
    }
    Ok(module)
}

pub(super) fn written_regs(f: &lir::Function<Inst>, allocation: &Output) -> (PRegSet, PRegSet) {
    let mut defined = PRegSet::empty();
    let mut written = PRegSet::empty();
    for (i, op) in f.insts().iter().enumerate() {
        let i = regalloc2::Inst::new(i);
        written.union_from(op.clobbers());
        for (operand, loc) in f.inst_operands(i).iter().zip(allocation.inst_allocs(i)) {
            if operand.kind() == OperandKind::Def
                && let Some(r) = loc.as_reg()
            {
                defined.add(r);
            }
        }
    }
    for (_, Edit::Move { to, .. }) in &allocation.edits {
        if let Some(r) = to.as_reg() {
            defined.add(r);
        }
    }
    // Move resolution may also write R10, which is reserved throughout
    // the internal ABI and cannot hold a caller value across a call.
    let regs = &Inst::machine_env().preferred_regs_by_class;
    written.union_from(defined);
    written.intersect_from(regs[0] | regs[1] | regs[2] | nonvolatile());
    (written, defined)
}
