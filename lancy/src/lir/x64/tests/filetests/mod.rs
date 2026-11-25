//! FileCheck listings for x64 selection, allocation and assembly.
mod listing;

use crate::CompileError;
use crate::filetests::{Case, Result};
use crate::hir::ssa::VerifiedBody;
use crate::lir::{
    layout, regalloc,
    x64::{
        Machine,
        compile::{compile, written_regs},
        emit::emit_compiled,
        isel,
    },
};
use crate::module::Module;

pub(crate) fn check_machine(
    case: &mut Case<'_>,
    target: &splice_target::Target,
    module: Module<VerifiedBody<'_>>,
) -> Result<()> {
    let machine = Machine::new(target);
    if case.checks.contains_key("isel") {
        let text = selected(machine, &module).map_err(|e| format!("isel: {e}"))?;
        case.check("isel", || text)?;
    }
    if ["alloc", "asm", "layout"]
        .iter()
        .any(|stage| case.checks.contains_key(stage))
    {
        let module = compile(machine, module).map_err(|e| format!("alloc: {e}"))?;
        case.check("alloc", || {
            let mut out = String::new();
            for (id, f) in module.functions.iter() {
                listing::lir(&mut out, id, &f.body, Some(&f.allocation));
            }
            out
        })?;
        case.check("layout", || {
            layout::listing(&module.functions[module.call_graph.root()].body)
        })?;
        if case.checks.contains_key("asm") {
            let code = emit_compiled(&module).map_err(|e| format!("asm: {e}"))?;
            let asm = listing::asm(&code.bytes[..code.text_len])?;
            case.check("asm", || asm)?;
        }
    }
    Ok(())
}

// ISel does not depend on the root allocation. For outlined calls it needs
// the exact transitive clobber set of each already-allocated child.
fn selected(
    machine: Machine,
    module: &Module<VerifiedBody>,
) -> std::result::Result<String, CompileError> {
    let mut out = String::new();
    let mut clobbers = crate::support::slotmap::DenseMap::new();
    let mut ctx = regalloc2::Ctx::default();
    for (id, src) in module.functions.iter() {
        let root = id == module.call_graph.root();
        let mut f = isel::select(machine, src, root, |callee| {
            let id = src.callee(callee);
            (id, clobbers[id])
        })?;
        listing::lir(&mut out, id, &f, None);
        if !root {
            let allocation = regalloc::allocate(&mut f, &mut ctx)?;
            clobbers.insert(written_regs(&f, &allocation).0);
        }
    }
    Ok(out)
}
