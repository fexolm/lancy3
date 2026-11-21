//! FileCheck listings for x64 selection, allocation and assembly.
mod listing;

use crate::CompileError;
use crate::filetests::{Case, Result};
use crate::hir::{Function, ssa::VerifiedBody};
use crate::lir::{
    regalloc,
    x64::{
        compile::{compile, written_regs},
        emit::emit_compiled,
        isel,
    },
};
use crate::module::{FunctionIds, Module};

pub(crate) fn check_machine(
    case: &mut Case<'_>,
    module: Module<VerifiedBody<'_>>,
    ids: &FunctionIds<Function>,
) -> Result<()> {
    if case.checks.contains_key("isel") {
        let text = selected(&module, ids).map_err(|e| format!("isel: {e}"))?;
        case.check("isel", || text)?;
    }
    if case.checks.contains_key("alloc") || case.checks.contains_key("asm") {
        let module = compile(module, ids).map_err(|e| format!("alloc: {e}"))?;
        case.check("alloc", || {
            let mut out = String::new();
            for (id, f) in module.functions.iter() {
                listing::lir(&mut out, id, &f.body, Some(&f.allocation));
            }
            out
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
    module: &Module<VerifiedBody>,
    ids: &FunctionIds<Function>,
) -> std::result::Result<String, CompileError> {
    let mut out = String::new();
    let mut clobbers = crate::support::slotmap::DenseMap::new();
    let mut ctx = regalloc2::Ctx::default();
    for (id, src) in module.functions.iter() {
        let root = id == module.call_graph.root();
        let mut f = isel::select(src.function, &src.cfg, root, |callee| {
            let id = ids[&std::ptr::from_ref(callee)];
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
