mod masks;
mod rules;

pub(super) use rules::packed_half;

use super::{Machine, inst::Inst};
use crate::lir::frame::argument_words;
use crate::{
    CompileError,
    hir::{Call, ssa::VerifiedBody, transforms::canonical},
    lir::{
        self,
        isel::{self, Edges},
    },
    module::FunctionId,
};
use regalloc2::{PRegSet, RegClass};

pub(super) fn select(
    machine: Machine,
    source: &VerifiedBody<'_>,
    root: bool,
    callee_info: impl Fn(Call) -> (FunctionId, PRegSet),
) -> Result<lir::Function<Inst>, CompileError> {
    let _span = tracing::debug_span!(target: "lancy::compile", "select_instructions").entered();
    let cpu = machine.cpu;
    let masks = masks::Masks::analyze(source.function, &source.cfg, cpu);
    let expanded = canonical::expand(source.function, |f, inst| {
        rules::native_canonical(f, inst, cpu, &masks)
    });
    let expanded = expanded.as_ref().map(|f| source.with_instructions(f));
    let (source, masks) = match &expanded {
        // The definitions keep each predicate's register class.
        Some(body) => (body, masks::Masks::analyze(body.function, &body.cfg, cpu)),
        None => (source, masks),
    };
    let (f, cfg) = (source.function, &source.cfg);
    let reg = |v| masks.reg(f, v);
    let edges = Edges::new(source);
    let mut body = edges.function(machine)?;
    let mut blocks = isel::select(f, cfg, |ctx, root, out| {
        rules::select(ctx, f, &mut body, out, (cpu, &masks), root, &callee_info)
    })?;
    for &b in cfg.reverse_postorder() {
        let entry = b == f.entry();
        if entry {
            let argv = if root && f.arity() != 0 {
                let addr = body.temp(RegClass::Int)?;
                body.push(Inst::Argv(addr))?;
                Some(addr)
            } else {
                None
            };
            let params = f.params(b);
            let types = params.iter().map(|&v| f.value_type(v));
            for (&arg, (ty, index)) in params.iter().zip(argument_words(types, cpu.bytes)) {
                body.push(Inst::LoadArg {
                    dst: reg(arg),
                    argv,
                    ty,
                    index,
                })?;
            }
        }
        for op in blocks[b].drain(..) {
            body.push(op)?;
        }
        edges.finish(&mut body, b, reg, || Inst::Jump)?;
    }
    Ok(body)
}
