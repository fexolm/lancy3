mod masks;
mod rules;

pub(super) use rules::packed_half;

use super::{Machine, inst::Inst};
use crate::{
    CompileError,
    hir::{Call, ssa::VerifiedBody},
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
    let (f, cfg) = (source.function, &source.cfg);
    let cpu = machine.cpu;
    let masks = masks::Masks::analyze(f, cfg, cpu);
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
                body.push(Inst::Argv(addr));
                Some(addr)
            } else {
                None
            };
            let mut index = 0;
            for &arg in f.params(b) {
                let ty = f.value_type(arg);
                body.push(Inst::LoadArg {
                    dst: reg(arg),
                    argv,
                    ty,
                    index,
                });
                index += ty.bytes(cpu.bytes).div_ceil(8);
            }
        }
        for op in blocks[b].drain(..) {
            body.push(op);
        }
        edges.finish(&mut body, b, reg, || Inst::Jump)?;
    }
    Ok(body)
}
