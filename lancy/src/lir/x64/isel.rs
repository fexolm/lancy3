mod rules;

pub(super) use rules::packed_half;

use super::{Features, inst::Inst};
use crate::{
    CompileError,
    hir::{Function, transforms::ControlFlowGraph},
    lir::{
        self,
        isel::{self, Edges},
    },
    module::FunctionId,
};
use regalloc2::{PRegSet, RegClass};

pub(super) fn select(
    f: &Function,
    cfg: &ControlFlowGraph,
    root: bool,
    callee_info: impl Fn(&Function) -> (FunctionId, PRegSet),
) -> Result<lir::Function<Inst>, CompileError> {
    let reg = |v| rules::reg(f, v);
    let cpu = Features::host();
    let edges = Edges::new(f, cfg);
    let mut body = edges.function()?;
    let mut blocks = isel::select(f, cfg, |ctx, root, out| {
        rules::select(ctx, f, &mut body, out, cpu, root, &callee_info)
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
