mod rules;

use super::{Result, inst::Inst};
use crate::{
    hir::{Function, transforms::ControlFlowGraph},
    lir::{
        self,
        isel::{self, Edges},
    },
    module::FunctionId,
};
use regalloc2::PRegSet;
use splice_target::Simd as Config;

pub(super) fn select(
    f: &Function,
    cfg: &ControlFlowGraph,
    config: Config,
    entry: bool,
    callee: impl Fn(&Function) -> (FunctionId, PRegSet),
) -> Result<lir::Function<Inst>> {
    let reg = |v| rules::reg(f, v);
    let edges = Edges::new(f, cfg);
    let mut selected = isel::select(f, cfg, |ctx, root, out| {
        rules::select(ctx, f, root, out, config, entry, &callee);
        Ok(())
    })?;

    let mut body = edges.function()?;
    for &b in cfg.reverse_postorder() {
        if b == f.entry() {
            let mut offset = 0;
            for &v in f.params(b) {
                let ty = f.value_type(v);
                body.push(Inst::Arg(ty, offset, reg(v)));
                offset += ty.bytes(config.bytes).next_multiple_of(8);
            }
        }
        for inst in selected[b].drain(..) {
            body.push(inst);
        }
        edges.finish(&mut body, b, reg, || Inst::Jump)?;
    }
    Ok(body)
}
