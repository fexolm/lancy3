mod rules;

use super::{Result, inst::Inst};
use crate::lir::aarch64::backend::Config;
use crate::{
    hir::{Call, ssa::VerifiedBody, transforms::canonical},
    lir::{
        self,
        isel::{self, Edges},
    },
    module::FunctionId,
};
use regalloc2::PRegSet;

pub(super) fn select(
    source: &VerifiedBody<'_>,
    config: Config,
    entry: bool,
    callee: impl Fn(Call) -> (FunctionId, PRegSet),
) -> Result<lir::Function<Inst>> {
    let _span = tracing::debug_span!(target: "lancy::compile", "select_instructions").entered();
    // No rule selects a canonical predicate operation yet: SVE `whilelt`,
    // `cntp` and `compact` would.
    let expanded = canonical::expand(source.function, |_, _| false);
    let expanded = expanded.as_ref().map(|f| source.with_instructions(f));
    let source = expanded.as_ref().unwrap_or(source);
    let (f, cfg) = (source.function, &source.cfg);
    let reg = |v| rules::reg(f, v);
    let edges = Edges::new(source);
    let mut selected = isel::select(f, cfg, |ctx, root, out| {
        rules::select(ctx, f, root, out, config, entry, &callee);
        Ok(())
    })?;

    let mut body = edges.function(config)?;
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
