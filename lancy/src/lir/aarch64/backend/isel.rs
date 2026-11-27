mod rules;

use super::{Result, inst::Inst};
use crate::lir::aarch64::backend::Config;
use crate::lir::frame::argument_words;
use crate::{
    hir::{Call, Type, ssa::VerifiedBody, transforms::canonical},
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

    // Without vector values, a vector register holds at most a 16-byte
    // scalar, so its spills and moves need not cover the whole register.
    let vectors = f.value_types().iter().any(|t| matches!(t, Type::Vector(_)));
    let config = Config {
        register_bytes: if vectors { config.bytes } else { 16 },
        ..config
    };
    let mut body = edges.function(config)?;
    for &b in cfg.reverse_postorder() {
        if b == f.entry() {
            let params = f.params(b);
            let types = params.iter().map(|&v| f.value_type(v));
            for (&v, (ty, word)) in params.iter().zip(argument_words(types, config.bytes)) {
                body.push(Inst::Arg(ty, word * 8, reg(v)))?;
            }
        }
        for inst in selected[b].drain(..) {
            body.push(inst)?;
        }
        edges.finish(&mut body, b, reg, || Inst::Jump)?;
    }
    Ok(body)
}
