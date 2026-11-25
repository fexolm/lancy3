use crate::hir::{self, ssa::VerifiedBody};
use crate::module::Module;
use std::fmt::{Display, Write};

pub(crate) fn list(items: impl IntoIterator<Item = impl Display>) -> String {
    items
        .into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The functions of `module` in its order, each named by its ID.
pub(super) fn hir(module: &Module<VerifiedBody>) -> String {
    let mut out = String::new();
    for (id, src) in module.functions.iter() {
        let callee = |call| format!("f{}", src.callee(call).index());
        hir::text::write_function(&mut out, &format!("f{}", id.index()), src.function, &callee)
            .unwrap();
    }
    out
}

pub(super) fn cfg(module: &Module<VerifiedBody>) -> String {
    let mut out = String::new();
    for (id, src) in module.functions.iter() {
        let cfg = &src.cfg;
        let dom = &src.dom;
        let block = |b: hir::Block| format!("b{}", b.index());
        writeln!(out, "fn f{}", id.index()).unwrap();
        for &b in cfg.reverse_postorder() {
            let preds = cfg.predecessors(b).iter().copied().map(block);
            let successors = cfg.successors(b).map(block);
            let dominators = dom
                .preorder()
                .iter()
                .copied()
                .filter(|&d| dom.dominates(d, b))
                .map(block);
            writeln!(
                out,
                "{} preds=[{}] succs=[{}] dom=[{}]",
                block(b),
                list(preds),
                list(successors),
                list(dominators)
            )
            .unwrap();
        }
    }
    out
}
