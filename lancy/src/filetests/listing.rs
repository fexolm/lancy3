use crate::hir::{
    self, Function, Op, Simd, Terminator, Value, ssa::VerifiedBody, transforms::DomTree,
};
use crate::module::{FunctionIds, Module};
use std::fmt::{Display, Write};

pub(crate) fn list(items: impl IntoIterator<Item = impl Display>) -> String {
    items
        .into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn v(value: Value) -> String {
    format!("v{}", value.index())
}

pub(super) fn hir(module: &Module<VerifiedBody>, ids: &FunctionIds<Function>) -> String {
    let mut out = String::new();
    for (id, src) in module.functions.iter() {
        let f = src.function;
        writeln!(out, "fn f{} -> {:?}", id.index(), f.return_type()).unwrap();
        let edge = |e| {
            let (b, args) = f.edge(e);
            format!("b{}({})", b.index(), list(args.iter().copied().map(v)))
        };
        for b in f.blocks() {
            let params = f
                .params(b)
                .iter()
                .map(|&x| format!("{}:{:?}", v(x), f.value_type(x)));
            writeln!(out, "b{}({})", b.index(), list(params)).unwrap();
            for i in f.instructions(b) {
                out.push_str("  ");
                if let Some(x) = f.result(i) {
                    write!(out, "{} = ", v(x)).unwrap();
                }
                let op = match *f.op(i) {
                    Op::Const(n) => format!("const {n}"),
                    Op::Copy(x) => format!("copy {}", v(x)),
                    Op::Bin(op, x, y) => format!("{op:?} {} {}", v(x), v(y)),
                    Op::Simd(Simd::LaneCount(lane)) => format!("simd.lanes {lane:?}"),
                    Op::Simd(Simd::Load(l, p)) => format!("simd.load {l:?} {}", v(p)),
                    Op::Simd(Simd::Store(p, x)) => format!("simd.store {} {}", v(p), v(x)),
                    Op::PtrOffset(p, i, s) => format!("offset {} {} {}", v(p), v(i), 1 << s as u8),
                    Op::BoundsCheck(i, n) => format!("bounds {} {}", v(i), v(n)),
                    Op::Simd(Simd::MaskedLoad {
                        address,
                        mask,
                        passthrough: x,
                    })
                    | Op::Simd(Simd::MaskedStore {
                        address,
                        mask,
                        value: x,
                    }) => {
                        let op = if matches!(f.op(i), Op::Simd(Simd::MaskedLoad { .. })) {
                            "maskedload"
                        } else {
                            "maskedstore"
                        };
                        format!(
                            "simd.{op} {} {} {} {}",
                            v(address.base),
                            address.indices.map(v).unwrap_or_else(|| "-".into()),
                            v(mask),
                            v(x)
                        )
                    }
                    Op::Literal(l, n) => format!("Literal({n}) {:?} ", l.scalar()),
                    op @ (Op::Simd(_)
                    | Op::Unary(..)
                    | Op::Ternary(..)
                    | Op::Cast(..)
                    | Op::Bitcast(..)) => {
                        let prefix = if matches!(op, Op::Simd(_)) {
                            "simd."
                        } else {
                            ""
                        };
                        let (action, _, output, args) =
                            crate::hir::operation::operation(op, |v| f.value_type(v)).unwrap();
                        format!(
                            "{prefix}{:?} {:?} {}",
                            action,
                            output,
                            list(args.into_iter().flatten().map(v))
                        )
                    }
                    Op::Load(t, p) => format!("load {t:?} {}", v(p)),
                    Op::Store(p, x) => format!("store {} {}", v(p), v(x)),
                    Op::Call(site) => {
                        let (callee, args) = f.call(site);
                        let id = ids[&std::sync::Arc::as_ptr(callee)];
                        format!(
                            "call f{}({})",
                            id.index(),
                            list(args.iter().copied().map(v))
                        )
                    }
                };
                writeln!(out, "{op}").unwrap();
            }
            let term = match *f.terminator(b) {
                Terminator::Return(x) => format!("ret {}", v(x)),
                Terminator::Jump(e) => format!("jump {}", edge(e)),
                Terminator::Branch { cond, yes, no } => {
                    format!("br {} {} | {}", v(cond), edge(yes), edge(no))
                }
            };
            writeln!(out, "  {term}").unwrap();
        }
    }
    out
}

pub(super) fn cfg(module: &Module<VerifiedBody>) -> String {
    let mut out = String::new();
    for (id, src) in module.functions.iter() {
        let cfg = &src.cfg;
        let dom = DomTree::compute(cfg);
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
