//! Staging of mutable locals requires control dependence as well as data
//! dependence. A literal-initialized counter must move into the IR if either
//! its update or the number of loop iterations depends on a runtime input.
//!
//! An unannotated literal-initialized local also takes its integer type
//! from its uses, as in Rust: `let mut i = 0; while i < xs.len()` is a
//! `usize` (`Flow::usize_locals`).
use super::{bind_pattern, lower::Binding, lower::Scalar};
use std::collections::{HashMap, HashSet};
use syn::visit::Visit;
use syn::{BinOp, Block, Expr, Lit, Pat, Stmt};

type Names = HashMap<String, (bool, Option<usize>)>;
#[derive(Default)]
pub(super) struct Flow {
    pub locals: HashSet<usize>,
    pub loops: HashSet<usize>,
    /// Locals `let [mut] name = <unsuffixed integer literal>` that Rust
    /// infers as `usize`.
    pub usize_locals: HashSet<String>,
}

pub(super) fn key<T>(node: &T) -> usize {
    std::ptr::from_ref(node) as usize
}

pub(super) fn analyze(block: &Block, bindings: &HashMap<String, Binding>) -> Flow {
    let names = bindings
        .iter()
        .map(|(s, b)| (s.clone(), (!matches!(b, Binding::Host(_)), None)))
        .collect();
    let mut flow = Flow::default();
    loop {
        let old = flow.locals.len() + flow.loops.len();
        flow.block(block, &names, false);
        if old == flow.locals.len() + flow.loops.len() {
            flow.usize_locals = usize_locals(block, bindings);
            return flow;
        }
    }
}

impl Flow {
    fn block(&mut self, block: &Block, outer: &Names, control: bool) -> bool {
        let mut names = outer.clone();
        let mut dynamic = false;
        for s in &block.stmts {
            match s {
                Stmt::Local(l) => {
                    let runtime = l
                        .init
                        .as_ref()
                        .is_some_and(|v| self.expr(&v.expr, &names, control));
                    let id = key(l);
                    let reg = runtime || self.locals.contains(&id);
                    bind_pattern(&mut names, &l.pat, (reg, Some(id)));
                    dynamic |= reg;
                }
                Stmt::Expr(e, _) => dynamic |= self.expr(e, &names, control),
                _ => {}
            }
        }
        dynamic
    }
    fn mark_lhs(&mut self, left: &Expr, names: &Names) {
        if let Expr::Path(p) = left
            && let Some(id) = p.path.get_ident()
            && let Some((_, Some(binding))) = names.get(&id.to_string())
        {
            self.locals.insert(*binding);
        }
    }
    fn expr(&mut self, e: &Expr, names: &Names, control: bool) -> bool {
        match e {
            Expr::Path(p) => p
                .path
                .get_ident()
                .and_then(|id| names.get(&id.to_string()))
                .is_some_and(|(r, k)| *r || k.is_some_and(|k| self.locals.contains(&k))),
            Expr::Assign(a) => {
                let v = self.expr(&a.right, names, control);
                if control || v {
                    self.mark_lhs(&a.left, names)
                }
                self.expr(&a.left, names, control) || v
            }
            Expr::Binary(a) => {
                let l = self.expr(&a.left, names, control);
                use syn::BinOp::*;
                let short_circuit = matches!(a.op, And(_) | Or(_));
                let r = self.expr(&a.right, names, control || (short_circuit && l));
                if (control || r || l) && super::compound_op(&a.op).is_some() {
                    self.mark_lhs(&a.left, names);
                }
                l || r
            }
            Expr::Block(b) => self.block(&b.block, names, control),
            Expr::If(i) => {
                let c = self.expr(&i.cond, names, control);
                let a = self.block(&i.then_branch, names, control || c);
                let b = i
                    .else_branch
                    .as_ref()
                    .is_some_and(|(_, z)| self.expr(z, names, control || c));
                c || a || b
            }
            Expr::Match(m) => {
                let c = self.expr(&m.expr, names, control);
                let mut result = c;
                for arm in &m.arms {
                    let mut local = names.clone();
                    bind_pattern(&mut local, &arm.pat, (c, None));
                    let guard = arm
                        .guard
                        .as_ref()
                        .is_some_and(|(_, g)| self.expr(g, &local, control || c));
                    result |= self.expr(&arm.body, &local, control || c || guard);
                }
                result
            }
            Expr::Loop(l) => self.scan_loop(&l.body, names, control, false),
            Expr::While(w) => {
                let c = self.expr(&w.cond, names, control);
                self.scan_loop(&w.body, names, control, c)
            }
            Expr::ForLoop(f) => {
                let c = self.expr(&f.expr, names, control);
                let mut locals = names.clone();
                // Runtime ranges have a symbolic induction variable. A host
                // iterator still yields Rust pattern bindings when its body emits IR.
                let induction =
                    c || (matches!(&*f.expr, Expr::Range(_)) && self.loops.contains(&key(&f.body)));
                bind_pattern(&mut locals, &f.pat, (induction, None));
                self.scan_loop(&f.body, &locals, control, c)
            }
            Expr::Call(c) => {
                self.operands(&c.args, names, control)
                    | matches!(c.func.as_ref(), Expr::Path(p) if super::lower::simd::constructor_arity(&p.path).is_some())
            }
            Expr::MethodCall(m) => {
                self.expr(&m.receiver, names, control) | self.operands(&m.args, names, control)
            }
            Expr::Return(r) => {
                if let Some(value) = &r.expr {
                    self.expr(value, names, control);
                }
                true
            }
            Expr::Break(x) => x
                .expr
                .as_ref()
                .is_some_and(|e| self.expr(e, names, control)),
            Expr::Unary(u) => self.expr(&u.expr, names, control),
            Expr::Paren(p) => self.expr(&p.expr, names, control),
            Expr::Group(g) => self.expr(&g.expr, names, control),
            Expr::Range(r) => self.operands(
                r.start.iter().chain(&r.end).map(Box::as_ref),
                names,
                control,
            ),
            Expr::Cast(c) => self.expr(&c.expr, names, control),
            Expr::Field(f) => self.expr(&f.base, names, control),
            Expr::Reference(r) => self.expr(&r.expr, names, control),
            Expr::Index(i) => {
                self.expr(&i.expr, names, control) | self.expr(&i.index, names, control)
            }
            _ => false,
        }
    }
    fn operands<'a>(
        &mut self,
        exprs: impl IntoIterator<Item = &'a Expr>,
        names: &Names,
        control: bool,
    ) -> bool {
        // Each operand can promote a binding, even after an earlier dynamic one.
        exprs
            .into_iter()
            .fold(false, |runtime, e| self.expr(e, names, control) | runtime)
    }
    fn scan_loop(&mut self, body: &Block, names: &Names, control: bool, cond: bool) -> bool {
        let id = key(body);
        let runtime = self.loops.contains(&id) || cond;
        let content = self.block(body, names, control || runtime);
        if runtime || content {
            self.loops.insert(id);
            true
        } else {
            false
        }
    }
}

/// The candidates are bound exactly once in the body, so a name denotes one
/// local; rustc has already checked that its uses agree on one type. A use
/// beside a `usize` operand (`.len()` of a slice argument, `as usize`, a
/// `usize` argument or candidate, a `usize` literal) or as a slice index
/// makes the candidate a `usize`.
fn usize_locals(block: &Block, bindings: &HashMap<String, Binding>) -> HashSet<String> {
    #[derive(Default)]
    struct Collect {
        bound: HashMap<String, usize>,
        literal: HashSet<String>,
        uses: Vec<(Expr, Expr)>,
        indices: Vec<Expr>,
    }
    impl<'a> Visit<'a> for Collect {
        fn visit_pat_ident(&mut self, pat: &'a syn::PatIdent) {
            *self.bound.entry(pat.ident.to_string()).or_default() += 1;
            syn::visit::visit_pat_ident(self, pat);
        }
        fn visit_local(&mut self, local: &'a syn::Local) {
            if let Pat::Ident(name) = &local.pat
                && let Some(init) = &local.init
                && let Expr::Lit(literal) = &*init.expr
                && let Lit::Int(int) = &literal.lit
                && int.suffix().is_empty()
            {
                self.literal.insert(name.ident.to_string());
            }
            syn::visit::visit_local(self, local);
        }
        fn visit_expr_binary(&mut self, binary: &'a syn::ExprBinary) {
            if !matches!(
                binary.op,
                BinOp::Shl(_)
                    | BinOp::Shr(_)
                    | BinOp::ShlAssign(_)
                    | BinOp::ShrAssign(_)
                    | BinOp::And(_)
                    | BinOp::Or(_)
            ) {
                self.uses
                    .push(((*binary.left).clone(), (*binary.right).clone()));
            }
            syn::visit::visit_expr_binary(self, binary);
        }
        fn visit_expr_assign(&mut self, assign: &'a syn::ExprAssign) {
            self.uses
                .push(((*assign.left).clone(), (*assign.right).clone()));
            syn::visit::visit_expr_assign(self, assign);
        }
        fn visit_expr_index(&mut self, index: &'a syn::ExprIndex) {
            self.indices.push(index.clone().into());
            syn::visit::visit_expr_index(self, index);
        }
    }
    let mut collect = Collect::default();
    collect.visit_block(block);
    let candidates: HashSet<String> = collect
        .literal
        .iter()
        .filter(|name| collect.bound[*name] == 1 && !bindings.contains_key(*name))
        .cloned()
        .collect();
    let buffer = |e: &Expr| matches!(super::lower::path_name(e), Some(name) if matches!(bindings.get(&name), Some(Binding::Buffer(_))));
    let mut found = HashSet::new();
    loop {
        let before = found.len();
        let known = |e: &Expr| is_usize(e, bindings, &found, &buffer);
        let mut marks = Vec::new();
        for (left, right) in &collect.uses {
            if known(right) {
                marks.push(left);
            } else if known(left) {
                marks.push(right);
            }
        }
        for index in &collect.indices {
            if let Expr::Index(index) = index
                && buffer(&index.expr)
            {
                marks.push(&index.index);
            }
        }
        for e in marks {
            mark(e, &candidates, &mut found);
        }
        if found.len() == before {
            return found;
        }
    }
}

fn is_usize(
    e: &Expr,
    bindings: &HashMap<String, Binding>,
    found: &HashSet<String>,
    buffer: &dyn Fn(&Expr) -> bool,
) -> bool {
    let known = |e: &Expr| is_usize(e, bindings, found, buffer);
    match e {
        Expr::Paren(p) => known(&p.expr),
        Expr::Group(g) => known(&g.expr),
        Expr::Cast(c) => {
            matches!(&*c.ty, syn::Type::Path(t) if t.qself.is_none() && t.path.is_ident("usize"))
        }
        Expr::Lit(l) => matches!(&l.lit, Lit::Int(int) if int.suffix() == "usize"),
        Expr::MethodCall(m) => m.method == "len" && m.args.is_empty() && buffer(&m.receiver),
        Expr::Binary(b) if !matches!(b.op, BinOp::Shl(_) | BinOp::Shr(_)) && arithmetic(&b.op) => {
            known(&b.left) || known(&b.right)
        }
        _ => super::lower::path_name(e).is_some_and(|name| {
            found.contains(&name)
                || matches!(
                    bindings.get(&name),
                    Some(Binding::Reg(t, _) | Binding::Word(t)) if t.base() == Scalar::Index
                )
        }),
    }
}

fn arithmetic(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::Add(_)
            | BinOp::Sub(_)
            | BinOp::Mul(_)
            | BinOp::Div(_)
            | BinOp::Rem(_)
            | BinOp::BitAnd(_)
            | BinOp::BitOr(_)
            | BinOp::BitXor(_)
    )
}

/// Mark the candidates that `e`, a `usize` operand, consists of.
fn mark(e: &Expr, candidates: &HashSet<String>, found: &mut HashSet<String>) {
    match e {
        Expr::Paren(p) => mark(&p.expr, candidates, found),
        Expr::Group(g) => mark(&g.expr, candidates, found),
        Expr::Binary(b) if arithmetic(&b.op) => {
            mark(&b.left, candidates, found);
            mark(&b.right, candidates, found);
        }
        _ => {
            if let Some(name) = super::lower::path_name(e)
                && candidates.contains(&name)
            {
                found.insert(name);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::lower::Scalar;
    use syn::parse_quote;

    #[test]
    fn short_circuit_rhs_promotes_the_local_it_conditionally_writes() {
        let block = parse_quote!({
            let mut visited = 0;
            let _hit = x > 0 && {
                visited = 1;
                true
            };
            visited
        });
        let flow = analyze(
            &block,
            &HashMap::from([("x".to_owned(), Binding::Reg(Scalar::Int, false))]),
        );
        let Stmt::Local(visited) = &block.stmts[0] else {
            panic!()
        };
        assert!(flow.locals.contains(&key(visited)));
    }

    #[test]
    fn return_operands_propagate_conditional_writes() {
        let block = parse_quote!({
            let mut a = 0;
            return {
                if x > 0 {
                    a = 1;
                }
                a
            };
        });
        let flow = analyze(
            &block,
            &HashMap::from([("x".to_owned(), Binding::Reg(Scalar::Int, false))]),
        );
        let Stmt::Local(a) = &block.stmts[0] else {
            panic!()
        };
        assert!(flow.locals.contains(&key(a)));
    }

    /// A literal local is a `usize` through a `.len()` comparison, a slice
    /// index or another `usize` local; a shadowed name, a host counter
    /// beside an `i32` and a suffixed literal keep their own types.
    #[test]
    fn literal_locals_take_the_usize_type_of_their_uses() {
        let block = parse_quote!({
            let mut i = 0;
            let mut j = 0;
            let mut k = 0;
            let mut n = 0;
            let s = 0u32;
            while i < xs.len() {
                j = i + 1;
                n += xs[k];
                i += 1;
            }
            let t = 0;
            let t = t + x;
            (j + k) as i32 + n + t
        });
        let buffer = Binding::Buffer(crate::scalar::BufferType {
            elem: crate::scalar::Elem::I32,
            write: false,
        });
        let bindings = HashMap::from([
            ("xs".to_owned(), buffer),
            ("x".to_owned(), Binding::Reg(Scalar::Int, false)),
        ]);
        let mut found: Vec<_> = analyze(&block, &bindings)
            .usize_locals
            .into_iter()
            .collect();
        found.sort();
        assert_eq!(found, ["i", "j", "k"]);
    }
}
