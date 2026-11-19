//! Staging of mutable locals requires control dependence as well as data
//! dependence. A literal-initialized counter must move into the IR if either
//! its update or the number of loop iterations depends on a runtime input.
use super::{bind_pattern, lower::Binding};
use std::collections::{HashMap, HashSet};
use syn::{Block, Expr, Stmt};

type Names = HashMap<String, (bool, Option<usize>)>;
#[derive(Default)]
pub(super) struct Flow {
    pub locals: HashSet<usize>,
    pub loops: HashSet<usize>,
}

pub(super) fn key<T>(node: &T) -> usize {
    std::ptr::from_ref(node) as usize
}

pub(super) fn analyze(block: &Block, bindings: &HashMap<String, Binding>) -> Flow {
    let names = bindings
        .iter()
        .map(|(s, b)| (s.clone(), (!matches!(b, Binding::Host), None)))
        .collect();
    let mut flow = Flow::default();
    loop {
        let old = flow.locals.len() + flow.loops.len();
        flow.block(block, &names, false);
        if old == flow.locals.len() + flow.loops.len() {
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
            Expr::Macro(m) if m.mac.path.is_ident("vec") => true,
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
}
