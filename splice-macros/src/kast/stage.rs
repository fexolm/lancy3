//! Staging of locals and loops. A local moves into the IR if its value or an
//! assignment to it depends on a runtime input, including through control:
//! a literal-initialized counter is a register if its update or the number
//! of loop iterations depends on a runtime value. A loop is staged (lowered
//! by the generator, at runtime or unrolled) if its condition or body
//! depends on a runtime value.
//!
//! Each pass reads the declarations through the resolved ids; a pass that
//! promotes nothing ends the analysis. A promotion only adds facts, so the
//! number of passes is bounded by the number of locals and loops.
use super::{Body, DeclId, NodeId, Origin, Param};
use std::collections::HashSet;
use syn::{Block, Expr, Stmt};

#[derive(Default)]
pub(super) struct Stages {
    pub(super) locals: HashSet<NodeId>,
    pub(super) loops: HashSet<NodeId>,
}

pub(super) fn analyze(body: &Body<'_>) -> Stages {
    let mut pass = Pass {
        body,
        stages: Stages::default(),
        patterns: vec![false; body.decls.len()],
    };
    loop {
        let before = pass.stages.locals.len() + pass.stages.loops.len();
        pass.block(body.block, false);
        if before == pass.stages.locals.len() + pass.stages.loops.len() {
            return pass.stages;
        }
    }
}

struct Pass<'b, 'a> {
    body: &'b Body<'a>,
    stages: Stages,
    /// The stage of each pattern-bound declaration in the current pass.
    patterns: Vec<bool>,
}

impl Pass<'_, '_> {
    fn runtime(&self, decl: DeclId) -> bool {
        match self.body.decl(decl).origin {
            Origin::Param(param) => param != Param::Known,
            Origin::Let(local) => self.stages.locals.contains(&local),
            Origin::Pattern => self.patterns[decl.0 as usize],
        }
    }

    fn bind(&mut self, pat: &syn::Pat, runtime: bool) {
        for decl in self.body.pattern_decls(pat) {
            self.patterns[decl.0 as usize] = runtime;
        }
    }

    fn block(&mut self, block: &Block, control: bool) -> bool {
        let mut dynamic = false;
        for stmt in &block.stmts {
            match stmt {
                Stmt::Local(local) => {
                    let runtime = local
                        .init
                        .as_ref()
                        .is_some_and(|init| self.expr(&init.expr, control));
                    let id = Body::lookup(&self.body.ids.locals, local);
                    if runtime {
                        self.stages.locals.insert(id);
                    }
                    dynamic |= self.stages.locals.contains(&id);
                }
                Stmt::Expr(expr, _) => dynamic |= self.expr(expr, control),
                _ => {}
            }
        }
        dynamic
    }

    fn assigned(&mut self, left: &Expr) {
        if let Expr::Path(path) = left
            && let Some(decl) = self.body.decl_of(path)
            && let Origin::Let(local) = self.body.decl(decl).origin
        {
            self.stages.locals.insert(local);
        }
    }

    fn expr(&mut self, e: &Expr, control: bool) -> bool {
        match e {
            Expr::Path(path) => self.body.decl_of(path).is_some_and(|d| self.runtime(d)),
            Expr::Assign(assign) => {
                let value = self.expr(&assign.right, control);
                if control || value {
                    self.assigned(&assign.left);
                }
                self.expr(&assign.left, control) || value
            }
            Expr::Binary(binary) => {
                let left = self.expr(&binary.left, control);
                let short_circuit = matches!(binary.op, syn::BinOp::And(_) | syn::BinOp::Or(_));
                let right = self.expr(&binary.right, control || (short_circuit && left));
                if (control || left || right) && crate::scalar::compound_op(&binary.op).is_some() {
                    self.assigned(&binary.left);
                }
                left || right
            }
            Expr::Block(block) => self.block(&block.block, control),
            Expr::If(branch) => {
                let cond = self.expr(&branch.cond, control);
                let yes = self.block(&branch.then_branch, control || cond);
                let no = branch
                    .else_branch
                    .as_ref()
                    .is_some_and(|(_, no)| self.expr(no, control || cond));
                cond || yes || no
            }
            Expr::Let(binding) => {
                let value = self.expr(&binding.expr, control);
                self.bind(&binding.pat, value);
                value
            }
            Expr::Match(choice) => {
                let scrutinee = self.expr(&choice.expr, control);
                let mut result = scrutinee;
                for arm in &choice.arms {
                    self.bind(&arm.pat, scrutinee);
                    let guard = arm
                        .guard
                        .as_ref()
                        .is_some_and(|(_, guard)| self.expr(guard, control || scrutinee));
                    result |= self.expr(&arm.body, control || scrutinee || guard);
                }
                result
            }
            Expr::Loop(repeat) => self.staged_loop(e, &repeat.body, control, false),
            Expr::While(repeat) => {
                let cond = self.expr(&repeat.cond, control);
                self.staged_loop(e, &repeat.body, control, cond)
            }
            Expr::ForLoop(repeat) => {
                let iterable = self.expr(&repeat.expr, control);
                // A runtime range has a symbolic induction variable. A host
                // iterator still yields Rust pattern bindings when its body
                // emits IR.
                let induction = iterable
                    || (matches!(&*repeat.expr, Expr::Range(_))
                        && self.stages.loops.contains(&self.body.expr(e)));
                self.bind(&repeat.pat, induction);
                self.staged_loop(e, &repeat.body, control, iterable)
            }
            Expr::Call(call) => {
                self.operands(&call.args, control)
                    | matches!(&*call.func, Expr::Path(p) if crate::scalar::constructor_arity(&p.path).is_some())
            }
            Expr::MethodCall(call) => {
                self.expr(&call.receiver, control) | self.operands(&call.args, control)
            }
            Expr::Return(ret) => {
                if let Some(value) = &ret.expr {
                    self.expr(value, control);
                }
                // A return leaves the generated function through the IR.
                true
            }
            Expr::Break(exit) => exit.expr.as_ref().is_some_and(|e| self.expr(e, control)),
            Expr::Unary(unary) => self.expr(&unary.expr, control),
            Expr::Paren(inner) => self.expr(&inner.expr, control),
            Expr::Group(inner) => self.expr(&inner.expr, control),
            Expr::Range(range) => self.operands(
                range.start.iter().chain(&range.end).map(Box::as_ref),
                control,
            ),
            Expr::Cast(cast) => self.expr(&cast.expr, control),
            Expr::Field(field) => self.expr(&field.base, control),
            Expr::Reference(reference) => self.expr(&reference.expr, control),
            Expr::Index(index) => {
                self.expr(&index.expr, control) | self.expr(&index.index, control)
            }
            _ => false,
        }
    }

    fn operands<'e>(&mut self, exprs: impl IntoIterator<Item = &'e Expr>, control: bool) -> bool {
        // Each operand can promote a local, even after an earlier runtime one.
        exprs
            .into_iter()
            .fold(false, |runtime, e| self.expr(e, control) | runtime)
    }

    fn staged_loop(&mut self, e: &Expr, body: &Block, control: bool, cond: bool) -> bool {
        let id = self.body.expr(e);
        let runtime = self.stages.loops.contains(&id) || cond;
        let content = self.block(body, control || runtime);
        if runtime || content {
            self.stages.loops.insert(id);
            true
        } else {
            false
        }
    }
}
