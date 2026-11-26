//! Staging of locals, loops and expressions. An expression is known to the
//! generator, deferred, or runtime:
//!
//! - a known value is computed by the generator while it expands the
//!   kernel (a literal, a `#[comptime]` parameter, `comptime!`);
//! - a deferred value is known only when the kernel is built: the result of
//!   a call (folded when its callee evaluates, a runtime call otherwise) or
//!   an aggregate local, whose fields the builder keeps as words;
//! - a runtime value depends on a runtime input.
//!
//! Inside a known loop, which runs as Rust, nothing is deferred: the
//! generator evaluates a call there with the mandatory evaluator.
//!
//! A local moves into the IR if its value or an assignment to it is not
//! known, including through control: a literal-initialized counter is a
//! register if its update or the number of loop iterations is not known. An
//! immutable local with a deferred initializer is itself deferred. A loop is
//! staged (lowered by the generator, at runtime or unrolled) if its
//! condition or body depends on a runtime value; a known loop runs as Rust.
//!
//! Each pass reads the declarations through the resolved ids; a pass that
//! promotes nothing ends the analysis. A promotion only adds facts, so the
//! number of passes is bounded by the number of locals and loops. The last
//! pass records the level of every expression it visits.
use super::{Body, DeclId, NodeId, Origin, Param};
use std::collections::HashSet;
use syn::{Block, Expr, Stmt};

/// What the generator knows about a value, in increasing order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Level {
    #[default]
    Known,
    Deferred,
    Runtime,
}

impl Level {
    fn known(self) -> bool {
        self == Self::Known
    }
}

#[derive(Default)]
pub(super) struct Stages {
    /// The `let`s whose declarations are IR registers.
    pub(super) locals: HashSet<NodeId>,
    /// The `let`s whose declarations are deferred.
    pub(super) deferred: HashSet<NodeId>,
    pub(super) loops: HashSet<NodeId>,
    /// The level of each expression, by id. An expression the analysis
    /// does not enter (a closure body, a `comptime!` argument) is known.
    pub(super) exprs: Vec<Level>,
}

pub(super) fn analyze(body: &Body<'_>) -> Stages {
    let mut pass = Pass {
        body,
        stages: Stages {
            exprs: vec![Level::Known; body.ids.next as usize],
            ..Stages::default()
        },
        patterns: vec![Level::Known; body.decls.len()],
        loops: Vec::new(),
        host: false,
    };
    loop {
        let count =
            |stages: &Stages| stages.locals.len() + stages.deferred.len() + stages.loops.len();
        let before = count(&pass.stages);
        pass.block(body.block, false);
        if before == count(&pass.stages) {
            return pass.stages;
        }
    }
}

struct Pass<'b, 'a> {
    body: &'b Body<'a>,
    stages: Stages,
    /// The level of each pattern-bound declaration in the current pass.
    patterns: Vec<Level>,
    /// The labels and ids of the enclosing loops, innermost last.
    loops: Vec<(Option<String>, NodeId)>,
    /// Inside a loop that is known so far: its body runs as Rust.
    host: bool,
}

impl Pass<'_, '_> {
    fn level(&self, decl: DeclId) -> Level {
        match self.body.decl(decl).origin {
            Origin::Param(Param::Known) => Level::Known,
            Origin::Param(_) => Level::Runtime,
            Origin::Let(local) if self.stages.locals.contains(&local) => Level::Runtime,
            Origin::Let(local) if self.stages.deferred.contains(&local) => Level::Deferred,
            Origin::Let(_) => Level::Known,
            Origin::Pattern => self.patterns[decl.0 as usize],
        }
    }

    fn bind(&mut self, pat: &syn::Pat, level: Level) {
        for decl in self.body.pattern_decls(pat) {
            self.patterns[decl.0 as usize] = level;
        }
    }

    /// `control` is whether the block runs under a condition that is not known.
    fn block(&mut self, block: &Block, control: bool) -> Level {
        let mut level = Level::Known;
        for stmt in &block.stmts {
            level = level.max(match stmt {
                Stmt::Local(local) => self.local(local, control),
                Stmt::Expr(expr, _) => self.expr(expr, control),
                _ => Level::Known,
            });
        }
        level
    }

    fn local(&mut self, local: &syn::Local, control: bool) -> Level {
        let id = Body::lookup(&self.body.ids.locals, local);
        let Some(init) = &local.init else {
            return Level::Known;
        };
        let mut value = self.expr(&init.expr, control);
        // An aggregate keeps its fields as words of the builder (a tuple
        // struct constructor is a call, deferred anyway).
        if !self.host && matches!(&*init.expr, Expr::Struct(_)) {
            value = value.max(Level::Deferred);
        }
        let mutable = self
            .body
            .ids
            .let_decls
            .get(&id)
            .is_some_and(|decls| decls.iter().any(|decl| self.body.mutable(*decl)));
        match value {
            Level::Runtime => {
                self.stages.locals.insert(id);
            }
            // A mutable local holds its later values in a register.
            Level::Deferred if mutable => {
                self.stages.locals.insert(id);
            }
            Level::Deferred => {
                self.stages.deferred.insert(id);
            }
            Level::Known => {}
        }
        if self.stages.locals.contains(&id) {
            Level::Runtime
        } else if self.stages.deferred.contains(&id) {
            Level::Deferred
        } else {
            Level::Known
        }
    }

    fn assigned(&mut self, left: &Expr) {
        if let Expr::Path(path) = left
            && let Some(decl) = self.body.decl_of(path)
            && let Origin::Let(local) = self.body.decl(decl).origin
        {
            self.stages.locals.insert(local);
        }
    }

    fn expr(&mut self, e: &Expr, control: bool) -> Level {
        let level = self.visit(e, control);
        self.stages.exprs[self.body.expr(e).0 as usize] = level;
        level
    }

    fn visit(&mut self, e: &Expr, control: bool) -> Level {
        match e {
            Expr::Path(path) => self
                .body
                .decl_of(path)
                .map_or(Level::Known, |decl| self.level(decl)),
            Expr::Assign(assign) => {
                let value = self.expr(&assign.right, control);
                if control || !value.known() {
                    self.assigned(&assign.left);
                }
                self.expr(&assign.left, control).max(value)
            }
            Expr::Binary(binary) => {
                let left = self.expr(&binary.left, control);
                let short_circuit = matches!(binary.op, syn::BinOp::And(_) | syn::BinOp::Or(_));
                let right = self.expr(&binary.right, control || (short_circuit && !left.known()));
                let level = left.max(right);
                if (control || !level.known()) && crate::scalar::compound_op(&binary.op).is_some() {
                    self.assigned(&binary.left);
                }
                level
            }
            Expr::Block(block) => self.block(&block.block, control),
            Expr::If(branch) => {
                let cond = self.expr(&branch.cond, control);
                let control = control || !cond.known();
                let yes = self.block(&branch.then_branch, control);
                let no = branch
                    .else_branch
                    .as_ref()
                    .map_or(Level::Known, |(_, no)| self.expr(no, control));
                cond.max(yes).max(no)
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
                    let control = control || !scrutinee.known();
                    let guard = arm
                        .guard
                        .as_ref()
                        .map_or(Level::Known, |(_, guard)| self.expr(guard, control));
                    result = result
                        .max(guard)
                        .max(self.expr(&arm.body, control || !guard.known()));
                }
                result
            }
            Expr::Loop(repeat) => self.staged_loop(
                e,
                repeat.label.as_ref(),
                &repeat.body,
                control,
                Level::Known,
            ),
            Expr::While(repeat) => {
                let cond = self.loop_operand(e, &repeat.cond, control);
                self.staged_loop(e, repeat.label.as_ref(), &repeat.body, control, cond)
            }
            Expr::ForLoop(repeat) => {
                let iterable = self.loop_operand(e, &repeat.expr, control);
                // A staged range has a symbolic induction variable. A known
                // iterator still yields Rust pattern bindings when its body
                // emits IR.
                let induction = !iterable.known()
                    || (matches!(&*repeat.expr, Expr::Range(_))
                        && self.stages.loops.contains(&self.body.expr(e)));
                self.bind(
                    &repeat.pat,
                    if induction {
                        Level::Runtime
                    } else {
                        Level::Known
                    },
                );
                self.staged_loop(e, repeat.label.as_ref(), &repeat.body, control, iterable)
            }
            Expr::Call(call) => {
                let operands = self.operands(&call.args, control);
                let constructor = matches!(&*call.func, Expr::Path(p) if crate::scalar::constructor_arity(&p.path).is_some());
                operands.max(if constructor {
                    Level::Runtime
                } else if self.host {
                    Level::Known
                } else {
                    Level::Deferred
                })
            }
            Expr::MethodCall(call) => {
                let receiver = self.expr(&call.receiver, control);
                receiver.max(self.operands(&call.args, control))
            }
            Expr::Return(ret) => {
                if let Some(value) = &ret.expr {
                    self.expr(value, control);
                }
                // A return leaves the generated function through the IR.
                Level::Runtime
            }
            Expr::Break(exit) => {
                let value = exit
                    .expr
                    .as_ref()
                    .map_or(Level::Known, |e| self.expr(e, control));
                value.max(self.leaves_staged_loop(exit.label.as_ref()))
            }
            Expr::Continue(next) => self.leaves_staged_loop(next.label.as_ref()),
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
                let base = self.expr(&index.expr, control);
                base.max(self.expr(&index.index, control))
            }
            Expr::Tuple(tuple) => self.operands(&tuple.elems, control),
            Expr::Array(array) => self.operands(&array.elems, control),
            Expr::Struct(literal) => {
                let fields = self.operands(literal.fields.iter().map(|field| &field.expr), control);
                let rest = literal
                    .rest
                    .as_ref()
                    .map_or(Level::Known, |rest| self.expr(rest, control));
                fields.max(rest)
            }
            _ => Level::Known,
        }
    }

    fn operands<'e>(&mut self, exprs: impl IntoIterator<Item = &'e Expr>, control: bool) -> Level {
        // Each operand can promote a local, even after an earlier runtime one.
        exprs
            .into_iter()
            .fold(Level::Known, |level, e| self.expr(e, control).max(level))
    }

    /// Whether a jump to the loop labeled `label` crosses into a staged
    /// loop: a Rust loop cannot jump to a label that only the generated IR
    /// has, so the loops it leaves are staged as well.
    fn leaves_staged_loop(&self, label: Option<&syn::Lifetime>) -> Level {
        let Some(label) = label else {
            return Level::Known;
        };
        let name = label.ident.to_string();
        let target = self
            .loops
            .iter()
            .rposition(|(loop_label, _)| loop_label.as_deref() == Some(&name));
        let staged = target.is_some_and(|index| {
            index + 1 < self.loops.len() && self.stages.loops.contains(&self.loops[index].1)
        });
        if staged { Level::Runtime } else { Level::Known }
    }

    /// The condition or iterable of loop `e`, which runs as Rust with the
    /// loop while the loop is known.
    fn loop_operand(&mut self, e: &Expr, operand: &Expr, control: bool) -> Level {
        let host = self.host;
        self.host |= !self.stages.loops.contains(&self.body.expr(e));
        let level = self.expr(operand, control);
        self.host = host;
        level
    }

    fn staged_loop(
        &mut self,
        e: &Expr,
        label: Option<&syn::Label>,
        body: &Block,
        control: bool,
        cond: Level,
    ) -> Level {
        let id = self.body.expr(e);
        let staged = self.stages.loops.contains(&id) || !cond.known();
        self.loops
            .push((label.map(|label| label.name.ident.to_string()), id));
        let host = self.host;
        self.host |= !staged;
        let content = self.block(body, control || staged);
        self.host = host;
        self.loops.pop();
        if staged || !content.known() {
            self.stages.loops.insert(id);
            Level::Runtime
        } else {
            Level::Known
        }
    }
}
