//! Track and rewrite free references to specialization arguments in Rust bodies.
use proc_macro2::Ident;
use std::collections::HashMap;
use syn::visit::{self, Visit};
use syn::visit_mut::{self, VisitMut};
use syn::{BinOp, Expr, FnArg, Pat, Signature, Stmt, Type};

#[derive(Clone, Copy)]
pub(super) enum Usage {
    Unused,
    Binding,
    Field,
}

#[derive(Clone, Copy)]
struct Argument<'a> {
    index: usize,
    field: Option<&'a Ident>,
    macro_visible: bool,
}

pub(super) fn rewrite(expr: &mut Expr, signature: &Signature, fields: &[Ident]) -> Vec<Usage> {
    let mut active = HashMap::new();
    for (index, (arg, field)) in signature.inputs.iter().zip(fields).enumerate() {
        let FnArg::Typed(arg) = arg else { continue };
        if !matches!(arg.attrs.as_slice(), [attr] if attr.path().is_ident("comptime")) {
            continue;
        }
        let macro_visible = scalar_copy(&arg.ty);
        let field = match &*arg.pat {
            Pat::Ident(binding)
                if !macro_visible
                    && binding.by_ref.is_none()
                    && binding.mutability.is_none()
                    && binding.subpat.is_none() =>
            {
                Some(field)
            }
            _ => None,
        };
        let argument = Argument {
            index,
            field,
            macro_visible,
        };
        for_binding(&arg.pat, |binding| {
            active.insert(binding.to_string(), argument);
        });
    }
    let mut captures = Captures {
        active,
        shadowed: Vec::new(),
        used: vec![Usage::Unused; fields.len()],
    };
    captures.visit_expr_mut(expr);
    captures.used
}

fn scalar_copy(ty: &Type) -> bool {
    crate::kast::types::Primitive::of(ty).is_some()
}

pub(super) fn pattern_bindings(pat: &Pat) -> Vec<Ident> {
    let mut names = Vec::new();
    for_binding(pat, |name| names.push(name.clone()));
    names
}

fn for_binding(pat: &Pat, binding: impl FnMut(&Ident)) {
    struct Bindings<F>(F);
    impl<'ast, F: FnMut(&Ident)> Visit<'ast> for Bindings<F> {
        fn visit_pat_ident(&mut self, binding: &'ast syn::PatIdent) {
            self.0(&binding.ident);
            visit::visit_pat_ident(self, binding);
        }

        fn visit_expr(&mut self, _: &'ast Expr) {}
    }
    Bindings(binding).visit_pat(pat);
}

struct Captures<'a> {
    active: HashMap<String, Argument<'a>>,
    shadowed: Vec<(String, Argument<'a>)>,
    used: Vec<Usage>,
}

impl Captures<'_> {
    fn scope(&mut self, visit: impl FnOnce(&mut Self)) {
        let outer = self.shadowed.len();
        visit(self);
        self.active.extend(self.shadowed.drain(outer..));
    }

    fn shadow(&mut self, pat: &Pat) {
        for_binding(pat, |name| {
            if let Some(binding) = self.active.remove_entry(&name.to_string()) {
                self.shadowed.push(binding);
            }
        });
    }

    fn condition(&mut self, condition: &mut Expr) {
        match condition {
            Expr::Let(binding) => {
                self.visit_expr_mut(&mut binding.expr);
                self.shadow(&binding.pat);
            }
            Expr::Binary(test) if matches!(test.op, BinOp::And(_)) => {
                self.condition(&mut test.left);
                self.condition(&mut test.right);
            }
            Expr::Group(group) => self.condition(&mut group.expr),
            Expr::Paren(paren) => self.condition(&mut paren.expr),
            expr => self.visit_expr_mut(expr),
        }
    }
}

impl VisitMut for Captures<'_> {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if self.active.is_empty() {
            return;
        }
        if let Expr::Path(path) = expr
            && let Some(name) = path.path.get_ident()
            && let Some(argument) = self.active.get(&name.to_string())
        {
            self.used[argument.index] = if let Some(field) = argument.field {
                *expr =
                    syn::parse2(emit!(__splice_environment.#field)).expect("an environment field");
                Usage::Field
            } else {
                Usage::Binding
            };
        } else {
            visit_mut::visit_expr_mut(self, expr);
        }
    }

    fn visit_block_mut(&mut self, block: &mut syn::Block) {
        self.scope(|this| {
            for stmt in &mut block.stmts {
                this.visit_stmt_mut(stmt);
                if let Stmt::Local(local) = stmt {
                    this.shadow(&local.pat);
                }
            }
        });
    }

    fn visit_expr_closure_mut(&mut self, closure: &mut syn::ExprClosure) {
        self.scope(|this| {
            for pat in &closure.inputs {
                this.shadow(pat);
            }
            this.visit_expr_mut(&mut closure.body);
        });
    }

    fn visit_arm_mut(&mut self, arm: &mut syn::Arm) {
        self.scope(|this| {
            this.shadow(&arm.pat);
            if let Some((_, guard)) = &mut arm.guard {
                this.visit_expr_mut(guard);
            }
            this.visit_expr_mut(&mut arm.body);
        });
    }

    fn visit_expr_if_mut(&mut self, test: &mut syn::ExprIf) {
        self.scope(|this| {
            this.condition(&mut test.cond);
            this.visit_block_mut(&mut test.then_branch);
        });
        if let Some((_, otherwise)) = &mut test.else_branch {
            self.visit_expr_mut(otherwise);
        }
    }

    fn visit_expr_while_mut(&mut self, walk: &mut syn::ExprWhile) {
        self.scope(|this| {
            this.condition(&mut walk.cond);
            this.visit_block_mut(&mut walk.body);
        });
    }

    fn visit_expr_for_loop_mut(&mut self, walk: &mut syn::ExprForLoop) {
        self.visit_expr_mut(&mut walk.expr);
        self.scope(|this| {
            this.shadow(&walk.pat);
            this.visit_block_mut(&mut walk.body);
        });
    }

    fn visit_item_mut(&mut self, _: &mut syn::Item) {}

    fn visit_expr_const_mut(&mut self, _: &mut syn::ExprConst) {}

    fn visit_macro_mut(&mut self, _: &mut syn::Macro) {
        // A foreign macro can read scalar locals through its token body or
        // through syntax such as a format string's implicit named arguments.
        for argument in self
            .active
            .values()
            .filter(|argument| argument.macro_visible)
        {
            self.used[argument.index] = Usage::Binding;
        }
    }

    fn visit_expr_call_mut(&mut self, call: &mut syn::ExprCall) {
        visit_mut::visit_expr_call_mut(self, call);
        if matches!(&*call.func, Expr::Field(_)) {
            let function = &call.func;
            *call.func = syn::parse_quote!((#function));
        }
    }

    fn visit_field_value_mut(&mut self, field: &mut syn::FieldValue) {
        visit_mut::visit_field_value_mut(self, field);
        if field.colon_token.is_none() && !matches!(field.expr, Expr::Path(_)) {
            field.colon_token = Some(Default::default());
        }
    }
}
