//! Name resolution: one walk that numbers the nodes of a body and links each
//! use of a local or parameter to its declaration through a scope tree.
//! Blocks, closures, `match` arms, `for` bodies and the conditions of
//! `if`/`while` (for `let` chains) open scopes; `let` binds after its
//! initializer, so `let x = x + 1` reads the outer `x`. Items declared in the
//! body are opaque Rust and are not entered.
use super::{Decl, DeclId, Ids, NodeId, Origin, Param, address};
use proc_macro2::Ident;
use std::collections::HashMap;
use syn::visit::{self, Visit};
use syn::{Block, Error, Expr, Result};

pub(super) type Resolved = (Ids, Vec<Decl>, HashMap<NodeId, Box<Expr>>);

pub(super) fn resolve(block: &Block, params: &[(Ident, Param)]) -> Result<Resolved> {
    let mut walk = Resolve::default();
    for (name, param) in params {
        walk.declare(name, Origin::Param(*param));
    }
    walk.visit_block(block);
    match walk.error {
        Some(error) => Err(error),
        None => Ok((walk.ids, walk.decls, walk.comptime)),
    }
}

#[derive(Default)]
struct Resolve {
    ids: Ids,
    decls: Vec<Decl>,
    /// Visible names, innermost last; `scopes` marks where each scope starts.
    names: Vec<(String, DeclId)>,
    scopes: Vec<usize>,
    /// The `let` whose pattern is being bound.
    binding: Option<NodeId>,
    /// Inside a pattern whose declarations are already being recorded.
    nested: bool,
    comptime: HashMap<NodeId, Box<Expr>>,
    error: Option<Error>,
}

impl Resolve {
    fn declare(&mut self, name: &Ident, origin: Origin) {
        let id = DeclId(self.decls.len() as u32);
        self.decls.push(Decl {
            name: name.clone(),
            origin,
        });
        if let Origin::Let(local) = origin {
            self.ids.let_decls.entry(local).or_default().push(id);
        }
        self.names.push((name.to_string(), id));
    }

    fn scoped(&mut self, visit: impl FnOnce(&mut Self)) {
        self.scopes.push(self.names.len());
        visit(self);
        let start = self.scopes.pop().expect("balanced scopes");
        self.names.truncate(start);
    }
}

impl<'ast> Visit<'ast> for Resolve {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        let id = self.ids.fresh();
        self.ids.exprs.insert(address(expr), id);
        visit::visit_expr(self, expr);
    }

    fn visit_block(&mut self, block: &'ast Block) {
        self.scoped(|this| visit::visit_block(this, block));
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        let id = self.ids.fresh();
        self.ids.locals.insert(address(local), id);
        if let Some(init) = &local.init {
            self.visit_expr(&init.expr);
            if let Some((_, otherwise)) = &init.diverge {
                self.visit_expr(otherwise);
            }
        }
        let outer = self.binding.replace(id);
        self.visit_pat(&local.pat);
        self.binding = outer;
    }

    /// Bind a pattern of a `match` arm, `for`, `let` condition or closure.
    fn visit_pat(&mut self, pat: &'ast syn::Pat) {
        if self.binding.is_some() || self.nested {
            return visit::visit_pat(self, pat);
        }
        let start = self.decls.len() as u32;
        self.nested = true;
        visit::visit_pat(self, pat);
        self.nested = false;
        self.ids
            .patterns
            .insert(address(pat), start..self.decls.len() as u32);
    }

    fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
        let origin = self.binding.map_or(Origin::Pattern, Origin::Let);
        self.declare(&pat.ident, origin);
        if let Some((_, sub)) = &pat.subpat {
            self.visit_pat(sub);
        }
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.qself.is_none()
            && let Some(name) = path.path.get_ident()
        {
            let name = name.to_string();
            if let Some((_, decl)) = self.names.iter().rev().find(|(n, _)| *n == name) {
                self.ids.uses.insert(address(path), *decl);
            }
        }
        visit::visit_expr_path(self, path);
    }

    fn visit_expr_let(&mut self, expr: &'ast syn::ExprLet) {
        // `let` in a condition binds into the enclosing condition scope.
        self.visit_expr(&expr.expr);
        self.visit_pat(&expr.pat);
    }

    fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
        self.scoped(|this| {
            this.visit_expr(&expr.cond);
            this.visit_block(&expr.then_branch);
        });
        if let Some((_, otherwise)) = &expr.else_branch {
            self.visit_expr(otherwise);
        }
    }

    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        self.scoped(|this| {
            this.visit_expr(&expr.cond);
            this.visit_block(&expr.body);
        });
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        self.visit_expr(&expr.expr);
        self.scoped(|this| {
            this.visit_pat(&expr.pat);
            this.visit_block(&expr.body);
        });
    }

    fn visit_arm(&mut self, arm: &'ast syn::Arm) {
        self.scoped(|this| visit::visit_arm(this, arm));
    }

    fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
        self.scoped(|this| visit::visit_expr_closure(this, closure));
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if !mac.path.is_ident("comptime") {
            // Other macros own their tokens.
            return;
        }
        let id = self.ids.fresh();
        self.ids.macros.insert(address(mac), id);
        match mac.parse_body::<Expr>() {
            Ok(expr) => {
                let expr = Box::new(expr);
                self.visit_expr(&expr);
                self.comptime.insert(id, expr);
            }
            Err(error) => {
                self.error.get_or_insert(error);
            }
        }
    }

    fn visit_item(&mut self, _: &'ast syn::Item) {}
}
