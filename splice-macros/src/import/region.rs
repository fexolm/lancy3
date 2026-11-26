//! Keep Rust control flow while rejecting unfulfilled staging requirements.
use proc_macro2::TokenStream;
use quote::ToTokens;
use std::collections::HashSet;
use syn::visit::{self, Visit};
use syn::{Expr, ExprPath, FnArg, ItemFn, Pat, Path, Signature, Visibility, parse_quote};

use crate::declaration::rebase_expression;
use crate::scalar::methods;

mod check;
pub(super) use check::Check;

pub(crate) fn publication(
    root: &Path,
    vis: &Visibility,
    signature: &Signature,
    needs_stages: bool,
) -> TokenStream {
    let staged = signature.inputs.iter().map(|arg| {
        matches!(arg, FnArg::Typed(arg) if arg.attrs.iter().any(|a| a.path().is_ident("comptime")))
    });
    emit! {
        #[doc(hidden)]
        #vis use #root::__splice_choose_kernel as __splice_region_dispatch;
        #[doc(hidden)]
        #vis const __SPLICE_REGION_STAGED: &[bool] = &[#(#staged,)*];
        #[doc(hidden)]
        #vis const __SPLICE_REGION_NEEDS_STAGES: bool = #needs_stages;
    }
}

pub(super) fn in_child_module(path: ExprPath) -> ExprPath {
    let mut expression = Expr::Path(path);
    rebase_expression(&mut expression, &syn::Generics::default());
    let Expr::Path(path) = expression else {
        unreachable!()
    };
    path
}

pub(super) struct Guards {
    pub(super) calls: Vec<Check>,
    pub(super) complete: bool,
}

pub(super) fn checks(function: &ItemFn) -> Guards {
    collect(&function.sig, &function.block, false)
}

pub(crate) fn method_policy(root: &Path, signature: &Signature, body: &syn::Block) -> TokenStream {
    let guards = collect(signature, body, true);
    if !guards.complete {
        return emit!(true);
    }
    if guards.calls.is_empty() {
        return emit!(false);
    }
    let checks = guards
        .calls
        .into_iter()
        .map(|check| check.emit(root, std::convert::identity));
    emit!({
        let ::core::result::Result::Ok(_guard) = #root::frontend::scalar::enter_expansion() else { return true; };
        let check = || -> #root::Result<()> { #(#checks)* ::core::result::Result::Ok(()) };
        check().is_err()
    })
}

fn collect(signature: &Signature, body: &syn::Block, method_policy: bool) -> Guards {
    #[derive(Default)]
    struct Locals(HashSet<String>);
    impl<'ast> Visit<'ast> for Locals {
        fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
            self.0.insert(pattern.ident.to_string());
            visit::visit_pat_ident(self, pattern);
        }
    }
    let mut locals = Locals::default();
    locals.visit_block(body);
    let mut parameters = HashSet::new();
    let mut known = HashSet::new();
    for parameter in &signature.inputs {
        match parameter {
            FnArg::Receiver(receiver) => {
                parameters.insert("self".into());
                if receiver.mutability.is_none()
                    && receiver.attrs.iter().any(|a| a.path().is_ident("comptime"))
                {
                    known.insert("self".into());
                }
            }
            FnArg::Typed(parameter) => {
                for name in super::pattern_bindings(&parameter.pat) {
                    parameters.insert(name.to_string());
                }
                if let Pat::Ident(binding) = &*parameter.pat
                    && binding.mutability.is_none()
                    && parameter
                        .attrs
                        .iter()
                        .any(|a| a.path().is_ident("comptime"))
                    && !locals.0.contains(&binding.ident.to_string())
                {
                    known.insert(binding.ident.to_string());
                }
            }
        }
    }
    let generic_names = signature
        .generics
        .params
        .iter()
        .map(|param| match param {
            syn::GenericParam::Type(param) => param.ident.to_string(),
            syn::GenericParam::Const(param) => param.ident.to_string(),
            syn::GenericParam::Lifetime(param) => param.lifetime.ident.to_string(),
        })
        .collect();
    #[derive(Default)]
    struct Types(HashSet<String>);
    impl Visit<'_> for Types {
        fn visit_type_path(&mut self, path: &syn::TypePath) {
            if path.qself.is_none() {
                self.0.insert(path.path.to_token_stream().to_string());
            }
            visit::visit_type_path(self, path);
        }
    }
    let mut types = Types::default();
    types.visit_signature(signature);
    let mut collector = Collect {
        parameters,
        known,
        locals: locals.0,
        generic_names,
        types: types.0,
        method_policy,
        checks: Vec::new(),
        unsupported: false,
    };
    collector.visit_block(body);
    Guards {
        calls: collector.checks,
        complete: !collector.unsupported,
    }
}

struct Collect {
    parameters: HashSet<String>,
    known: HashSet<String>,
    locals: HashSet<String>,
    generic_names: HashSet<String>,
    types: HashSet<String>,
    method_policy: bool,
    checks: Vec<Check>,
    unsupported: bool,
}

impl Collect {
    fn branch(&mut self, visit: impl FnOnce(&mut Self)) -> Vec<Check> {
        let start = self.checks.len();
        visit(self);
        self.checks.split_off(start)
    }

    fn conditional(&mut self, condition: &Expr, yes: Vec<Check>, no: Vec<Check>) {
        if !yes.is_empty() || !no.is_empty() {
            self.checks.push(Check::Branch {
                condition: condition.clone(),
                yes,
                no,
            });
        }
    }

    fn known(&self, expression: &Expr) -> bool {
        match expression {
            Expr::Lit(_) | Expr::Const(_) => true,
            Expr::Paren(parenthesized) => self.known(&parenthesized.expr),
            Expr::Group(group) => self.known(&group.expr),
            Expr::Path(path) if path.qself.is_none() => path
                .path
                .get_ident()
                .is_some_and(|id| self.known.contains(&id.to_string())),
            Expr::Tuple(tuple) => tuple.elems.iter().all(|value| self.known(value)),
            Expr::Array(array) => array.elems.iter().all(|value| self.known(value)),
            Expr::Unary(unary)
                if matches!(unary.op, syn::UnOp::Neg(_))
                    && matches!(&*unary.expr, Expr::Lit(lit) if matches!(lit.lit, syn::Lit::Int(_) | syn::Lit::Float(_))) =>
            {
                true
            }
            _ => false,
        }
    }

    fn method_path_in_scope(&self, path: &ExprPath) -> bool {
        if !self.method_policy || self.generic_names.is_empty() {
            return true;
        }
        struct Mentions<'a>(&'a HashSet<String>, bool);
        impl Visit<'_> for Mentions<'_> {
            fn visit_ident(&mut self, id: &syn::Ident) {
                self.1 |= self.0.contains(&id.to_string());
            }
        }
        let mut path = path.clone();
        // The metadata entries do not use the method's own turbofish.
        path.path.segments.last_mut().unwrap().arguments = syn::PathArguments::None;
        let mut mentions = Mentions(&self.generic_names, false);
        mentions.visit_expr_path(&path);
        !mentions.1
    }

    fn inherent_owner(&self, path: &ExprPath) -> bool {
        if let Some(qualifier) = &path.qself {
            return qualifier.as_token.is_none();
        }
        if path.path.segments.len() < 2 {
            return false;
        }
        let mut owner = path.path.clone();
        owner.segments.pop();
        owner.segments.pop_punct();
        let standard_primitive = owner.leading_colon.is_some()
            && owner.segments.len() == 3
            && matches!(owner.segments[0].ident.to_string().as_str(), "core" | "std")
            && owner.segments[1].ident == "primitive";
        (self.method_policy && owner.is_ident("Self"))
            || standard_primitive
            || self.types.contains(&owner.to_token_stream().to_string())
    }
}

fn unambiguous_free_path(path: &Path) -> bool {
    // A type's associated function cannot be imported through `use`.
    match path.segments.len() {
        1 => true,
        2 => {
            let owner = &path.segments.first().unwrap().ident;
            path.leading_colon.is_some() || owner == "self" || owner == "super" || owner == "crate"
        }
        _ => false,
    }
}

fn ungroup(mut expression: &Expr) -> &Expr {
    loop {
        expression = match expression {
            Expr::Paren(paren) => &paren.expr,
            Expr::Group(group) => &group.expr,
            _ => return expression,
        };
    }
}

fn literal_bool(expression: &Expr) -> Option<bool> {
    match expression {
        Expr::Lit(literal) => match &literal.lit {
            syn::Lit::Bool(value) => Some(value.value),
            _ => None,
        },
        Expr::Paren(paren) => literal_bool(&paren.expr),
        Expr::Group(group) => literal_bool(&group.expr),
        _ => None,
    }
}

impl<'ast> Visit<'ast> for Collect {
    fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
        if let Some(condition) = literal_bool(&expr.cond) {
            if condition {
                self.visit_block(&expr.then_branch);
            } else if let Some((_, otherwise)) = &expr.else_branch {
                self.visit_expr(otherwise);
            }
        } else if !self.method_policy && self.known(&expr.cond) {
            let yes = self.branch(|this| this.visit_block(&expr.then_branch));
            let no = self.branch(|this| {
                if let Some((_, otherwise)) = &expr.else_branch {
                    this.visit_expr(otherwise);
                }
            });
            self.conditional(&expr.cond, yes, no);
        } else {
            visit::visit_expr_if(self, expr);
        }
    }

    fn visit_expr_binary(&mut self, expr: &'ast syn::ExprBinary) {
        if !matches!(expr.op, syn::BinOp::And(_) | syn::BinOp::Or(_)) {
            visit::visit_expr_binary(self, expr);
        } else if let Some(left) = literal_bool(&expr.left) {
            if left == matches!(expr.op, syn::BinOp::And(_)) {
                self.visit_expr(&expr.right);
            }
        } else if !self.method_policy && self.known(&expr.left) {
            let rhs = self.branch(|this| this.visit_expr(&expr.right));
            if matches!(expr.op, syn::BinOp::And(_)) {
                self.conditional(&expr.left, rhs, Vec::new());
            } else {
                self.conditional(&expr.left, Vec::new(), rhs);
            }
        } else {
            visit::visit_expr_binary(self, expr);
        }
    }
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let Expr::Path(path) = ungroup(&call.func) {
            let trait_method = methods::trait_path(path).is_some();
            let name = path
                .path
                .get_ident()
                .filter(|_| path.qself.is_none())
                .map(ToString::to_string);
            let last = path.path.segments.last().unwrap();
            let variant = match last.ident.to_string().as_str() {
                "Ok" | "Err" => Some(parse_quote!(::core::result::Result<(), ()>)),
                "Some" => Some(parse_quote!(::core::option::Option<()>)),
                _ => None,
            };
            if name
                .as_ref()
                .is_some_and(|name| self.parameters.contains(name))
                || path.path.is_ident("Self")
            {
                // Calling an argument or a tuple constructor is ordinary Rust.
            } else if trait_method || self.inherent_owner(path) {
                if self.method_path_in_scope(path) {
                    let known = call.args.iter().map(|arg| self.known(arg)).collect();
                    self.checks.push(Check::Method {
                        path: path.clone(),
                        trait_method,
                        known,
                    });
                } else {
                    self.unsupported = true;
                }
            } else if let Some(ty) = variant {
                let mut path = path.clone();
                path.path.segments.last_mut().unwrap().arguments = syn::PathArguments::None;
                self.checks.push(Check::Constructor(path, ty));
            } else if methods::method_path(path)
                || (self.method_policy && !unambiguous_free_path(&path.path))
                || name.as_ref().is_some_and(|name| self.locals.contains(name))
            {
                self.unsupported = true;
            } else {
                let known = call.args.iter().map(|arg| self.known(arg)).collect();
                self.checks.push(Check::Function(path.clone(), known));
            }
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        // Foreign macro tokens are Rust syntax owned by that macro.
        self.unsupported |= mac.path.is_ident("comptime");
    }

    fn visit_item(&mut self, _: &'ast syn::Item) {
        // Hoisted checks resolve names from the function's outer scope.
        self.unsupported = true;
    }
}
