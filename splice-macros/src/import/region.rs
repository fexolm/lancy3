//! Keep Rust control flow while rejecting unfulfilled staging requirements.
use proc_macro2::TokenStream;
use std::collections::HashSet;
use syn::visit::{self, Visit};
use syn::{Expr, ExprPath, FnArg, ItemFn, Pat, Path, Signature, Type, Visibility, parse_quote};

use crate::declaration::rebase_expression;
use crate::scalar::{methods, select_free};

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

pub(crate) enum Check {
    Function(ExprPath, Vec<bool>),
    Constructor(ExprPath, Type),
    Branch {
        condition: Expr,
        yes: Vec<Check>,
        no: Vec<Check>,
    },
}

impl Check {
    pub(super) fn emit(self, root: &Path, resolve: fn(ExprPath) -> ExprPath) -> TokenStream {
        match self {
            Self::Branch { condition, yes, no } => {
                let yes = yes.into_iter().map(|check| check.emit(root, resolve));
                let no = no.into_iter().map(|check| check.emit(root, resolve));
                emit!(if #condition { #(#yes)* } else { #(#no)* })
            }
            Self::Function(path, known) => {
                let path = resolve(path);
                let mut module = path.path.clone();
                module.segments.last_mut().unwrap().arguments = syn::PathArguments::None;
                let selector = quote::format_ident!("__splice_region_dispatch");
                let staged_call = emit!({
                    let known: &[bool] = &[#(#known,)*];
                    #module::__SPLICE_REGION_NEEDS_STAGES ||
                        #module::__SPLICE_REGION_STAGED.iter().zip(known).any(|(&staged, &known)| staged && !known)
                });
                let guard = select_free(root, &path, &selector, staged_call, emit!(false));
                let message =
                    "a published kernel call cannot be specialized inside this Rust region";
                emit!(if #guard { return Err(#message.into()); })
            }
            Self::Constructor(path, ty) => {
                let path = resolve(path);
                // Constructors also occupy the type namespace. Verify the
                // variant through a pattern if its name shadows the prelude.
                emit!(let _ = |value: #ty| ::core::matches!(value, #path(..));)
            }
        }
    }
}

pub(super) fn in_child_module(path: ExprPath) -> ExprPath {
    let mut expression = Expr::Path(path);
    rebase_expression(&mut expression);
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
    #[derive(Default)]
    struct Locals(HashSet<String>);
    impl<'ast> Visit<'ast> for Locals {
        fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
            self.0.insert(pattern.ident.to_string());
            visit::visit_pat_ident(self, pattern);
        }
    }
    let mut locals = Locals::default();
    locals.visit_block(&function.block);
    let mut parameters = HashSet::new();
    let mut known = HashSet::new();
    for parameter in &function.sig.inputs {
        let FnArg::Typed(parameter) = parameter else {
            continue;
        };
        for name in super::pattern_bindings(&parameter.pat) {
            parameters.insert(name.to_string());
        }
        if let Pat::Ident(binding) = &*parameter.pat
            && binding.mutability.is_none()
            && matches!(&*parameter.ty, Type::Path(path) if path.path.is_ident("i32") || path.path.is_ident("bool"))
            && parameter
                .attrs
                .iter()
                .any(|a| a.path().is_ident("comptime"))
            && !locals.0.contains(&binding.ident.to_string())
        {
            known.insert(binding.ident.to_string());
        }
    }
    let mut collector = Collect {
        parameters,
        known,
        locals: locals.0,
        checks: Vec::new(),
        unsupported: false,
    };
    collector.visit_block(&function.block);
    Guards {
        calls: collector.checks,
        complete: !collector.unsupported,
    }
}

struct Collect {
    parameters: HashSet<String>,
    known: HashSet<String>,
    locals: HashSet<String>,
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
            Expr::Lit(_) => true,
            Expr::Paren(parenthesized) => self.known(&parenthesized.expr),
            Expr::Group(group) => self.known(&group.expr),
            Expr::Path(path) => path
                .path
                .get_ident()
                .is_some_and(|id| self.known.contains(&id.to_string())),
            _ => false,
        }
    }
}

impl<'ast> Visit<'ast> for Collect {
    fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
        if self.known(&expr.cond) {
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
        if matches!(expr.op, syn::BinOp::And(_) | syn::BinOp::Or(_)) && self.known(&expr.left) {
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
        let mut callee = &*call.func;
        while let Expr::Paren(group) = callee {
            callee = &group.expr;
        }
        if let Expr::Path(path) = callee {
            let name = path.path.get_ident().map(ToString::to_string);
            let last = path.path.segments.last().unwrap();
            let variant = match last.ident.to_string().as_str() {
                "Ok" | "Err" => Some(parse_quote!(::core::result::Result<(), ()>)),
                "Some" => Some(parse_quote!(::core::option::Option<()>)),
                _ => None,
            };
            if name
                .as_ref()
                .is_some_and(|name| self.parameters.contains(name))
            {
                // Function pointers and closures have ordinary Rust calls.
            } else if let Some(ty) = variant {
                let mut path = path.clone();
                path.path.segments.last_mut().unwrap().arguments = syn::PathArguments::None;
                self.checks.push(Check::Constructor(path, ty));
            } else if methods::method_path(path)
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
