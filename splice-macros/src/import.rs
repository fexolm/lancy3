//! Typed Rust leaf operations retain their Rust ownership and borrowing rules.
use crate::scalar::abi::{Elision, body_generics, param_name, parameter_names, used_generics};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use std::collections::HashSet;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{
    Error, Expr, FnArg, GenericParam, ItemFn, Meta, Pat, Path, Result, ReturnType, Stmt, Type,
};

macro_rules! emit {
    ($($tokens:tt)*) => { quote::quote_spanned!(proc_macro2::Span::mixed_site()=> $($tokens)*) };
}

mod captures;
mod opaque;
mod region;
use captures::{Usage, pattern_bindings};
use region::Check;
pub(crate) use region::{method_policy, publication};

pub(crate) fn scope_checks(root: &Path, function: &ItemFn) -> Option<TokenStream> {
    let guards = region::checks(function);
    guards.complete.then(|| {
        let calls = guards
            .calls
            .into_iter()
            .map(|check| check.emit(root, std::convert::identity));
        emit!(#(#calls)*)
    })
}

pub(crate) fn expand_function(root: Path, mut function: ItemFn) -> Result<TokenStream> {
    if needs_rust_abi(&function) {
        let checks = region::checks(&function).calls;
        return expand(&root, function, checks);
    }
    match crate::scalar::expand(root.clone(), &mut function) {
        Ok(native) => Ok(native),
        Err(error) => {
            let guards = region::checks(&function);
            if guards.complete {
                expand(&root, function, guards.calls).or(Err(error))
            } else {
                Err(error)
            }
        }
    }
}

fn needs_rust_abi(function: &ItemFn) -> bool {
    let operation = leaf(&function.block.stmts);
    let scalar_result = match &function.sig.output {
        ReturnType::Default => true,
        ReturnType::Type(_, ty) => scalar_type(ty, &[]),
    };
    let types: Vec<_> = function
        .sig
        .generics
        .type_params()
        .map(|t| &t.ident)
        .collect();
    // A single named call can inline a published kernel on the borrowed
    // object. Let scalar lowering resolve the declaration in scope.
    if scalar_result
        && let Some(Expr::Call(call)) = operation
        && let Expr::Path(path) = &*call.func
    {
        if crate::scalar::methods::method_path(path) {
            if let Some(Expr::Path(receiver)) = call.args.first()
                && function.sig.inputs.iter().any(|arg| matches!(arg, FnArg::Typed(a)
                    if a.attrs.is_empty() && matches!(&*a.ty, Type::Reference(r) if r.mutability.is_some() && !matches!(&*r.elem,Type::Slice(_)))
                        && matches!(&*a.pat, Pat::Ident(id) if receiver.path.is_ident(&id.ident))))
            {
                return false;
            }
        } else if crate::scalar::supports_context_call(&function.sig, &types, &call.args) {
            return false;
        }
    }
    let scalar_params = function.sig.inputs.iter().all(|arg| {
        let FnArg::Typed(arg) = arg else { return false };
        if arg
            .attrs
            .iter()
            .any(|attr| attr.path().is_ident("comptime"))
        {
            if let (Some(operation), Pat::Ident(binding)) = (operation, &*arg.pat)
                && !builtin_operand(&arg.ty, &types)
            {
                return !calls_overload(operation, &binding.ident);
            }
            return true;
        }
        arg.attrs.iter().any(|attr| attr.path().is_ident("columns")) || scalar_type(&arg.ty, &types)
    });
    if scalar_result && scalar_params {
        return false;
    }
    operation.is_some_and(|expr| {
        let mut visitor = LeafSyntax { forbidden: false };
        visitor.visit_expr(expr);
        !visitor.forbidden
    }) || (operation.is_none() && rust_region(&function.block.stmts))
}

fn builtin_operand(ty: &Type, types: &[&Ident]) -> bool {
    if scalar_type(ty, types) {
        return true;
    }
    match ty {
        Type::Reference(reference) => builtin_operand(&reference.elem, types),
        Type::Array(_) => true,
        Type::Path(path) => path.path.get_ident().is_some_and(|id| {
            matches!(
                id.to_string().as_str(),
                "char"
                    | "isize"
                    | "usize"
                    | "i8"
                    | "u8"
                    | "i16"
                    | "u16"
                    | "u32"
                    | "i64"
                    | "u64"
                    | "i128"
                    | "u128"
                    | "f32"
                    | "f64"
            )
        }),
        _ => false,
    }
}

fn calls_overload(operation: &Expr, name: &Ident) -> bool {
    struct Operators<'a> {
        name: &'a Ident,
        found: bool,
    }
    impl Operators<'_> {
        fn reads_operand(&self, mut expr: &Expr) -> bool {
            loop {
                match expr {
                    Expr::Paren(value) => expr = &value.expr,
                    Expr::Group(value) => expr = &value.expr,
                    Expr::Path(value) => return value.path.is_ident(self.name),
                    _ => return false,
                }
            }
        }
    }
    impl Visit<'_> for Operators<'_> {
        fn visit_expr(&mut self, expr: &Expr) {
            self.found |= match expr {
                Expr::Call(operation) => self.reads_operand(&operation.func),
                Expr::Binary(operation) => {
                    self.reads_operand(&operation.left) || self.reads_operand(&operation.right)
                }
                Expr::Unary(operation) if !matches!(operation.op, syn::UnOp::Deref(_)) => {
                    self.reads_operand(&operation.expr)
                }
                Expr::Index(operation) => self.reads_operand(&operation.expr),
                Expr::MethodCall(operation) => self.reads_operand(&operation.receiver),
                _ => false,
            };
            syn::visit::visit_expr(self, expr);
        }
    }
    let mut operators = Operators { name, found: false };
    operators.visit_expr(operation);
    operators.found
}

fn scalar_type(ty: &Type, types: &[&Ident]) -> bool {
    match ty {
        Type::Path(path) => path.path.segments.last().is_some_and(|s| {
            matches!(
                s.ident.to_string().as_str(),
                "i32" | "bool" | "Lane" | "Mask"
            )
        }),
        Type::Tuple(ty) => ty.elems.is_empty(),
        Type::Reference(reference) => {
            matches!(&*reference.elem, Type::Slice(slice) if crate::scalar::Elem::parse(&slice.elem, types).is_some())
        }
        _ => false,
    }
}

fn leaf(stmts: &[Stmt]) -> Option<&Expr> {
    match stmts {
        [Stmt::Expr(expr, _)] => Some(expr),
        _ => None,
    }
}

fn rust_region(stmts: &[Stmt]) -> bool {
    let mut owned = HashSet::new();
    let mut uses_owner = false;
    for stmt in stmts {
        match stmt {
            Stmt::Item(syn::Item::Use(_)) if owned.is_empty() => (),
            Stmt::Local(local) => {
                let Some(init) = &local.init else {
                    return false;
                };
                let names = region_reads(&init.expr);
                if owned.is_empty() {
                    if !matches!(
                        &*init.expr,
                        Expr::Call(_) | Expr::MethodCall(_) | Expr::Struct(_)
                    ) {
                        return false;
                    }
                } else if owned.is_disjoint(&names) {
                    return false;
                } else {
                    uses_owner = true;
                }
                owned.extend(
                    pattern_bindings(&local.pat)
                        .into_iter()
                        .map(|id| id.to_string()),
                );
            }
            Stmt::Expr(expr, _) if !owned.is_disjoint(&region_reads(expr)) => uses_owner = true,
            _ => return false,
        }
    }
    if !uses_owner {
        return false;
    }
    struct HasComptime(bool);
    impl<'ast> Visit<'ast> for HasComptime {
        fn visit_macro(&mut self, mac: &'ast syn::Macro) {
            self.0 |= mac.path.is_ident("comptime");
        }
    }
    let mut inspect = HasComptime(false);
    for stmt in stmts {
        inspect.visit_stmt(stmt);
    }
    !inspect.0
}

struct LeafSyntax {
    forbidden: bool,
}

impl<'ast> Visit<'ast> for LeafSyntax {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        self.forbidden |= matches!(
            expr,
            Expr::Block(_)
                | Expr::If(_)
                | Expr::Match(_)
                | Expr::Loop(_)
                | Expr::ForLoop(_)
                | Expr::While(_)
                | Expr::Try(_)
                | Expr::Return(_)
                | Expr::Break(_)
                | Expr::Continue(_)
                | Expr::Closure(_)
                | Expr::Async(_)
        ) || matches!(expr, Expr::Macro(mac) if mac.mac.path.is_ident("comptime"));
        syn::visit::visit_expr(self, expr);
    }
}

fn expand(root: &Path, mut function: ItemFn, checks: Vec<Check>) -> Result<TokenStream> {
    let mut sig = function.sig.clone();
    crate::declaration::rebase_signature(&mut sig);
    let names: Vec<_> = parameter_names(&sig).collect();
    // Argument-position impl Trait is an ordinary named generic parameter on
    // the factory, listed after the parameters explicitly written on the item.
    let mut implicit = Vec::<GenericParam>::new();
    for arg in &mut sig.inputs {
        let FnArg::Typed(arg) = arg else {
            unreachable!()
        };
        if let Type::ImplTrait(ty) = &*arg.ty {
            let name = Ident::new(
                &format!("__SpliceArgument{}", implicit.len()),
                Span::mixed_site(),
            );
            let bounds = &ty.bounds;
            implicit.push(syn::parse_quote!(#name: #bounds));
            *arg.ty = syn::parse_quote!(#name);
        }
    }
    sig.generics.params.extend(implicit);
    let module_vis = &function.vis;
    let vis = crate::declaration::nested_visibility(module_vis);
    let region_policy = publication(
        root,
        &vis,
        &function.sig,
        crate::scalar::uses_stage_requirements(&function.block),
    );
    let gates = function
        .attrs
        .iter()
        .filter_map(|a| crate::scalar::configuration(&a.meta).transpose())
        .collect::<Result<Vec<_>>>()?;
    let name = &sig.ident;
    let mut expr = if let Some(operation) = leaf(&function.block.stmts) {
        operation.clone()
    } else {
        Expr::Block(syn::ExprBlock {
            attrs: Vec::new(),
            label: None,
            block: (*function.block).clone(),
        })
    };
    crate::declaration::rebase_expression(&mut expr);
    let semi =
        matches!(function.block.stmts.as_slice(), [Stmt::Expr(_, Some(_))]).then(|| quote!(;));
    let captures = captures::rewrite(&mut expr, &sig, &names);
    let mut elision = Elision::new(&sig.generics);
    let mut stage = Vec::new();
    let mut call = Vec::new();
    let mut environment = Vec::new();
    let mut patterns = Vec::new();
    for ((arg, name), capture) in sig.inputs.iter().zip(&names).zip(captures) {
        let FnArg::Typed(arg) = arg else {
            unreachable!()
        };
        let mut ty = (*arg.ty).clone();
        elision.visit_type_mut(&mut ty);
        let pattern = &arg.pat;
        match arg.attrs.as_slice() {
            [attr] if matches!(&attr.meta, Meta::Path(path) if path.is_ident("comptime")) => {
                if matches!(capture, Usage::Binding) {
                    patterns.push(emit!(#[allow(unused_variables)] let #pattern = __splice_environment.#name;));
                }
                if !matches!(capture, Usage::Unused) {
                    environment.push((name, ty.clone()));
                }
                stage.push((name, ty));
            }
            [] => {
                patterns.push(emit!(let #pattern = #name;));
                call.push((name, ty));
            }
            _ => {
                return Err(Error::new_spanned(
                    arg,
                    "expected one #[comptime] marker on a specialization parameter",
                ));
            }
        }
    }
    for arg in &mut function.sig.inputs {
        let FnArg::Typed(arg) = arg else {
            unreachable!()
        };
        arg.attrs.clear();
    }
    function.attrs.push(syn::parse_quote!(#[allow(dead_code)]));
    crate::declaration::native_body(&mut function.block)?;

    let mut output = match &sig.output {
        ReturnType::Default => syn::parse_quote!(()),
        ReturnType::Type(_, ty) => (**ty).clone(),
    };
    elision.visit_result(&mut output);
    let opaque_result = opaque::contains(&output);
    let host_generics = used_generics(&elision.generics, stage.iter().map(|(_, ty)| ty));
    let (_, host_args, host_where) = host_generics.split_for_impl();
    let call_generics = used_generics(&elision.generics, call.iter().map(|(_, ty)| ty));
    let (_, call_args, call_where) = call_generics.split_for_impl();
    let environment_generics =
        used_generics(&elision.generics, environment.iter().map(|(_, ty)| ty));
    let (_, environment_args, environment_where) = environment_generics.split_for_impl();
    let borrowed_environment = environment_generics.lifetimes().next().is_some();
    let retained_types = environment.iter().filter(|_| !borrowed_environment);
    let invoke_types = call
        .iter()
        .chain(retained_types)
        .map(|(_, ty)| ty)
        .chain(std::iter::once(&output));
    let invoke_generics = if opaque_result || borrowed_environment {
        used_generics(&elision.generics, invoke_types)
    } else {
        body_generics(&elision.generics, invoke_types, &expr)
    };
    let (invoke_decls, _, invoke_where) = invoke_generics.split_for_impl();
    let constrained: HashSet<_> = call_generics
        .params
        .iter()
        .chain(
            environment_generics
                .params
                .iter()
                .filter(|_| !borrowed_environment),
        )
        .map(param_name)
        .collect();
    let needs_signature_dispatch = invoke_generics
        .params
        .iter()
        .any(|p| !constrained.contains(&param_name(p)));
    let mut signature_generics = if opaque_result {
        sig.generics.clone()
    } else {
        invoke_generics.clone()
    };
    signature_generics.params = signature_generics
        .params
        .into_iter()
        .filter(|p| {
            !matches!(p, GenericParam::Lifetime(_))
                || (!opaque_result && !constrained.contains(&param_name(p)))
        })
        .map(|mut p| {
            match &mut p {
                GenericParam::Type(t) => t.bounds = syn::parse_quote!(?Sized),
                GenericParam::Lifetime(l) => l.bounds.clear(),
                GenericParam::Const(_) => (),
            }
            p
        })
        .collect();
    signature_generics.where_clause = None;
    let (_, signature_args, _) = signature_generics.split_for_impl();
    let marker_lifetimes = signature_generics.lifetimes().map(|p| &p.lifetime);
    let type_params = signature_generics.type_params().map(|p| &p.ident);
    let marker = emit!(Signature #signature_args);
    let environment_type = if borrowed_environment {
        emit!(__SpliceEnvironment)
    } else {
        emit!(__SpliceEnvironment #environment_args)
    };
    let ir_type = if opaque_result {
        emit!(__SpliceOpaqueIr<#marker, #environment_type>)
    } else {
        emit!(#root::RustImport<#marker, #environment_type>)
    };

    let host_fields: Vec<_> = stage.iter().map(|(n, t)| emit!(#n: #t)).collect();
    let host_names: Vec<_> = stage.iter().map(|(n, _)| n).collect();
    let call_fields = call.iter().map(|(n, t)| emit!(#n: #t));
    let call_names = call.iter().map(|(n, _)| n);
    let environment_fields = environment.iter().map(|(n, t)| emit!(#n: #t));
    let environment_names = environment.iter().map(|(n, _)| n);
    let unused_host_names = stage
        .iter()
        .rev()
        .filter(|(host, _)| !environment.iter().any(|(captured, _)| captured == host))
        .map(|(name, _)| name);
    let mut create_import = emit!(#root::RustImport::new(
        ::core::stringify!(#name),
        __SpliceEnvironment { #(#environment_names,)* },
    ));
    if opaque_result {
        create_import = emit!(__SpliceOpaqueIr(#create_import));
    }
    let (environment_decl, stage_body, call_body) = if borrowed_environment {
        // Such a kernel can never be linked: it needs the caller's plan
        // address after build has ended. The environment is uninhabited.
        (
            emit!(#vis enum __SpliceEnvironment {}),
            emit! {
                #(let _ = ( #host_names, );)*
                ::core::result::Result::Err(::core::convert::Into::into("Rust import would retain a borrowed specialization argument"))
            },
            emit!(match *__splice_environment {}),
        )
    } else {
        let checks = checks
            .into_iter()
            .map(|check| check.emit(root, region::in_child_module));
        (
            emit!(#vis struct __SpliceEnvironment #environment_generics #environment_where { #(#environment_fields,)* }),
            emit! {
                #(#checks)*
                #(let _ = (#unused_host_names,);)*
                ::core::result::Result::Ok(#create_import)
            },
            emit! {
                let CallArgs { #(#call_names,)* } = __splice_arguments;
                #(#patterns)*
                #expr #semi
            },
        )
    };
    let mut ir_generics = sig.generics.clone();
    ir_generics.params = elision.generics.params.iter().filter(|p| {
        !matches!(p, GenericParam::Lifetime(l) if !host_generics.lifetimes().any(|h| h.lifetime == l.lifetime) && !sig.generics.lifetimes().any(|h| h.lifetime == l.lifetime))
    }).cloned().collect();
    ir_generics
        .make_where_clause()
        .predicates
        .push(syn::parse_quote!(#environment_type: 'static));
    let (ir_decls, _, ir_where) = ir_generics.split_for_impl();
    let mut build_generics = ir_generics.clone();
    build_generics.params.insert(
        build_generics.lifetimes().count(),
        syn::parse_quote!(__SpliceBackend: #root::Codegen),
    );
    build_generics
        .make_where_clause()
        .predicates
        .push(syn::parse_quote!(<__SpliceBackend as #root::Codegen>::Output: #root::ImportTarget));
    let (build_decls, _, build_where) = build_generics.split_for_impl();
    let args = sig.generics.params.iter().filter_map(|p| match p {
        GenericParam::Type(p) => Some(&p.ident),
        GenericParam::Const(p) => Some(&p.ident),
        _ => None,
    });
    let mut output_type = emit!(<<__SpliceBackend as #root::Codegen>::Output as #root::ImportTarget>::Linked<#marker, #environment_type>);
    let invocation = if opaque_result {
        output_type = emit!(OpaqueKernel<#output_type>);
        opaque::Interface {
            root,
            vis: &vis,
            signature: &marker,
            environment: &environment_type,
            declared: &sig.generics,
            invocation: &invoke_generics,
            arguments: &call_generics,
            output: &mut output,
            body: &call_body,
        }
        .declarations()
    } else if needs_signature_dispatch {
        emit! {
            impl #invoke_decls #root::ImportInvocation<#environment_type, CallArgs #call_args>
                for #marker #invoke_where {
                type Output = #output;
                fn invoke(__splice_environment: &#environment_type, __splice_arguments: CallArgs #call_args)
                    -> Self::Output {
                    #call_body
                }
            }
        }
    } else {
        emit! {
            impl #invoke_decls #root::ImportArgs<#environment_type> for CallArgs #call_args #invoke_where {
                type Signature = #marker;
                type Output = #output;
                fn invoke(self, __splice_environment: &#environment_type) -> Self::Output {
                    let __splice_arguments = self;
                    #call_body
                }
            }
        }
    };
    Ok(emit! {
        #function

        #(#[#gates])*
        #module_vis mod #name {
            #[allow(unused_imports)]
            use super::*;

            #[doc(hidden)]
            #vis use #root::__splice_choose_rust as __splice_dispatch;
            #region_policy
            #[doc(hidden)]
            #vis fn __splice_eval<Args, Output>(_: Args) -> #root::Result<Output> {
                ::core::result::Result::Err(::core::convert::Into::into("a Rust import cannot run during specialization"))
            }

            #vis struct BuildArgs #host_generics #host_where { #(#vis #host_fields,)* }
            #vis struct CallArgs #call_generics #call_where { #(#vis #call_fields,)* }
            #[doc(hidden)]
            #vis struct Signature #signature_generics (
                ::core::marker::PhantomData<fn(#(*const &#marker_lifetimes (),)* #(*const #type_params),*)>
            );
            #[doc(hidden)]
            #environment_decl

            #vis fn build #build_decls (args: BuildArgs #host_args)
                -> ::core::result::Result<#output_type, #root::BuildError> #build_where
            {
                let BuildArgs { #(#host_names,)* } = args;
                let ir = ir::<#(#args),*>(#(#host_names),*)?;
                <__SpliceBackend as #root::Codegen>::compile(&ir)
            }

            #vis fn ir #ir_decls (#(#host_fields),*)
                -> ::core::result::Result<#ir_type, #root::BuildError> #ir_where
            {
                #root::build_boundary(move || { #stage_body })
            }

            #invocation
        }
    })
}

fn region_reads(expr: &Expr) -> HashSet<String> {
    #[derive(Default)]
    struct Reads {
        names: HashSet<String>,
        locals: HashSet<String>,
    }
    impl<'ast> Visit<'ast> for Reads {
        fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
            if let Some(name) = path.path.get_ident()
                && !self.locals.contains(&name.to_string())
            {
                self.names.insert(name.to_string());
            }
        }
        fn visit_block(&mut self, block: &'ast syn::Block) {
            let outer = self.locals.clone();
            for stmt in &block.stmts {
                self.visit_stmt(stmt);
                if let Stmt::Local(local) = stmt {
                    self.locals.extend(
                        pattern_bindings(&local.pat)
                            .into_iter()
                            .map(|id| id.to_string()),
                    );
                }
            }
            self.locals = outer;
        }
    }
    let mut reads = Reads::default();
    reads.visit_expr(expr);
    reads.names
}
