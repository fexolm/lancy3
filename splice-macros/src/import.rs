//! Typed Rust leaf operations retain their Rust ownership and borrowing rules.
use crate::scalar::abi::{Elision, body_generics, param_name, parameter_names, used_generics};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use std::collections::HashSet;
use syn::visit_mut::VisitMut;
use syn::{Error, Expr, FnArg, GenericParam, ItemFn, Meta, Path, Result, ReturnType, Stmt, Type};

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

/// Publish a free function. `#[kernel]` lowers the body to Splice IR or
/// reports why it cannot; only `#[kernel(rust)]` keeps the body in Rust.
pub(crate) fn expand_function(
    root: Path,
    mut function: ItemFn,
    lowering: Lowering,
) -> Result<TokenStream> {
    match lowering {
        Lowering::Ir => crate::scalar::expand(root, &mut function).map_err(|mut error| {
            error.combine(Error::new_spanned(
                &function.sig.ident,
                "this function is not a Splice kernel; mark it #[kernel(rust)] to publish it as a Rust call",
            ));
            error
        }),
        Lowering::Rust => {
            let checks = region::checks(&function).calls;
            expand(&root, function, checks)
        }
    }
}

/// How `#[kernel]` publishes a free function or an `impl`/`trait` method.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum Lowering {
    /// Lower the body to Splice IR; an unsupported body is a compile error.
    Ir,
    /// `#[kernel(rust)]`: keep the body in Rust behind a typed import.
    Rust,
}

fn leaf(stmts: &[Stmt]) -> Option<&Expr> {
    match stmts {
        [Stmt::Expr(expr, _)] => Some(expr),
        _ => None,
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
                    let borrowed = used_generics(&elision.generics, std::iter::once(&ty))
                        .lifetimes()
                        .next()
                        .is_some();
                    if borrowed {
                        return Err(Error::new_spanned(
                            arg,
                            "a #[kernel(rust)] function keeps its specialization arguments after build, so a borrowed one must be passed at call time or owned",
                        ));
                    }
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
    let invoke_types = call
        .iter()
        .chain(&environment)
        .map(|(_, ty)| ty)
        .chain(std::iter::once(&output));
    let invoke_generics = if opaque_result {
        used_generics(&elision.generics, invoke_types)
    } else {
        body_generics(&elision.generics, invoke_types, &expr)
    };
    let (invoke_decls, _, invoke_where) = invoke_generics.split_for_impl();
    let constrained: HashSet<_> = call_generics
        .params
        .iter()
        .chain(&environment_generics.params)
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
    let environment_type = emit!(__SpliceEnvironment #environment_args);
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
    let checks = checks
        .into_iter()
        .map(|check| check.emit(root, region::in_child_module));
    let environment_decl = emit!(#vis struct __SpliceEnvironment #environment_generics #environment_where { #(#environment_fields,)* });
    let stage_body = emit! {
        #(#checks)*
        #(let _ = (#unused_host_names,);)*
        ::core::result::Result::Ok(#create_import)
    };
    let call_body = emit! {
        let CallArgs { #(#call_names,)* } = __splice_arguments;
        #(#patterns)*
        #expr #semi
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
    let args: Vec<_> = sig
        .generics
        .params
        .iter()
        .filter_map(|p| match p {
            GenericParam::Type(p) => Some(&p.ident),
            GenericParam::Const(p) => Some(&p.ident),
            _ => None,
        })
        .collect();
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
                ::core::result::Result::Err(#root::BuildError::staging("a Rust import cannot run during specialization"))
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

            /// The Rust body has no Splice code generation to configure.
            #vis fn build_with_options #build_decls (_: #root::BuildOptions, args: BuildArgs #host_args)
                -> ::core::result::Result<#output_type, #root::BuildError> #build_where
            {
                build::<__SpliceBackend, #(#args),*>(args)
            }

            #vis fn ir #ir_decls (#(#host_fields),*)
                -> ::core::result::Result<#ir_type, #root::BuildError> #ir_where
            {
                #root::build_boundary(move || { #stage_body })
            }

            /// The Rust body has no Splice code generation to configure.
            #vis fn ir_with_options #ir_decls (_: #root::BuildOptions, #(#host_fields),*)
                -> ::core::result::Result<#ir_type, #root::BuildError> #ir_where
            {
                ir::<#(#args),*>(#(#host_names),*)
            }

            #vis fn print #ir_decls (#(#host_fields),*) -> #root::Result<::std::string::String> #ir_where {
                #(let _ = #host_names;)*
                ::core::result::Result::Err(#root::BuildError::staging(::core::concat!(
                    "`", ::core::stringify!(#name), "` is a #[kernel(rust)] function without Splice source",
                )))
            }

            #invocation
        }
    })
}
