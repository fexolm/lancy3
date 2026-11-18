//! Expand an explicitly column-oriented kernel into a projection builder.
mod lower;

use crate::scalar_type;
use lower::{Lane, Lower};
use proc_macro2::{Ident, Span, TokenStream as Tokens};
use quote::quote;
use syn::{Error, FnArg, GenericParam, ItemFn, Meta, Pat, Path, Result, ReturnType};

pub(super) fn expand(root: Path, function: ItemFn) -> Result<Tokens> {
    let ItemFn {
        attrs,
        vis,
        sig,
        block,
    } = function;
    if sig.asyncness.is_some()
        || sig.constness.is_some()
        || sig.unsafety.is_some()
        || sig.abi.is_some()
        || sig.variadic.is_some()
    {
        return Err(Error::new_spanned(
            &sig,
            "a kernel must be an ordinary Rust function",
        ));
    }
    if !matches!(&sig.output, ReturnType::Type(_, ty) if scalar_type(ty, "i32")) {
        return Err(Error::new_spanned(&sig, "a kernel returns one i32 per row"));
    }

    let name = &sig.ident;
    let generics = &sig.generics;
    let where_clause = &generics.where_clause;
    let ctx = Ident::new("__lancy_context", Span::mixed_site());
    let helper = Ident::new("__lancy_expand", Span::mixed_site());
    let mut host_args = Vec::new();
    let mut helper_args = Vec::new();
    let mut supplied = Vec::new();
    let mut bindings = std::collections::HashMap::new();
    let mut count: usize = 0;
    let mut host_params = Vec::new();
    for arg in &sig.inputs {
        let FnArg::Typed(arg) = arg else {
            return Err(Error::new_spanned(
                arg,
                "kernel parameters must be named arguments",
            ));
        };
        let Pat::Ident(pat) = &*arg.pat else {
            return Err(Error::new_spanned(&arg.pat, "use a simple parameter name"));
        };
        if pat.by_ref.is_some() || pat.subpat.is_some() || pat.mutability.is_some() {
            return Err(Error::new_spanned(pat, "use an immutable parameter name"));
        }
        let ident = &pat.ident;
        let ty = &arg.ty;
        let host = match arg.attrs.as_slice() {
            [] => false,
            [attr] if matches!(&attr.meta, Meta::Path(p) if p.is_ident("comptime")) => true,
            _ => {
                return Err(Error::new_spanned(
                    arg,
                    "expected a single #[comptime] or an unannotated i32 column",
                ));
            }
        };
        host_params.push(host);
        bindings.insert(ident.to_string(), (!host).then_some(Lane::Int));
        if host {
            host_args.push(quote!(#ident: #ty));
            helper_args.push(quote!(#ident: #ty));
            supplied.push(quote!(#ident));
        } else {
            if !scalar_type(ty, "i32") {
                return Err(Error::new_spanned(
                    ty,
                    "runtime columns have type i32; mark host arguments #[comptime]",
                ));
            }
            let index = u16::try_from(count)
                .map_err(|_| Error::new_spanned(ident, "too many kernel columns"))?;
            count += 1;
            helper_args.push(quote!(#ident: #root::frontend::batch::Operand));
            supplied.push(quote!(#root::frontend::batch::Operand::Input(#index)));
        }
    }
    let generic_args: Vec<_> = generics
        .params
        .iter()
        .filter_map(|p| match p {
            GenericParam::Type(t) => Some(&t.ident),
            GenericParam::Const(c) => Some(&c.ident),
            GenericParam::Lifetime(_) => None,
        })
        .collect();
    let turbofish = (!generic_args.is_empty()).then(|| quote!(::<#(#generic_args),*>));
    let expand_fn = quote!(#helper #turbofish);
    let lower = Lower::new(&root, name, &ctx, &expand_fn, &host_params, bindings);
    let body = lower.block(&block)?.operand(Lane::Int, &root, &block)?;
    Ok(quote! {
        #(#attrs)* #vis fn #name #generics (#(#host_args),*)
            -> ::core::result::Result<#root::hir::Program, #root::hir::BuildError> #where_clause {
            #[allow(clippy::too_many_arguments)]
            fn #helper #generics (#(#helper_args,)* #ctx: &mut #root::frontend::batch::Context)
                -> #root::frontend::batch::Operand #where_clause {
                #body
            }
            let mut #ctx = #root::frontend::batch::Context::new(#count);
            let result = #expand_fn(#(#supplied,)* &mut #ctx);
            #ctx.finish(result)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn rejects(function: ItemFn, message: &str) {
        let error = expand(parse_quote!(::lancy), function).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }

    #[test]
    fn runtime_values_cannot_control_a_host_match() {
        rejects(
            parse_quote! {
                fn f(x: i32) -> i32 { let k = x + 1; match k { 0 => 1, _ => x } }
            },
            "match requires a #[comptime] value",
        );
        rejects(
            parse_quote! {
                fn f(#[comptime] k: Option<i32>, x: i32) -> i32 {
                    match k { Some(v) if x > v => v, _ => x }
                }
            },
            "expected a comptime Rust value",
        );
    }

    #[test]
    fn recursive_host_parameters_cannot_receive_a_runtime_operand() {
        rejects(
            parse_quote! {
                fn f(#[comptime] k: i32, x: i32) -> i32 { f(x + 1, x) }
            },
            "expected a comptime Rust value",
        );
    }

    #[test]
    fn column_types_and_integer_conditions_are_checked() {
        rejects(
            parse_quote! { fn f(x: String) -> i32 { 1 } },
            "runtime columns have type i32",
        );
        rejects(
            parse_quote! { fn f(x: i32) -> i32 { if x { 1 } else { 2 } } },
            "expected a boolean condition",
        );
    }
}
