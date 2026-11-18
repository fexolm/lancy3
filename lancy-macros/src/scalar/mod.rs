//! Staged scalar functions. Rust executes host code while symbolic runtime
//! expressions append register operations and control-flow edges to the IR.
mod flow;
mod lower;
use lower::{Binding, Lower, Scalar, Stage};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use std::collections::HashMap;
use syn::visit_mut::VisitMut;
use syn::{Error, FnArg, GenericParam, ItemFn, Meta, Pat, Path, Result, ReturnType};

pub(super) fn private(name: &str) -> Ident {
    Ident::new(name, Span::mixed_site())
}

struct Native;
impl VisitMut for Native {
    fn visit_expr_mut(&mut self, e: &mut syn::Expr) {
        match e {
            syn::Expr::Macro(m) if m.mac.path.is_ident("comptime") => {
                if let Ok(expr) = m.mac.parse_body() {
                    *e = expr;
                    self.visit_expr_mut(e);
                }
            }
            _ => syn::visit_mut::visit_expr_mut(self, e),
        }
    }
}

// Only configuration gates belong on all generated items. A cfg_attr may
// also contain function-specific attributes such as inline or test.
fn configuration(meta: &Meta) -> Result<Option<TokenStream>> {
    if meta.path().is_ident("cfg") {
        return Ok(Some(quote!(#meta)));
    }
    let Meta::List(list) = meta else {
        return Ok(None);
    };
    if !list.path.is_ident("cfg_attr") {
        return Ok(None);
    }
    let args = list
        .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)?;
    let mut args = args.iter();
    let condition = args
        .next()
        .ok_or_else(|| Error::new_spanned(list, "expected a cfg condition"))?;
    let gates = args
        .map(configuration)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    Ok((!gates.is_empty()).then(|| quote!(cfg_attr(#condition, #(#gates),*))))
}

pub(super) fn expand(root: Path, func: ItemFn) -> Result<TokenStream> {
    let mut native = func.clone();
    for arg in &mut native.sig.inputs {
        if let FnArg::Typed(arg) = arg {
            arg.attrs.clear();
        }
    }
    Native.visit_block_mut(&mut native.block);
    // A scalar kernel also provides a build-only entry point. In applications
    // that use exclusively this entry, the native reference body is unused.
    native.attrs.push(syn::parse_quote!(#[allow(dead_code)]));
    let ItemFn {
        attrs,
        vis,
        sig,
        block,
    } = func;
    let gates = attrs
        .iter()
        .map(|a| configuration(&a.meta))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if sig.asyncness.is_some()
        || sig.unsafety.is_some()
        || sig.abi.is_some()
        || sig.variadic.is_some()
    {
        return Err(Error::new_spanned(
            &sig,
            "expected an ordinary kernel function",
        ));
    }
    if !matches!(&sig.output, ReturnType::Type(_, t) if super::scalar_type(t, "i32")) {
        return Err(Error::new_spanned(&sig.output, "scalar kernels return i32"));
    }
    let name = &sig.ident;
    let generic = &sig.generics;
    let where_ = &generic.where_clause;
    let args: Vec<_> = generic
        .params
        .iter()
        .filter_map(|p| match p {
            GenericParam::Type(t) => Some(&t.ident),
            GenericParam::Const(c) => Some(&c.ident),
            _ => None,
        })
        .collect();
    let fish = (!args.is_empty()).then(|| quote!(::<#(#args),*>));
    let hir = quote!(#root::hir);
    let support = quote!(#root::frontend::scalar);
    let builder = private("__lancy_builder");
    let exit = private("__lancy_return_label");
    let result = private("__lancy_result");
    let callee = private("__lancy_callee");
    let guard = private("__lancy_guard");
    let mut bindings = HashMap::new();
    let mut formals = Vec::new();
    let mut host_decls = Vec::new();
    let mut host_names = Vec::new();
    let mut names = Vec::new();
    let mut reg_names = Vec::new();
    let mut buffer_names = Vec::new();
    let mut buffer_kinds = Vec::new();
    let mut inline_decls = Vec::new();
    let mut call_decls = Vec::new();
    let mut bind_args = Vec::new();
    let mut unpack = Vec::new();
    let mut copy_mutable = Vec::new();
    for arg in &sig.inputs {
        let FnArg::Typed(arg) = arg else {
            return Err(Error::new_spanned(arg, "kernel methods are unsupported"));
        };
        let Pat::Ident(pat) = &*arg.pat else {
            return Err(Error::new_spanned(arg, "expected a named argument"));
        };
        if pat.by_ref.is_some() || pat.subpat.is_some() {
            return Err(Error::new_spanned(pat, "expected a named argument"));
        }
        let host = match arg.attrs.as_slice() {
            [] => false,
            [a] if matches!(&a.meta,Meta::Path(p) if p.is_ident("comptime")) => true,
            _ => {
                return Err(Error::new_spanned(
                    arg,
                    "expected #[comptime] or an unannotated runtime argument",
                ));
            }
        };
        let id = &pat.ident;
        let ty = &arg.ty;
        names.push(id);
        let formal = if host {
            if pat.mutability.is_some() {
                return Err(Error::new_spanned(pat, "comptime arguments are immutable"));
            }
            host_decls.push(quote!(#id: #ty));
            inline_decls.push(quote!(#id: #ty));
            host_names.push(id);
            call_decls.push(quote!(#id: #support::Host<#ty>));
            unpack.push(quote!(let #id = #id.0;));
            Binding::Host
        } else if let Some(writable) = slice_type(ty) {
            if pat.mutability.is_some() {
                return Err(Error::new_spanned(
                    pat,
                    "use &mut [i32] to write to a slice",
                ));
            }
            let index = buffer_names.len() as u32;
            buffer_names.push(id);
            buffer_kinds.push(if writable {
                quote!(#hir::BufferKind::Write)
            } else {
                quote!(#hir::BufferKind::Read)
            });
            bind_args.push(quote!(let #id = #hir::Buffer(#index);));
            inline_decls.push(quote!(#id: #hir::Buffer));
            call_decls.push(quote!(#id: #hir::Buffer));
            Binding::Buffer(writable)
        } else {
            if !super::scalar_type(ty, "i32") {
                return Err(Error::new_spanned(
                    ty,
                    "runtime parameters must be i32, &[i32], or &mut [i32]",
                ));
            }
            let index = reg_names.len();
            reg_names.push(id);
            if pat.mutability.is_some() {
                let target = private("__lancy_mutable_argument");
                copy_mutable.push(quote!(let #id = {let #target=#builder.local();#builder.assign(#target,#id);#target};));
            }
            bind_args.push(quote!(let #id = #builder.arg(#index);));
            inline_decls.push(quote!(#id: #hir::Reg));
            call_decls.push(quote!(#id: impl #support::RunArg));
            unpack.push(quote!(let #id = #support::RunArg::reg(#id, #builder);));
            Binding::Reg(Scalar::Int, pat.mutability.is_some())
        };
        bindings.insert(id.to_string(), formal);
        formals.push(formal);
    }
    let arity = reg_names.len();
    let (new_builder, call) = if buffer_names.is_empty() {
        (
            quote!(#hir::FunctionBuilder::new(#arity)),
            quote!(Ok(#builder.call(#callee, &[#(#reg_names),*]))),
        )
    } else {
        (
            quote!(#hir::FunctionBuilder::with_buffers(#arity, vec![#(#buffer_kinds),*])),
            quote!(#builder.call_with_buffers(#callee, &[#(#reg_names),*], &[#(#buffer_names),*])),
        )
    };
    let flow = flow::analyze(&block, &bindings);
    let lower = Lower {
        root: &root,
        name,
        formals: &formals,
        bindings,
        flow: &flow,
        builder: &builder,
        exit: &exit,
        result: &result,
        fish: quote!(#fish),
        loop_labels: None,
        region_depth: 0,
    };
    let body = lower.block(&block)?;
    let code = if body.stage == Stage::Never {
        body.code
    } else {
        let reg = lower.reg(body, Scalar::Int, &block)?;
        quote!({let value = #reg; #builder.assign(#result,value); #builder.jump(#exit);})
    };
    Ok(quote! {
        #native
        #(#[#gates])*
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        #vis enum #name {}
        #(#[#gates])*
        impl #name {
            #vis fn build #generic (#(#host_decls),*) -> ::core::result::Result<#hir::Function,String> #where_ {
                let mut #builder = #new_builder;
                #(#bind_args)*
                let output = Self::__lancy_inline #fish (#(#names,)* &mut #builder)?;
                #builder.ret(output);
                #builder.finish()
            }
            #[doc(hidden)]
            #[allow(clippy::too_many_arguments)]
            #vis fn __lancy_call #generic (#(#call_decls,)* #builder: &mut #hir::FunctionBuilder)
                -> ::core::result::Result<#hir::Reg,String> #where_ {
                #(#unpack)*
                let #callee = Self::build #fish (#(#host_names),*)?;
                #call
            }
            #[allow(unreachable_code, clippy::too_many_arguments)]
            fn __lancy_inline #generic (#(#inline_decls,)* #builder:&mut #hir::FunctionBuilder)
                -> ::core::result::Result<#hir::Reg,String> #where_ {
                let #guard = #support::enter_expansion()?;
                #(#copy_mutable)*
                let #result = #builder.local();
                let #exit = #builder.label();
                #code
                #builder.mark(#exit);
                Ok(#result)
            }
        }
    })
}

fn slice_type(ty: &syn::Type) -> Option<bool> {
    let syn::Type::Reference(reference) = ty else {
        return None;
    };
    let syn::Type::Slice(slice) = &*reference.elem else {
        return None;
    };
    super::scalar_type(&slice.elem, "i32").then_some(reference.mutability.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::{Item, parse_quote};

    #[test]
    fn a_load_cannot_be_used_as_a_recursive_comptime_argument() {
        let f = parse_quote! {
            fn walk(#[comptime] depth: i32, column: &[i32]) -> i32 {
                if depth == 0 { 1 } else { walk(column[0], column) }
            }
        };
        let error = expand(parse_quote!(::lancy_jit), f).unwrap_err();
        assert!(error.to_string().contains("expected a comptime value"));
    }

    #[test]
    fn writes_require_a_mutable_slice() {
        let f = parse_quote! {
            fn edit(column: &[i32]) -> i32 { column[0] = 1; 0 }
        };
        let error = expand(parse_quote!(::lancy_jit), f).unwrap_err();
        assert!(error.to_string().contains("requires &mut [i32]"));
    }

    #[test]
    fn all_generated_items_follow_the_kernel_configuration() {
        let f = parse_quote! {
            #[cfg(feature = "kernels")]
            #[cfg_attr(all(), inline, cfg(target_arch = "x86_64"))]
            fn sum(x: i32, y: i32) -> i32 { x + y }
        };
        let items = syn::parse2::<syn::File>(expand(parse_quote!(::lancy_jit), f).unwrap())
            .unwrap()
            .items;
        assert_eq!(items.len(), 3);
        for item in items {
            let attrs = match item {
                Item::Fn(f) => f.attrs,
                Item::Enum(e) => e.attrs,
                Item::Impl(i) => i.attrs,
                _ => panic!("unexpected item in scalar kernel expansion"),
            };
            assert!(attrs.iter().any(|a| a.path().is_ident("cfg")));
            let nested = attrs
                .iter()
                .find(|a| a.path().is_ident("cfg_attr"))
                .unwrap();
            assert!(quote!(#nested).to_string().contains("target_arch"));
        }
    }
}
