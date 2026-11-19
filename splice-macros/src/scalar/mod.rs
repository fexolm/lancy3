//! Staged scalar functions. Rust executes host code while symbolic runtime
//! expressions append register operations and control-flow edges to the IR.
// Generated locals and labels share a hygiene context across fragments. Tokens
// interpolated from the kernel keep their original spans and lexical bindings.
macro_rules! emit {
    ($($tokens:tt)*) => { quote::quote_spanned!(proc_macro2::Span::mixed_site()=> $($tokens)*) };
}

mod flow;
mod lower;
use lower::{Binding, BufferType, Elem, Lower, Scalar};
use proc_macro2::{Ident, TokenStream};
use std::collections::HashMap;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{Error, FnArg, GenericParam, ItemFn, Meta, Pat, Path, Result, ReturnType, Type};

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
        return Ok(Some(emit!(#meta)));
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
        .filter_map(|meta| configuration(meta).transpose())
        .collect::<Result<Vec<_>>>()?;
    Ok((!gates.is_empty()).then(|| emit!(cfg_attr(#condition, #(#gates),*))))
}

pub(super) fn expand(root: Path, mut native: ItemFn) -> Result<TokenStream> {
    let sig = &native.sig;
    let vis = &native.vis;
    let gates = native
        .attrs
        .iter()
        .filter_map(|a| configuration(&a.meta).transpose())
        .collect::<Result<Vec<_>>>()?;
    if sig.asyncness.is_some()
        || sig.unsafety.is_some()
        || sig.abi.is_some()
        || sig.variadic.is_some()
    {
        return Err(Error::new_spanned(
            sig,
            "expected an ordinary kernel function",
        ));
    }
    let ReturnType::Type(_, ty) = &sig.output else {
        return Err(Error::new_spanned(
            &sig.output,
            "expected a kernel return type",
        ));
    };
    let result_type = Scalar::from_type(ty)
        .filter(|t| matches!(t, Scalar::Int | Scalar::Vector(_) | Scalar::Mask(_)))
        .ok_or_else(|| Error::new_spanned(ty, "expected i32, Lane<T> or Mask<T>"))?;
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
    let fish = (!args.is_empty()).then(|| emit!(::<#(#args),*>));
    let hir = emit!(#root::hir);
    let factory = emit!(dyn #root::frontend::KernelBuilder);
    let support = emit!(#root::frontend::scalar);
    let mut bindings = HashMap::new();
    let mut formals = Vec::new();
    let mut host_types = Vec::new();
    let mut host_names = Vec::new();
    let mut helper_args = Vec::new();
    let mut inline_args = Vec::new();
    let mut reg_names = Vec::new();
    let mut buffer_names = Vec::new();
    let mut buffer_kinds = Vec::new();
    let mut call_decls = Vec::new();
    let mut copy_mutable = Vec::new();
    for arg in &sig.inputs {
        let (id, ty, formal) = parameter(arg)?;
        helper_args.push(if matches!(formal, Binding::Host) {
            emit!(#support::Host(#id))
        } else {
            emit!(#id)
        });
        let decl = match formal {
            Binding::Host => {
                host_names.push(id);
                host_types.push(ty);
                inline_args.push(emit!(#support::Host(#id)));
                emit!(#support::Host(#id): #support::Host<#ty>)
            }
            Binding::Buffer(writable) => {
                let index = buffer_names.len() as u32;
                inline_args.push(emit!(#hir::Buffer(#index)));
                buffer_names.push(id);
                let elem = writable.elem.tokens(&root);
                let write = writable.write;
                buffer_kinds.push(emit!(#hir::BufferKind::new(#elem,#write)));
                emit!(#id: #hir::Buffer)
            }
            Binding::Locals(..) => unreachable!("local register collections are not parameters"),
            Binding::Reg(_, mutable) => {
                let index = reg_names.len();
                inline_args.push(emit!(__splice_builder.arg(#index)));
                reg_names.push(id);
                if mutable {
                    copy_mutable.push(emit!(let #id = __splice_builder.copy(#id);));
                }
                emit!(#id: impl #support::RunArg)
            }
        };
        call_decls.push(decl);
        bindings.insert(id.to_string(), formal);
        formals.push(formal);
    }
    let arity = reg_names.len();
    let host_decls = emit!(#(#host_names: #host_types,)*);
    let unpack = emit!(#(let #reg_names = #support::RunArg::reg(#reg_names, __splice_builder);)*);
    let flow = flow::analyze(&native.block, &bindings);
    let lower = Lower {
        root: &root,
        name,
        result_type,
        formals: &formals,
        bindings,
        flow: &flow,
        fish: emit!(#fish),
        loop_kind: None,
        region_depth: 0,
    };
    let code = lower.kernel_body(&native.block)?;
    let kinds = emit!(const { &[#(#buffer_kinds),*] });
    // Rust requires lifetime parameters before the backend type; the backend
    // is still the first argument at the usual lifetime-elided turbofish.
    let mut compile_generics = generic.clone();
    let backend = emit!(__SpliceCodegen);
    let lifetimes = generic.lifetimes().count();
    compile_generics
        .params
        .insert(lifetimes, syn::parse_quote!(#backend: #root::Codegen));
    let callback = emit!(::std::boxed::Box::new(move |__splice_builder| Self::__stage_body #fish (#(#host_names,)* __splice_builder)));
    let frontend = emit!(#root::frontend);
    let value_args = host_names
        .iter()
        .copied()
        .chain(generic.const_params().map(|p| &p.ident));
    let type_args = generic.type_params().map(|p| &p.ident);
    let specialize = emit! {
        #[allow(unused_imports)]
        use #frontend::MaybeKey as _;
        #[allow(unused_mut)]
        let mut __splice_key = Some(#frontend::Specialization::new::<Self>(#arity, #kinds));
        #((&#frontend::KeyArgument::<#type_args>::for_type()).add_type(&mut __splice_key);)*
        #((&#frontend::KeyArgument::for_value(&#value_args)).add_value(&#value_args, &mut __splice_key);)*
    };
    let typed_helper = result_type != Scalar::Int
        || formals
            .iter()
            .any(|f| matches!(f, Binding::Reg(Scalar::Vector(_) | Scalar::Mask(_), _)));
    let result_value_type = result_type.value_type(&root);
    let standalone = emit! {
            #vis fn build #compile_generics (#host_decls) -> #root::Result<<#backend as #root::Codegen>::Output> #where_ {
                <#backend as #root::Codegen>::build(#arity, #kinds, #callback)
            }
            #vis fn ir #generic (#host_decls) -> #root::Result<#hir::Function> #where_ {
                #root::frontend::materialize(#arity, #kinds, #callback)
            }
            fn __stage_body #generic (#host_decls __splice_builder: &mut #factory) -> #root::Result<()> #where_ {
                __splice_builder.set_source(&#root::frontend::KernelSource {
                    name: stringify!(#name),
                    scalars: &[#(stringify!(#reg_names)),*],
                    buffers: &[#(stringify!(#buffer_names)),*],
                });
                let output = Self::__splice_inline #fish (#(#inline_args,)* __splice_builder)?;
                __splice_builder.ret(output);
                Ok(())
            }
            #[doc(hidden)]
            #[allow(clippy::too_many_arguments)]
            #vis fn __splice_call #generic (#(#call_decls,)* __splice_builder: &mut #factory)
                -> #root::Result<#hir::Reg> #where_ {
                #unpack
                #specialize
                let __splice_body: #frontend::Body<'_> = #callback;
                match __splice_key {
                    Some(key) => __splice_builder.call_specialized(key, &[#(#reg_names),*], &[#(#buffer_names),*], __splice_body),
                    None => __splice_builder.call_kernel(#arity, #kinds, &[#(#reg_names),*], &[#(#buffer_names),*], __splice_body),
                }
            }

    };
    // The public kernel ABI transports i32 scalars and slices. SIMD helpers
    // expand into their caller so vectors and masks stay in typed registers.
    let entry_points = if typed_helper {
        emit! {
            #[doc(hidden)]
            #[allow(clippy::too_many_arguments)]
            #vis fn __splice_call #generic (#(#call_decls,)* __splice_builder:&mut #factory)->#root::Result<#hir::Reg> #where_{
                Self::__splice_inline #fish (#(#helper_args,)* __splice_builder)
            }
        }
    } else {
        standalone
    };
    let generated = emit! {
        #(#[#gates])*
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        #vis enum #name {}
        #(#[#gates])*
        impl #name {
            #[doc(hidden)]
            #vis const __SPLICE_RETURN: #hir::ValueType = #result_value_type;
            #entry_points
            #[allow(unreachable_code, clippy::too_many_arguments)]
            fn __splice_inline #generic (#(#call_decls,)* __splice_builder:&mut #factory)
                -> #root::Result<#hir::Reg> #where_ {
                let __splice_guard = #support::enter_expansion()?;
                #unpack
                #(#copy_mutable)*
                let __splice_result = __splice_builder.typed_local(#result_value_type);
                let __splice_return_label = __splice_builder.label();
                #code
                __splice_builder.mark(__splice_return_label);
                Ok(__splice_result)
            }
        }
    };
    for arg in &mut native.sig.inputs {
        if let FnArg::Typed(arg) = arg {
            arg.attrs.clear();
        }
    }
    Native.visit_block_mut(&mut native.block);
    native.attrs.push(syn::parse_quote!(#[allow(dead_code)]));
    Ok(emit!(#native #generated))
}

fn parameter(arg: &FnArg) -> Result<(&Ident, &Type, Binding)> {
    let FnArg::Typed(arg) = arg else {
        return Err(Error::new_spanned(arg, "kernel methods are unsupported"));
    };
    let pat = named(&arg.pat)?;
    let mutable = pat.mutability.is_some();
    let binding = match arg.attrs.as_slice() {
        [attr] if matches!(&attr.meta, Meta::Path(p) if p.is_ident("comptime")) => {
            if mutable {
                return Err(Error::new_spanned(pat, "comptime arguments are immutable"));
            }
            Binding::Host
        }
        [] if matches!(
            Scalar::from_type(&arg.ty),
            Some(Scalar::Int | Scalar::Vector(_) | Scalar::Mask(_))
        ) =>
        {
            Binding::Reg(Scalar::from_type(&arg.ty).unwrap(), mutable)
        }
        [] => {
            let writable = slice_type(&arg.ty).ok_or_else(|| {
                Error::new_spanned(
                    &arg.ty,
                    "runtime parameters must be i32 or a slice of u16, i32, i64, f32, or f64",
                )
            })?;
            if mutable {
                return Err(Error::new_spanned(
                    pat,
                    "use &mut [i32] to write to a slice",
                ));
            }
            Binding::Buffer(writable)
        }
        _ => {
            return Err(Error::new_spanned(
                arg,
                "expected #[comptime] or an unannotated runtime argument",
            ));
        }
    };
    Ok((&pat.ident, &arg.ty, binding))
}

pub(super) fn compound_op(op: &syn::BinOp) -> Option<&'static str> {
    use syn::BinOp::*;
    match op {
        AddAssign(_) => Some("Add"),
        SubAssign(_) => Some("Sub"),
        MulAssign(_) => Some("Mul"),
        DivAssign(_) => Some("Div"),
        RemAssign(_) => Some("Rem"),
        BitAndAssign(_) => Some("And"),
        BitOrAssign(_) => Some("Or"),
        BitXorAssign(_) => Some("Xor"),
        ShlAssign(_) => Some("Shl"),
        ShrAssign(_) => Some("Shr"),
        _ => None,
    }
}

fn bind_pattern<T: Copy>(names: &mut HashMap<String, T>, pat: &Pat, binding: T) {
    struct Bind<'a, T>(&'a mut HashMap<String, T>, T);
    impl<'ast, T: Copy> Visit<'ast> for Bind<'_, T> {
        fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
            self.0.insert(pat.ident.to_string(), self.1);
            syn::visit::visit_pat_ident(self, pat);
        }
    }
    Bind(names, binding).visit_pat(pat);
}

fn named(pat: &Pat) -> Result<&syn::PatIdent> {
    match pat {
        Pat::Ident(p) if p.by_ref.is_none() && p.subpat.is_none() => Ok(p),
        _ => Err(Error::new_spanned(pat, "expected a named binding")),
    }
}

fn slice_type(ty: &syn::Type) -> Option<BufferType> {
    let syn::Type::Reference(reference) = ty else {
        return None;
    };
    let syn::Type::Slice(slice) = &*reference.elem else {
        return None;
    };
    Some(BufferType {
        elem: Elem::parse(&slice.elem)?,
        write: reference.mutability.is_some(),
    })
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
        let error = expand(parse_quote!(::splice), f).unwrap_err();
        assert!(error.to_string().contains("expected a comptime value"));
    }

    #[test]
    fn writes_require_a_mutable_slice() {
        let f = parse_quote! {
            fn edit(column: &[i32]) -> i32 { column[0] = 1; 0 }
        };
        let error = expand(parse_quote!(::splice), f).unwrap_err();
        assert!(error.to_string().contains("requires &mut [i32]"));
    }

    #[test]
    fn all_generated_items_follow_the_kernel_configuration() {
        let f = parse_quote! {
            #[cfg(feature = "kernels")]
            #[cfg_attr(all(), inline, cfg(target_arch = "x86_64"))]
            fn sum(x: i32, y: i32) -> i32 { x + y }
        };
        let items = syn::parse2::<syn::File>(expand(parse_quote!(::splice), f).unwrap())
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
            assert!(emit!(#nested).to_string().contains("target_arch"));
        }
    }
}
