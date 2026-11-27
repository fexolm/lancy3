//! `#[intrinsic]`: declare kernel intrinsics, methods without a body that
//! a backend implements.
//!
//! On a trait, the attribute keeps the trait and adds the hidden items its
//! `#[intrinsic] impl`s define: the owner identity, the list of the impl's
//! intrinsics and one stage entry per method, which a kernel's method call
//! reaches through Rust's method resolution. On an impl of such a trait,
//! it gives each body-less method a body that panics (the method runs only
//! in a kernel) and defines the hidden items.
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Error, FnArg, ImplItem, Item, ItemImpl, ItemTrait, Path, Result, ReturnType, TraitItem,
    TraitItemFn, Type,
};

pub(crate) fn expand(root: Path, item: TokenStream) -> Result<TokenStream> {
    match syn::parse2::<Item>(item)? {
        Item::Trait(declaration) => declare(&root, declaration),
        Item::Impl(implementation) => implement(&root, implementation),
        item => Err(Error::new_spanned(
            item,
            "#[intrinsic] declares a trait of intrinsics or implements one for a kernel type",
        )),
    }
}

/// The hidden stage entry of `method`.
fn stage_name(method: &syn::Ident) -> syn::Ident {
    format_ident!(
        "__splice_stage_{}",
        method.to_string().trim_start_matches("r#"),
        span = method.span()
    )
}

/// Check the shape of an intrinsic method: no body, no generics, a
/// receiver taken by value or shared reference, typed parameters and a
/// result.
fn method_shape(method: &TraitItemFn) -> Result<()> {
    let sig = &method.sig;
    if method.default.is_some() {
        return Err(Error::new_spanned(
            &method.default,
            "an intrinsic has no body: the backend implements it",
        ));
    }
    if sig.constness.is_some()
        || sig.asyncness.is_some()
        || sig.unsafety.is_some()
        || sig.abi.is_some()
        || sig.variadic.is_some()
        || !sig.generics.params.is_empty()
        || sig.generics.where_clause.is_some()
    {
        return Err(Error::new_spanned(
            sig,
            "an intrinsic is a plain method without generic parameters",
        ));
    }
    match sig.inputs.first() {
        Some(FnArg::Receiver(receiver)) if receiver.mutability.is_none() => {}
        _ => {
            return Err(Error::new_spanned(
                sig,
                "an intrinsic takes `self` or `&self` first",
            ));
        }
    }
    if let Some(FnArg::Receiver(receiver)) = sig.inputs.iter().nth(1) {
        return Err(Error::new_spanned(receiver, "one receiver"));
    }
    if matches!(sig.output, ReturnType::Default) {
        return Err(Error::new_spanned(
            sig,
            "an intrinsic returns a kernel value",
        ));
    }
    Ok(())
}

fn declare(root: &Path, mut declaration: ItemTrait) -> Result<TokenStream> {
    if !declaration.generics.params.is_empty() {
        return Err(Error::new_spanned(
            &declaration.generics,
            "an intrinsic trait has no generic parameters",
        ));
    }
    let frontend = quote!(#root::frontend);
    let mut stages = Vec::new();
    for item in &declaration.items {
        let TraitItem::Fn(method) = item else {
            return Err(Error::new_spanned(
                item,
                "an intrinsic trait declares only methods",
            ));
        };
        method_shape(method)?;
        let stage = stage_name(&method.sig.ident);
        stages.push(quote! {
            #[doc(hidden)]
            fn #stage(&self, __splice_builder: &mut dyn #frontend::KernelBuilder,
                __splice_operands: &[#frontend::methods::Operand]) -> #root::Result<#root::hir::Reg>;
        });
    }
    let name = &declaration.ident;
    let intrinsic = quote!(#root::intrinsic);
    declaration.items.push(syn::parse_quote! {
        /// The trait that declares these intrinsics.
        #[doc(hidden)]
        const __SPLICE_OWNER: #intrinsic::Owner = #intrinsic::Owner::Trait(
            ::core::concat!(::core::module_path!(), "::", ::core::stringify!(#name)));
    });
    declaration.items.push(syn::parse_quote! {
        /// The intrinsics of an implementation, in declaration order.
        #[doc(hidden)]
        const __SPLICE_INTRINSICS: &'static [#intrinsic::IntrinsicId];
    });
    for stage in stages {
        declaration.items.push(syn::parse2(stage)?);
    }
    Ok(quote!(#declaration))
}

/// The value type of a parameter or result type `ty`.
fn value_type(root: &Path, ty: &Type) -> TokenStream {
    quote!(<#ty as #root::intrinsic::IntrinsicValue>::TYPE)
}

fn implement(root: &Path, mut implementation: ItemImpl) -> Result<TokenStream> {
    let Some((None, trait_path, _)) = &implementation.trait_ else {
        return Err(Error::new_spanned(
            &implementation,
            "#[intrinsic] implements an intrinsic trait (`impl Trait for Type`)",
        ));
    };
    let trait_path = trait_path.clone();
    if !implementation.generics.params.is_empty() {
        return Err(Error::new_spanned(
            &implementation.generics,
            "an intrinsic implementation is for one concrete kernel type",
        ));
    }
    let self_ty = &implementation.self_ty;
    let intrinsic = quote!(#root::intrinsic);
    let frontend = quote!(#root::frontend);
    let mut items = Vec::new();
    let mut ids = Vec::new();
    let described = quote!(#self_ty as #trait_path).to_string();
    for item in std::mem::take(&mut implementation.items) {
        let method: TraitItemFn = match item {
            ImplItem::Verbatim(tokens) => syn::parse2(tokens)?,
            ImplItem::Fn(method) => {
                return Err(Error::new_spanned(
                    &method.block,
                    "an intrinsic has no body: the backend implements it",
                ));
            }
            item => {
                return Err(Error::new_spanned(
                    item,
                    "an intrinsic implementation declares only methods",
                ));
            }
        };
        method_shape(&method)?;
        let sig = &method.sig;
        let name = &sig.ident;
        let params = sig.inputs.iter().map(|input| match input {
            FnArg::Receiver(_) => value_type(root, &syn::parse_quote!(Self)),
            FnArg::Typed(typed) => value_type(root, &typed.ty),
        });
        let ReturnType::Type(_, result) = &sig.output else {
            unreachable!("method_shape requires a result")
        };
        let result = value_type(root, result);
        let method_name = name.to_string();
        let method_name = method_name.trim_start_matches("r#");
        ids.push(quote! {
            #intrinsic::IntrinsicId {
                owner: <Self as #trait_path>::__SPLICE_OWNER,
                generics: &[#intrinsic::GenericArg::Type(<Self as #intrinsic::IntrinsicValue>::TYPE)],
                method: #method_name,
                params: &[#(#params),*],
                result: #result,
            }
        });
        let index = ids.len() - 1;
        let message = format!(
            "`<{described}>::{method_name}` is a kernel intrinsic: only a #[kernel] calls it"
        );
        let attrs = &method.attrs;
        let stage = stage_name(name);
        items.push(quote! {
            #(#attrs)*
            #[allow(unused_variables)]
            #sig {
                ::core::unreachable!(#message)
            }
        });
        items.push(quote! {
            #[doc(hidden)]
            fn #stage(&self, __splice_builder: &mut dyn #frontend::KernelBuilder,
                __splice_operands: &[#frontend::methods::Operand]) -> #root::Result<#root::hir::Reg> {
                #intrinsic::stage(__splice_builder, &<Self as #trait_path>::__SPLICE_INTRINSICS[#index], __splice_operands)
            }
        });
    }
    items.push(quote! {
        #[doc(hidden)]
        const __SPLICE_INTRINSICS: &'static [#intrinsic::IntrinsicId] = &[#(#ids),*];
    });
    for item in items {
        implementation.items.push(syn::parse2(item)?);
    }
    Ok(quote!(#implementation))
}
