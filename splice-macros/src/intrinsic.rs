//! `#[intrinsic]`: declare kernel intrinsics, methods without a body that
//! a backend implements.
//!
//! On a unit struct, the attribute declares an opaque type that only
//! backends represent: a zero-sized struct (`PhantomData` of its type
//! parameters) that cannot be built in Rust, whose `#[intrinsic] impl`
//! declares its intrinsics as inherent methods.
//! On a trait, the attribute keeps the trait and adds the hidden items its
//! `#[intrinsic] impl`s define: the owner identity, the list of the impl's
//! intrinsics and one stage entry per method, which a kernel's method call
//! reaches through Rust's method resolution. On an impl of such a trait,
//! it gives each body-less method a body that panics (the method runs only
//! in a kernel) and defines the hidden items.
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Error, Fields, FnArg, GenericParam, ImplItem, Item, ItemImpl, ItemStruct, ItemTrait, Path,
    Result, ReturnType, TraitItem, TraitItemFn, Type,
};

pub(crate) fn expand(root: Path, item: TokenStream) -> Result<TokenStream> {
    match syn::parse2::<Item>(item)? {
        Item::Trait(declaration) => declare(&root, declaration),
        Item::Impl(implementation) if implementation.trait_.is_some() => {
            implement(&root, implementation)
        }
        Item::Impl(implementation) => inherent(&root, implementation),
        Item::Struct(declaration) => opaque(&root, declaration),
        item => Err(Error::new_spanned(
            item,
            "#[intrinsic] declares an intrinsic trait or struct, or implements its intrinsics",
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
    let owner = quote!(<Self as #trait_path>::__SPLICE_OWNER);
    let generics = quote!(&[#root::intrinsic::GenericArg::Type(<Self as #root::intrinsic::IntrinsicValue>::TYPE)]);
    let ids = quote!(<Self as #trait_path>::__SPLICE_INTRINSICS);
    let described = {
        let self_ty = &implementation.self_ty;
        quote!(<#self_ty as #trait_path>).to_string()
    };
    let items = methods(
        root,
        &mut implementation,
        &described,
        &owner,
        &generics,
        &ids,
    )?;
    for item in items {
        implementation.items.push(syn::parse2(item)?);
    }
    Ok(quote!(#implementation))
}

/// The methods of an `#[intrinsic] impl`: each body-less method gets a body
/// that panics and a hidden stage entry, and the impl gets the list of its
/// intrinsics. `owner`, `generics` and `list` are the tokens of the
/// intrinsics' owner, generic arguments and list.
fn methods(
    root: &Path,
    implementation: &mut ItemImpl,
    described: &str,
    owner: &TokenStream,
    generics: &TokenStream,
    list: &TokenStream,
) -> Result<Vec<TokenStream>> {
    // An inherent impl publishes its list; a trait impl's follows the trait.
    let list_vis = implementation.trait_.is_none().then(|| quote!(pub));
    let intrinsic = quote!(#root::intrinsic);
    let frontend = quote!(#root::frontend);
    let mut items = Vec::new();
    let mut ids = Vec::new();
    for item in std::mem::take(&mut implementation.items) {
        let (vis, method) = match item {
            ImplItem::Verbatim(tokens) => syn::parse::Parser::parse2(
                |input: syn::parse::ParseStream| {
                    let attrs = input.call(syn::Attribute::parse_outer)?;
                    let vis: syn::Visibility = input.parse()?;
                    let mut method: TraitItemFn = input.parse()?;
                    method.attrs.splice(0..0, attrs);
                    Ok((vis, method))
                },
                tokens,
            )?,
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
        if implementation.trait_.is_some() && !matches!(vis, syn::Visibility::Inherited) {
            return Err(Error::new_spanned(
                vis,
                "a trait's method has the trait's visibility",
            ));
        }
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
                owner: #owner,
                generics: #generics,
                method: #method_name,
                params: &[#(#params),*],
                result: #result,
            }
        });
        let index = ids.len() - 1;
        let message = format!(
            "`{described}::{method_name}` is a kernel intrinsic: only a #[kernel] calls it"
        );
        let attrs = &method.attrs;
        let stage = stage_name(name);
        items.push(quote! {
            #(#attrs)*
            #[allow(unused_variables)]
            #vis #sig {
                ::core::unreachable!(#message)
            }
        });
        items.push(quote! {
            #[doc(hidden)]
            #vis fn #stage(&self, __splice_builder: &mut dyn #frontend::KernelBuilder,
                __splice_operands: &[#frontend::methods::Operand]) -> #root::Result<#root::hir::Reg> {
                #intrinsic::stage(__splice_builder, &#list[#index], __splice_operands)
            }
        });
    }
    items.push(quote! {
        #[doc(hidden)]
        #list_vis const __SPLICE_INTRINSICS: &'static [#intrinsic::IntrinsicId] = &[#(#ids),*];
    });
    Ok(items)
}

fn opaque(root: &Path, mut declaration: ItemStruct) -> Result<TokenStream> {
    if !matches!(declaration.fields, Fields::Unit) {
        return Err(Error::new_spanned(
            &declaration.fields,
            "an intrinsic struct is opaque: declare it without fields (`struct Atomic<T>;`)",
        ));
    }
    let generics = &declaration.generics;
    if generics.where_clause.is_some()
        || generics.params.iter().any(
            |p| !matches!(p, GenericParam::Type(t) if t.bounds.is_empty() && t.default.is_none()),
        )
    {
        return Err(Error::new_spanned(
            generics,
            "an intrinsic struct takes only plain type parameters",
        ));
    }
    let params: Vec<_> = generics.type_params().map(|t| t.ident.clone()).collect();
    declaration.fields =
        Fields::Unnamed(syn::parse_quote!((::core::marker::PhantomData<fn() -> (#(#params,)*)>)));
    declaration.semi_token = Some(Default::default());
    let name = &declaration.ident;
    let intrinsic = quote!(#root::intrinsic);
    Ok(quote! {
        #declaration
        impl<#(#params: #intrinsic::IntrinsicValue),*> #intrinsic::IntrinsicType for #name<#(#params),*> {
            const TYPE: #intrinsic::OpaqueType = #intrinsic::OpaqueType {
                path: ::core::concat!(::core::module_path!(), "::", ::core::stringify!(#name)),
                generics: &[#(#intrinsic::GenericArg::Type(<#params as #intrinsic::IntrinsicValue>::TYPE)),*],
            };
        }
    })
}

/// `#[intrinsic] impl<T> Atomic<T> { fn swap(&self, value: T) -> T; }`:
/// the intrinsics of an intrinsic struct. Each type parameter gets the
/// bound `IntrinsicValue`.
fn inherent(root: &Path, mut implementation: ItemImpl) -> Result<TokenStream> {
    let intrinsic = quote!(#root::intrinsic);
    for param in implementation.generics.type_params_mut() {
        param
            .bounds
            .push(syn::parse_quote!(#intrinsic::IntrinsicValue));
    }
    let owner = quote!(#intrinsic::Owner::Type(<Self as #intrinsic::IntrinsicType>::TYPE.path));
    let generics = quote!(<Self as #intrinsic::IntrinsicType>::TYPE.generics);
    let ids = quote!(Self::__SPLICE_INTRINSICS);
    let described = {
        let self_ty = &implementation.self_ty;
        quote!(#self_ty).to_string()
    };
    let items = methods(
        root,
        &mut implementation,
        &described,
        &owner,
        &generics,
        &ids,
    )?;
    for item in items {
        implementation.items.push(syn::parse2(item)?);
    }
    Ok(quote!(#implementation))
}
