use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Error, Fields, Result, parse_quote};

pub fn expand(input: DeriveInput) -> Result<TokenStream> {
    let name = &input.ident;
    let root = crate::derive_root(&input.attrs)?;
    let key = quote!(#root::ComptimeKey);
    let mut generics = input.generics.clone();
    for param in generics.type_params_mut() {
        param.bounds.push(parse_quote!(#key));
    }
    let (impl_g, type_g, where_g) = generics.split_for_impl();
    let types = generics.type_params().map(|p| &p.ident);
    let consts: Vec<_> = generics.const_params().collect();
    let const_names = consts.iter().map(|p| &p.ident);
    // Each derive owns a nominal marker. Const and type instantiations stay
    // distinct while host lifetimes never enter the owned key.
    let marker_params = consts.iter().map(|p| {
        let name = &p.ident;
        let ty = &p.ty;
        quote!(const #name: #ty)
    });
    let arm = |path: TokenStream, fields: &Fields, tag: Option<u32>| {
        let bindings: Vec<_> = (0..fields.len())
            .map(|i| format_ident!("__field_{i}"))
            .collect();
        let pattern = match fields {
            Fields::Named(fields) => {
                let names = fields.named.iter().map(|f| &f.ident);
                quote!({ #(#names: #bindings),* })
            }
            Fields::Unnamed(_) => quote!((#(#bindings),*)),
            Fields::Unit => quote!(),
        };
        let tag = tag.map(|tag| quote!(#key::write_key(&#tag, bytes);));
        quote!(#path #pattern => { #tag #(#key::write_key(#bindings, bytes);)* })
    };
    let arms = match input.data {
        Data::Struct(s) => vec![arm(quote!(Self), &s.fields, None)],
        Data::Enum(e) => e
            .variants
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let variant = &v.ident;
                arm(quote!(Self::#variant), &v.fields, Some(i as u32))
            })
            .collect(),
        Data::Union(_) => {
            return Err(Error::new_spanned(
                name,
                "union fields do not have a known active variant",
            ));
        }
    };
    Ok(quote! {
        const _: () = {
            pub struct __SpliceFamily<#(#marker_params),*>;
            impl #impl_g #key for #name #type_g #where_g {
                type Family = (__SpliceFamily<#(#const_names),*>, #(<#types as #key>::Family,)*);
                fn write_key(&self, bytes: &mut ::std::vec::Vec<u8>) {
                    match self { #(#arms),* }
                }
            }
        };
    })
}
