//! Rust attribute frontend for Splice kernels.
use proc_macro::TokenStream;
use syn::{
    Data, DeriveInput, Error, Index, Member, Path, Token, parse_macro_input, parse_quote,
    punctuated::Punctuated,
};

mod comptime_key;
mod declaration;
mod import;
mod overflow;
mod scalar;

/// Derive a value key for a struct or enum used as a specialization parameter.
#[proc_macro_derive(ComptimeKey, attributes(splice))]
pub fn comptime_key(item: TokenStream) -> TokenStream {
    comptime_key::expand(parse_macro_input!(item as syn::DeriveInput))
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Describe the owned fields of a Rust structure for Splice's typed IR.
#[proc_macro_derive(KernelType)]
pub fn kernel_type(item: TokenStream) -> TokenStream {
    let item = parse_macro_input!(item as DeriveInput);
    let Data::Struct(data) = item.data else {
        return Error::new_spanned(item.ident, "KernelType can only be derived for structs")
            .into_compile_error()
            .into();
    };
    let name = item.ident;
    let (params, args, where_) = item.generics.split_for_impl();
    let fields = data.fields.iter().enumerate().map(|(index, field)| {
        let member = field
            .ident
            .clone()
            .map_or_else(|| Member::Unnamed(Index::from(index)), Member::Named);
        let label = match &member {
            Member::Named(name) => name.to_string().trim_start_matches("r#").to_owned(),
            Member::Unnamed(index) => index.index.to_string(),
        };
        let ty = &field.ty;
        quote::quote!(::splice::KernelField {
            name: #label,
            offset: ::core::mem::offset_of!(Self, #member),
            type_name: ::core::any::type_name::<#ty>,
        })
    });
    quote::quote! {
        impl #params ::splice::KernelType for #name #args #where_ {
            const FIELDS: &'static [::splice::KernelField] = &[#(#fields),*];
        }
    }
    .into()
}

/// Define a Splice kernel. `name::build::<Backend>(...)` stages its comptime
/// parameters into the selected backend. `name::ir(...)` builds portable IR.
/// `name::print(...)` specializes the Rust source without building IR.
#[proc_macro_attribute]
pub fn kernel(options: TokenStream, item: TokenStream) -> TokenStream {
    let paths = parse_macro_input!(options with Punctuated::<Path, Token![,]>::parse_terminated);
    if paths.len() > 1 {
        return Error::new_spanned(paths, "expected at most one crate path")
            .into_compile_error()
            .into();
    }
    let root = paths
        .into_iter()
        .next()
        .unwrap_or_else(|| parse_quote!(::splice));
    declaration::expand(root, item.into())
        .unwrap_or_else(Error::into_compile_error)
        .into()
}
