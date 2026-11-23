//! Rust attribute frontend for Splice kernels.
use proc_macro::TokenStream;
use syn::{
    Data, DeriveInput, Error, Expr, Index, Member, Meta, MetaNameValue, Path, Token,
    parse_macro_input, parse_quote, punctuated::Punctuated,
};

mod comptime_key;
mod declaration;
mod import;
mod overflow;
mod scalar;

/// Derive a value key for a struct or enum used as a specialization parameter.
/// `#[splice(crate = path)]` names the Splice crate when it is not `::splice`.
#[proc_macro_derive(ComptimeKey, attributes(splice))]
pub fn comptime_key(item: TokenStream) -> TokenStream {
    comptime_key::expand(parse_macro_input!(item as syn::DeriveInput))
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Describe the owned fields of a Rust structure for Splice's typed IR.
/// `#[splice(crate = path)]` names the Splice crate when it is not `::splice`.
#[proc_macro_derive(KernelType, attributes(splice))]
pub fn kernel_type(item: TokenStream) -> TokenStream {
    let item = parse_macro_input!(item as DeriveInput);
    let root = match derive_root(&item.attrs) {
        Ok(root) => root,
        Err(error) => return error.into_compile_error().into(),
    };
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
        quote::quote!(#root::KernelField {
            name: #label,
            offset: ::core::mem::offset_of!(Self, #member),
            type_name: ::core::any::type_name::<#ty>,
        })
    });
    quote::quote! {
        impl #params #root::KernelType for #name #args #where_ {
            const FIELDS: &'static [#root::KernelField] = &[#(#fields),*];
        }
    }
    .into()
}

/// The Splice crate path a derive names with `#[splice(crate = path)]`.
fn derive_root(attrs: &[syn::Attribute]) -> syn::Result<Path> {
    let mut root = None;
    for attr in attrs.iter().filter(|a| a.path().is_ident("splice")) {
        attr.parse_nested_meta(|meta| {
            if !meta.path.is_ident("crate") || root.is_some() {
                return Err(meta.error("expected `crate = path`, at most once"));
            }
            root = Some(meta.value()?.parse()?);
            Ok(())
        })?;
    }
    Ok(root.unwrap_or_else(|| parse_quote!(::splice)))
}

/// Publish a free function or all methods of an `impl` or `trait` block.
/// For a free function, `name::build::<Backend>(...)` stages its comptime
/// parameters into the selected backend. `name::ir(...)` builds portable IR.
/// `name::print(...)` specializes the Rust source without building IR.
///
/// A body Splice cannot lower is a compile error. `#[kernel(rust)]` (on the
/// function, or on a method inside a `#[kernel]` block) publishes the same API
/// around the unchanged Rust body. `#[kernel(crate = path)]` names the Splice
/// crate when it is not reachable as `::splice`.
#[proc_macro_attribute]
pub fn kernel(options: TokenStream, item: TokenStream) -> TokenStream {
    let options = parse_macro_input!(options with KernelOptions::parse);
    declaration::expand(options.root, options.lowering, item.into())
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

struct KernelOptions {
    root: Path,
    lowering: import::Lowering,
}

impl KernelOptions {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut root = None;
        let mut lowering = None;
        for option in Punctuated::<Meta, Token![,]>::parse_terminated(input)? {
            match &option {
                Meta::Path(path) if path.is_ident("rust") && lowering.is_none() => {
                    lowering = Some(import::Lowering::Rust);
                }
                Meta::NameValue(MetaNameValue {
                    path,
                    value: Expr::Path(value),
                    ..
                }) if path.is_ident("crate") && root.is_none() && value.qself.is_none() => {
                    root = Some(value.path.clone());
                }
                _ => {
                    return Err(Error::new_spanned(
                        option,
                        "expected `rust` or `crate = path`, each at most once",
                    ));
                }
            }
        }
        Ok(Self {
            root: root.unwrap_or_else(|| parse_quote!(::splice)),
            lowering: lowering.unwrap_or(import::Lowering::Ir),
        })
    }
}
