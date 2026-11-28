//! `#[derive(KernelType)]`: the field names a comptime projection reads,
//! and the layout of a struct a kernel keeps in memory.
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Error, Member, Path, Result};

/// What `#[kernel_type(...)]` and `#[repr(...)]` say about the struct.
#[derive(Default)]
struct Options {
    transparent: bool,
    pod: bool,
    packed: bool,
}

impl Options {
    fn parse(item: &DeriveInput) -> Result<Self> {
        let mut options = Self::default();
        for attr in &item.attrs {
            if attr.path().is_ident("kernel_type") {
                attr.parse_nested_meta(|meta| {
                    let name = meta.path.get_ident().map(ToString::to_string);
                    match name.as_deref() {
                        Some("transparent") => options.transparent = true,
                        Some("pod") => options.pod = true,
                        Some(other @ ("soa" | "align" | "readonly" | "noalias" | "niche")) => {
                            return Err(meta
                                .error(format!("`#[kernel_type({other})]` is not supported yet")));
                        }
                        _ => {
                            return Err(meta.error(
                                "expected `#[kernel_type(transparent)]` or `#[kernel_type(pod)]`",
                            ));
                        }
                    }
                    // Skip arguments such as `align(16)`.
                    if meta.input.peek(syn::token::Paren) {
                        let _: TokenStream = meta.input.parse()?;
                    }
                    Ok(())
                })?;
            } else if attr.path().is_ident("repr") {
                // Other representations keep the layout rustc reports.
                let _ = attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("transparent") {
                        options.transparent = true;
                    } else if meta.path.is_ident("packed") {
                        options.packed = true;
                    }
                    if meta.input.peek(syn::token::Paren) {
                        let _: TokenStream = meta.input.parse()?;
                    }
                    Ok(())
                });
            }
        }
        Ok(options)
    }
}

pub(crate) fn expand(item: DeriveInput, root: &Path) -> Result<TokenStream> {
    let options = Options::parse(&item)?;
    let Data::Struct(data) = &item.data else {
        return Err(Error::new_spanned(
            &item.ident,
            "KernelType can only be derived for structs",
        ));
    };
    let name = &item.ident;
    let (params, args, where_) = item.generics.split_for_impl();
    let members: Vec<Member> = data
        .fields
        .iter()
        .enumerate()
        .map(|(index, field)| match &field.ident {
            Some(ident) => Member::Named(ident.clone()),
            None => Member::Unnamed(index.into()),
        })
        .collect();
    let names: Vec<String> = members
        .iter()
        .map(|member| match member {
            Member::Named(ident) => ident.to_string().trim_start_matches("r#").to_owned(),
            Member::Unnamed(index) => index.index.to_string(),
        })
        .collect();
    let fields = quote! {
        impl #params #root::KernelType for #name #args #where_ {
            const FIELDS: &'static [&'static str] = &[#(#names),*];
        }
    };
    if options.transparent && data.fields.len() != 1 {
        return Err(Error::new_spanned(
            name,
            "a `transparent` kernel type has exactly one field",
        ));
    }
    // A struct that borrows is not kernel data; comptime projections still
    // read its fields.
    if item.generics.lifetimes().next().is_some() {
        if options.pod || options.transparent {
            return Err(Error::new_spanned(
                &item.generics,
                "kernel data cannot borrow: a `pod` or `transparent` kernel type has no lifetime parameters",
            ));
        }
        return Ok(fields);
    }
    let hir = quote!(#root::hir);
    let types: Vec<_> = data.fields.iter().map(|field| &field.ty).collect();
    // `for<'__splice>` keeps a bound on a concrete field type from being an
    // error when it does not hold: the struct is then just not kernel data,
    // which a kernel that uses it as data reports.
    let mut generics = item.generics.clone();
    let clause = generics.make_where_clause();
    for ty in &types {
        clause
            .predicates
            .push(syn::parse_quote!(for<'__splice> #ty: #root::KernelData));
    }
    clause
        .predicates
        .push(syn::parse_quote!(for<'__splice> Self: ::core::marker::Copy));
    let (_, _, data_where) = generics.split_for_impl();
    let value = if options.packed {
        quote!(::core::panic!(
            "a repr(packed) struct is not supported as kernel data yet"
        ))
    } else if options.transparent {
        let ty = types[0];
        quote! {
            match <#ty as #root::KernelData>::TYPE {
                #hir::DataType::Scalar(scalar) => #hir::DataType::Scalar(scalar),
                #hir::DataType::Struct(_) => ::core::panic!(
                    "a transparent kernel type wraps a kernel scalar"
                ),
            }
        }
    } else {
        quote! {
            #hir::DataType::Struct(&#hir::StructType {
                name: ::core::stringify!(#name),
                size: ::core::mem::size_of::<Self>(),
                align: ::core::mem::align_of::<Self>(),
                fields: &[#(#hir::Field {
                    name: #names,
                    offset: ::core::mem::offset_of!(Self, #members),
                    ty: <#types as #root::KernelData>::TYPE,
                }),*],
            })
        }
    };
    // `pod` asserts that the struct is kernel data where it is declared.
    let pod = (options.pod || options.transparent).then(|| {
        quote! {
            const _: () = {
                fn __splice_kernel_data<T: #root::KernelData + ?::core::marker::Sized>() {}
                fn __splice_check #params () #where_ { __splice_kernel_data::<#name #args>(); }
            };
        }
    });
    Ok(quote! {
        #fields
        impl #params #root::KernelData for #name #args #data_where {
            const TYPE: #hir::DataType = #value;
        }
        #pod
    })
}
