//! Rust attribute frontend for Splice kernels.
use proc_macro::TokenStream;
use syn::{
    DeriveInput, Error, Expr, Meta, MetaNameValue, Path, Token, parse_macro_input, parse_quote,
    punctuated::Punctuated,
};

// Generated locals and labels share a hygiene context across fragments.
// Tokens interpolated from the kernel keep their original spans and
// lexical bindings.
macro_rules! emit {
    ($($tokens:tt)*) => { quote::quote_spanned!(proc_macro2::Span::mixed_site()=> $($tokens)*) };
}

mod api;
mod comptime_key;
mod declaration;
mod kast;
mod kernel_type;
mod scalar;

/// Derive a value key for a struct or enum used as a specialization parameter.
/// `#[splice(crate = path)]` names the Splice crate when it is not `::splice`.
#[proc_macro_derive(ComptimeKey, attributes(splice))]
pub fn comptime_key(item: TokenStream) -> TokenStream {
    comptime_key::expand(parse_macro_input!(item as syn::DeriveInput))
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Describe a Rust struct for Splice: the names of its fields, which a
/// comptime projection reads, and, when its fields are kernel data and it
/// is `Copy`, the layout rustc chose for it (`splice::KernelData`), so that
/// kernels can take it by value and in slices.
/// `#[kernel_type(transparent)]` (or `#[repr(transparent)]`) makes a
/// newtype over a kernel scalar that scalar; `#[kernel_type(pod)]` asserts
/// that the struct is kernel data.
/// `#[splice(crate = path)]` names the Splice crate when it is not `::splice`.
#[proc_macro_derive(KernelType, attributes(splice, kernel_type))]
pub fn kernel_type(item: TokenStream) -> TokenStream {
    let item = parse_macro_input!(item as DeriveInput);
    derive_root(&item.attrs)
        .and_then(|root| kernel_type::expand(item, &root))
        .unwrap_or_else(Error::into_compile_error)
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

/// Publish a free function, an inline module, or the methods of an `impl` or `trait`.
/// An inline module shares its imports across kernel signatures and bodies.
/// For a free function, `name::build::<Backend>(...)` stages its comptime
/// parameters into the selected backend. `name::ir(...)` builds portable IR.
/// `#[kernel(print)]` also publishes `name::print(...)`, which specializes
/// the Rust source without building IR; a caller prints a call to a kernel
/// without it as written.
///
/// A body Splice cannot lower to IR is a compile error; a kernel calls
/// ordinary Rust functions and methods, which stay Rust.
/// `#[kernel(crate = path)]` names the Splice crate when it is not
/// reachable as `::splice`.
#[proc_macro_attribute]
pub fn kernel(options: TokenStream, item: TokenStream) -> TokenStream {
    let options = parse_macro_input!(options with KernelOptions::parse);
    declaration::expand(options.root, options.print, item.into())
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

struct KernelOptions {
    root: Path,
    print: bool,
}

impl KernelOptions {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut root = None;
        let mut print = false;
        for option in Punctuated::<Meta, Token![,]>::parse_terminated(input)? {
            match &option {
                Meta::Path(path) if path.is_ident("print") && !print => print = true,
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
                        "expected `print` or `crate = path`, each at most once",
                    ));
                }
            }
        }
        Ok(Self {
            root: root.unwrap_or_else(|| parse_quote!(::splice)),
            print,
        })
    }
}
