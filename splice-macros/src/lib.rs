//! Rust attribute frontend for Splice kernels.
use proc_macro::TokenStream;
use syn::{Error, ItemFn, Path, Token, parse_macro_input, parse_quote, punctuated::Punctuated};

mod comptime_key;
mod scalar;

/// Derive a value key for a struct or enum used as a specialization parameter.
#[proc_macro_derive(ComptimeKey, attributes(splice))]
pub fn comptime_key(item: TokenStream) -> TokenStream {
    comptime_key::expand(parse_macro_input!(item as syn::DeriveInput))
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Define a Splice kernel. `name::build::<Backend>(...)` stages its comptime
/// parameters into the selected backend. `name::ir(...)` builds portable IR.
/// `name::print(...)` specializes the Rust source without building IR.
#[proc_macro_attribute]
pub fn kernel(options: TokenStream, item: TokenStream) -> TokenStream {
    let function = parse_macro_input!(item as ItemFn);
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
    scalar::expand(root, function)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}
