//! Rust attribute frontend for scalar kernels and explicit batch projections.
use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use syn::{Error, ItemFn, Path, Result, Type, parse_macro_input, parse_quote};

mod batch;
mod scalar;

/// Build a scalar JIT function. Use `#[kernel(batch)]` for a column projection.
/// Only `#[comptime]` parameters are passed to the outer builder function.
#[proc_macro_attribute]
pub fn kernel(options: TokenStream, item: TokenStream) -> TokenStream {
    let function = parse_macro_input!(item as ItemFn);
    dispatch(options.into(), function)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn dispatch(options: Tokens, function: ItemFn) -> Result<Tokens> {
    use syn::parse::Parser;
    let paths =
        syn::punctuated::Punctuated::<Path, syn::Token![,]>::parse_terminated.parse2(options)?;
    let mut root: Path = parse_quote!(::lancy_jit);
    let mut batch = false;
    for path in paths {
        if path.is_ident("batch") {
            batch = true;
        } else {
            root = path;
        }
    }
    if batch {
        batch::expand(root, function)
    } else {
        scalar::expand(root, function)
    }
}

fn scalar_type(ty: &Type, name: &str) -> bool {
    matches!(ty, Type::Path(p) if p.qself.is_none() && p.path.is_ident(name))
}
