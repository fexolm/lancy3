//! Rust attribute frontend for Splice kernels.
use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use syn::{Error, ItemFn, Path, Result, Type, parse_macro_input, parse_quote};

mod scalar;

/// Define a Splice kernel. `name::ir(...)` stages the comptime parameters and
/// `name::build::<Backend>(...)` also compiles the result.
#[proc_macro_attribute]
pub fn kernel(options: TokenStream, item: TokenStream) -> TokenStream {
    let function = parse_macro_input!(item as ItemFn);
    dispatch(options.into(), function, false)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// Lancy's original native builder API, re-exported by the `lancy` crate.
#[proc_macro_attribute]
pub fn lancy_kernel(options: TokenStream, item: TokenStream) -> TokenStream {
    let function = parse_macro_input!(item as ItemFn);
    dispatch(options.into(), function, true)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KernelApi {
    Lancy,
    Splice,
}

fn dispatch(options: Tokens, function: ItemFn, lancy: bool) -> Result<Tokens> {
    use syn::parse::Parser;
    let paths =
        syn::punctuated::Punctuated::<Path, syn::Token![,]>::parse_terminated.parse2(options)?;
    let mut root: Path = if lancy {
        parse_quote!(::lancy)
    } else {
        parse_quote!(::splice)
    };
    if paths.len() > 1 {
        return Err(Error::new_spanned(paths, "expected at most one crate path"));
    }
    if let Some(path) = paths.into_iter().next() {
        root = path;
    }
    let api = if lancy {
        KernelApi::Lancy
    } else {
        KernelApi::Splice
    };
    scalar::expand(root, function, api)
}

fn scalar_type(ty: &Type, name: &str) -> bool {
    matches!(ty, Type::Path(p) if p.qself.is_none() && p.path.is_ident(name))
}
