//! Rust attribute frontend for Splice kernels.
use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use syn::{Error, ItemFn, Path, Result, Type, parse_macro_input, parse_quote};

mod scalar;

/// Define a Splice kernel. `name::build::<Backend>(...)` stages its comptime
/// parameters into the selected backend. `name::ir(...)` builds portable IR.
#[proc_macro_attribute]
pub fn kernel(options: TokenStream, item: TokenStream) -> TokenStream {
    let function = parse_macro_input!(item as ItemFn);
    expand(options.into(), function)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn expand(options: Tokens, function: ItemFn) -> Result<Tokens> {
    use syn::parse::Parser;
    let paths =
        syn::punctuated::Punctuated::<Path, syn::Token![,]>::parse_terminated.parse2(options)?;
    if paths.len() > 1 {
        return Err(Error::new_spanned(paths, "expected at most one crate path"));
    }
    let root = paths
        .into_iter()
        .next()
        .unwrap_or_else(|| parse_quote!(::splice));
    scalar::expand(root, function)
}

fn scalar_type(ty: &Type, name: &str) -> bool {
    matches!(ty, Type::Path(p) if p.qself.is_none() && p.path.is_ident(name))
}
