//! Publish the declaring crate's choice of scalar or Rust entry to its callers.
use crate::scalar::methods::TraitMethodKind;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use std::collections::BTreeMap;
use syn::{Path, Visibility};

impl TraitMethodKind {
    fn tag(self) -> TokenStream {
        match self {
            Self::Word => quote!(word),
            Self::EvalOnly => quote!(eval),
            Self::Rust => quote!(opaque),
        }
    }
}

pub(super) fn definition(
    name: &Ident,
    vis: &Visibility,
    gates: &[TokenStream],
    methods: Vec<(Ident, TraitMethodKind, Vec<TokenStream>)>,
) -> TokenStream {
    // macro_export places every public selector at the crate root, including
    // selectors for identically named traits declared in different modules.
    // The declaration's source position tells them apart and keeps the
    // expansion deterministic.
    let symbol = format_ident!(
        "__splice_trait_{}_{:016x}",
        name.to_string().trim_start_matches("r#"),
        position_hash(name.span())
    );
    let export = matches!(vis, Visibility::Public(_)).then(|| quote!(#[macro_export]));
    let mut groups = BTreeMap::<Ident, Vec<_>>::new();
    for (method, kind, cfg) in methods {
        groups.entry(method).or_default().push((kind, cfg));
    }
    let mut helpers = TokenStream::new();
    let mut arms = TokenStream::new();
    let cfg_scope =
        quote!(mod __splice_trait_cfg { #[allow(unused_imports)] pub(super) use $crate::*; });
    for (method, variants) in groups {
        let first = variants[0].0;
        if variants.iter().all(|(kind, _)| *kind == first) {
            let tag = first.tag();
            arms.extend(quote! {
                (@$mode:ident #method, $pick:ident, $args:tt) => { $pick! { #tag $args } };
            });
            continue;
        }
        // cfg applies in the declaring crate, even if the caller's features differ.
        let helper = format_ident!("{}_{}", symbol, method.to_string().trim_start_matches("r#"));
        for (kind, cfg) in variants {
            let tag = kind.tag();
            helpers.extend(quote! {
                #(#[#gates])* #(#[#cfg])* #[doc(hidden)] #[macro_export]
                macro_rules! #helper { ($pick:ident, $args:tt) => { $pick! { #tag $args } }; }
            });
        }
        arms.extend(quote! {
            (@expr #method, $pick:ident, $args:tt) => {{
                #cfg_scope
                __splice_trait_cfg::#helper! { $pick, $args }
            }};
            (@items #method, $pick:ident, $args:tt) => { __splice_trait_cfg::#helper! { $pick, $args } };
        });
    }
    let impl_scope = (!helpers.is_empty()).then_some(cfg_scope);
    quote! {
        #helpers
        #(#[#gates])* #[doc(hidden)] #export
        macro_rules! #symbol {
            #arms
            (@implementation $body:expr) => {{ #impl_scope $body }};
            ($published:expr, $ordinary:expr) => { $published };
        }
        #(#[#gates])* #[doc(hidden)] #[allow(unused_imports)]
        #vis use #symbol as #name;
    }
}

pub(crate) fn trait_method_eval_dispatch(
    root: &Path,
    owner: &Path,
    method: &Ident,
    evaluate: TokenStream,
    opaque: TokenStream,
) -> TokenStream {
    super::select_trait(
        root,
        owner,
        quote!(@expr #method, __splice_method_case, (evaluate #evaluate, #opaque)),
    )
}

pub(crate) fn trait_method_dispatch(
    root: &Path,
    owner: &Path,
    method: &Ident,
    scalar: TokenStream,
    opaque: TokenStream,
    ordinary: TokenStream,
) -> TokenStream {
    super::select_trait(
        root,
        owner,
        quote!(@expr #method, __splice_method_case, (call #scalar, #opaque, #ordinary)),
    )
}

/// FNV-1a of the file, line and column of `span`.
fn position_hash(span: proc_macro2::Span) -> u64 {
    let (file, line, column) = if proc_macro::is_available() {
        let span = span.unwrap();
        (span.file(), span.line(), span.column())
    } else {
        (String::new(), 0, 0)
    };
    let position = format!("{file}:{line}:{column}");
    position.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}
