//! Publish the declaring crate's choice of scalar or Rust entry to its callers.
use crate::scalar::methods::TraitMethodKind;
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use syn::{Path, Visibility};

impl TraitMethodKind {
    fn tag(self) -> TokenStream {
        match self {
            Self::Word => quote!(word),
            Self::Bool => quote!(boolean),
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
    // selectors for identically named traits generated in different modules.
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let symbol = format_ident!(
        "__splice_trait_{}_{}",
        name.to_string().trim_start_matches("r#"),
        NEXT.fetch_add(1, Ordering::Relaxed)
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
    boolean: TokenStream,
    opaque: TokenStream,
    ordinary: TokenStream,
) -> TokenStream {
    super::select_trait(
        root,
        owner,
        quote!(@expr #method, __splice_method_case, (call #scalar, #boolean, #opaque, #ordinary)),
    )
}
