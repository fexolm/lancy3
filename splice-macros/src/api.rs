//! The argument structures every published kernel shares, whether its body
//! is lowered to Splice IR or kept as a Rust call: `BuildArgs` holds the
//! `#[comptime]` parameters, `CallArgs` the runtime ones, and the entry
//! points (`build`, `ir`) take the generics the specialization needs.
use crate::scalar::abi::used_generics;
use proc_macro2::{Ident, TokenStream};
use syn::{GenericParam, Generics, Type, Visibility};

pub(crate) struct PublicApi {
    /// Generics of `BuildArgs`.
    pub(crate) host: Generics,
    /// Generics of `CallArgs`.
    pub(crate) call: Generics,
    /// Generics of `build` and `ir`: the declared ones with the elided
    /// lifetimes that `BuildArgs` borrows.
    pub(crate) entry: Generics,
    /// `BuildArgs` and `CallArgs`.
    pub(crate) items: TokenStream,
}

impl PublicApi {
    /// `declared` are the generics of the signature, `elided` the same with
    /// a named lifetime for each elided one (`abi::Elision`); the types of
    /// `host` and `call` use the elided generics.
    pub(crate) fn new(
        vis: &Visibility,
        declared: &Generics,
        elided: &Generics,
        host: &[(&Ident, Type)],
        call: &[(&Ident, Type)],
    ) -> Self {
        let host_generics = used_generics(elided, host.iter().map(|(_, ty)| ty));
        let call_generics = used_generics(elided, call.iter().map(|(_, ty)| ty));
        let (_, _, host_where) = host_generics.split_for_impl();
        let (_, _, call_where) = call_generics.split_for_impl();
        let host_fields = host.iter().map(|(name, ty)| emit!(#vis #name: #ty));
        let call_fields = call.iter().map(|(name, ty)| emit!(#vis #name: #ty));
        let items = emit! {
            #vis struct BuildArgs #host_generics #host_where { #(#host_fields,)* }
            #vis struct CallArgs #call_generics #call_where { #(#call_fields,)* }
        };
        let mut entry = declared.clone();
        entry.params = elided
            .params
            .iter()
            .filter(|p| match p {
                GenericParam::Lifetime(l) => {
                    host_generics.lifetimes().any(|h| h.lifetime == l.lifetime)
                        || declared.lifetimes().any(|h| h.lifetime == l.lifetime)
                }
                _ => true,
            })
            .cloned()
            .collect();
        Self {
            host: host_generics,
            call: call_generics,
            entry,
            items,
        }
    }
}
