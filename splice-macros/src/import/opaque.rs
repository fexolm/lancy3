//! Give opaque Rust return types their own call declaration. An associated
//! type cannot name `impl Trait`, and the returned value may borrow CallArgs.
use proc_macro2::TokenStream;
use syn::visit::{self, Visit};
use syn::visit_mut::{self, VisitMut};
use syn::{
    CapturedParam, GenericParam, Generics, Path, Type, TypeImplTrait, TypeParamBound, Visibility,
};

use crate::scalar::abi::used_generics;

pub(super) fn contains(ty: &Type) -> bool {
    struct ContainsOpaque(bool);
    impl Visit<'_> for ContainsOpaque {
        fn visit_type_impl_trait(&mut self, _: &TypeImplTrait) {
            self.0 = true;
        }
        fn visit_type_bare_fn(&mut self, _: &syn::TypeBareFn) {}
    }
    let mut contains = ContainsOpaque(false);
    visit::visit_type(&mut contains, ty);
    contains.0
}

pub(super) struct Interface<'a> {
    pub(super) root: &'a Path,
    pub(super) vis: &'a Visibility,
    pub(super) signature: &'a TokenStream,
    pub(super) environment: &'a TokenStream,
    pub(super) declared: &'a Generics,
    pub(super) invocation: &'a Generics,
    pub(super) arguments: &'a Generics,
    pub(super) output: &'a mut Type,
    pub(super) body: &'a TokenStream,
}

impl Interface<'_> {
    pub(super) fn declarations(&mut self) -> TokenStream {
        let Self {
            root,
            vis,
            signature,
            environment,
            declared,
            invocation,
            arguments,
            output,
            body,
        } = self;
        let type_uses: Vec<Type> = declared.params.iter().filter_map(parameter_type).collect();
        let context = used_generics(declared, type_uses.iter());
        let (context_params, _, context_where) = context.split_for_impl();
        let mut call = invocation.clone();
        call.params = invocation
            .lifetimes()
            .cloned()
            .map(GenericParam::Lifetime)
            .collect();
        let (call_params, _, call_where) = call.split_for_impl();
        let (_, call_arguments, _) = arguments.split_for_impl();
        let lifetimes = call
            .lifetimes()
            .map(|param| CapturedParam::Lifetime(param.lifetime.clone()));
        let generics = context.params.iter().filter_map(|param| match param {
            GenericParam::Type(param) => Some(CapturedParam::Ident(param.ident.clone())),
            GenericParam::Const(param) => Some(CapturedParam::Ident(param.ident.clone())),
            GenericParam::Lifetime(_) => None,
        });
        let captures = lifetimes.chain(generics).collect();
        PreserveCaptures(captures).visit_type_mut(output);
        let output = &**output;

        emit! {
            #[doc(hidden)]
            #vis struct __SpliceOpaqueIr<S, E>(#root::RustImport<S, E>);

            #[doc(hidden)]
            #vis struct OpaqueKernel<K>(K);

            impl<S, E, B: #root::Codegen> #root::CompileInput<B> for __SpliceOpaqueIr<S, E>
            where B::Output: #root::ImportTarget {
                type Output = OpaqueKernel<<B::Output as #root::ImportTarget>::Linked<S, E>>;
                type Error = #root::BuildError;

                fn compile_with(&self, options: ::core::option::Option<#root::BuildOptions>)
                    -> ::core::result::Result<Self::Output, Self::Error>
                {
                    <#root::RustImport<S, E> as #root::CompileInput<B>>::compile_with(&self.0, options)
                        .map(OpaqueKernel)
                }
            }

            impl<K> ::core::ops::Deref for OpaqueKernel<K> {
                type Target = K;
                fn deref(&self) -> &K { &self.0 }
            }

            impl<K: ::core::fmt::Display> ::core::fmt::Display for OpaqueKernel<K> {
                fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    ::core::fmt::Display::fmt(&self.0, formatter)
                }
            }

            impl #context_params OpaqueKernel<#root::LinkedImport<#signature, #environment>> #context_where {
                #vis fn call #call_params (&self, __splice_arguments: CallArgs #call_arguments)
                    -> ::core::result::Result<#output, #root::CallError> #call_where
                {
                    self.0.with_environment(move |__splice_environment| { #body })
                }
            }
        }
    }
}

fn parameter_type(param: &GenericParam) -> Option<Type> {
    let name = match param {
        GenericParam::Type(param) => &param.ident,
        GenericParam::Const(param) => &param.ident,
        GenericParam::Lifetime(_) => return None,
    };
    Some(syn::parse_quote!(#name))
}

struct PreserveCaptures(Vec<CapturedParam>);

impl VisitMut for PreserveCaptures {
    fn visit_type_impl_trait_mut(&mut self, ty: &mut TypeImplTrait) {
        visit_mut::visit_type_impl_trait_mut(self, ty);
        if ty
            .bounds
            .iter()
            .all(|bound| !matches!(bound, TypeParamBound::PreciseCapture(_)))
        {
            let params = &self.0;
            // Keep the source argument lifetimes. The borrow of the generated
            // kernel receiver must not become part of the opaque result type.
            ty.bounds.push(syn::parse_quote!(use<#(#params),*>));
        }
    }
}
