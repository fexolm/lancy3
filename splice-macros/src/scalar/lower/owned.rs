//! Run an ownership-sensitive scope in Rust with the actual scalar and slice operands.
use super::{Binding, Lower, Scalar};
use proc_macro2::TokenStream;
use syn::{FnArg, Signature, Type};

impl Lower<'_> {
    pub(super) fn owned_call(&self, signature: &Signature) -> Option<TokenStream> {
        if !matches!(self.result_type, Scalar::Int | Scalar::Bool) {
            return None;
        }
        let root = self.root;
        let scalar = emit!(#root::frontend::scalar);
        let mut words = Vec::new();
        let mut slices = Vec::new();
        let mut arguments = Vec::new();
        let names = super::super::abi::parameter_names(signature);
        for ((formal, binding), name) in signature.inputs.iter().zip(self.formals).zip(names) {
            let FnArg::Typed(formal) = formal else {
                return None;
            };
            match binding {
                Binding::Buffer(kind) => {
                    // A packet supplies a reborrow for this call. Only an independent
                    // input lifetime can be instantiated with that shorter lifetime.
                    let Type::Reference(reference) = formal.ty.as_ref() else {
                        return None;
                    };
                    if reference.lifetime.as_ref().is_some_and(|lt| {
                        lt.ident != "_"
                            && (lt.ident == "static"
                                || super::super::abi::has_early_bound_lifetime(
                                    &formal.ty,
                                    &signature.generics,
                                ))
                    }) {
                        return None;
                    }
                    let element = kind.elem.ty(root, self.types);
                    let writable = kind.write;
                    slices.push(
                        emit!(#root::hir::ContextSlice { buffer: #name, writable: #writable }),
                    );
                    arguments.push(if writable {
                        emit!(packet.write::<#element>())
                    } else {
                        emit!(packet.read::<#element>())
                    });
                }
                Binding::Reg(Scalar::Int | Scalar::Bool, _) | Binding::Host(_)
                    if matches!(
                        Scalar::from_type(&formal.ty, self.types),
                        Some(Scalar::Int | Scalar::Bool)
                    ) =>
                {
                    let index = words.len();
                    arguments.push(emit!(#scalar::ImportWord::decode(packet.word(#index))));
                    words.push(if matches!(binding, Binding::Host(_)) {
                        emit!(#scalar::Host(#name))
                    } else {
                        emit!(#name)
                    });
                }
                _ => return None,
            }
        }
        let name = self.name;
        let fish = &self.fish;
        if slices.is_empty() {
            return Some(self.import_call(&syn::parse_quote!(#name #fish), &words));
        }
        let generics = &signature.generics;
        let (decl, params, bounds) = generics.split_for_impl();
        let lifetimes = generics.lifetimes().map(|p| &p.lifetime);
        let types = generics.type_params().map(|p| &p.ident);
        let word_count = words.len() as u32;
        let slice_count = slices.len() as u32;
        Some(emit!({
            struct __SpliceScope #decl (::core::marker::PhantomData<fn() -> (#(&#lifetimes (),)* #(*const #types,)*)>) #bounds;
            impl #decl #root::StaticRustCall for __SpliceScope #params #bounds {
                const WORDS: u32 = #word_count;
                const SLICES: u32 = #slice_count;
                fn invoke(mut packet: #root::ContextArguments<'_>) -> i32 {
                    #scalar::ImportWord::encode(#name #fish (#(#arguments),*))
                }
            }
            let words = [#(#scalar::RunArg::word(#words, __splice_builder)?),*];
            __splice_builder.rust_call(
                #root::hir::RustTarget::Static(const { &#root::StaticRustSymbol::of::<__SpliceScope #params>(::core::stringify!(#name)) }),
                &words, &[#(#slices),*]
            )?
        }))
    }
}
