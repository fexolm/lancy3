//! Default trait entries, scalar signatures and compile-time parameter masks.
use super::{Binding, Scalar, abi, entry_name, normalize, parameter};
use proc_macro2::{Ident, TokenStream};
use quote::format_ident;
use syn::{FnArg, GenericParam, ReturnType, Signature, TraitItem, Type};

pub(crate) fn trait_defaults(root: &syn::Path, signature: &Signature) -> Vec<TraitItem> {
    let mut signature = normalize(signature, None);
    signature
        .generics
        .make_where_clause()
        .predicates
        .push(syn::parse_quote!(Self: Sized));
    let types: Vec<_> = signature.generics.type_params().map(|t| &t.ident).collect();
    let result_type = match &signature.output {
        ReturnType::Default => emit!(()),
        ReturnType::Type(_, ty)
            if matches!(&**ty, Type::Tuple(tuple) if tuple.elems.is_empty())
                || matches!(
                    Scalar::from_type(ty, &types),
                    Some(Scalar::Int | Scalar::Bool)
                ) =>
        {
            emit!(#ty)
        }
        _ => return Vec::new(),
    };
    let Ok(params) = signature
        .inputs
        .iter()
        .zip(abi::parameter_names(&signature))
        .map(|(argument, name)| parameter(argument, &types, name))
        .collect::<syn::Result<Vec<_>>>()
    else {
        return Vec::new();
    };
    let mut declarations = Vec::new();
    let mut args = Vec::new();
    let mut stage_checks = Vec::new();
    let support = emit!(#root::frontend::scalar);
    let hir = emit!(#root::hir);
    let mut materializable = true;
    for param in &params {
        let id = &param.name;
        let ty = param.ty;
        declarations.push(match param.binding {
            Binding::Host(_)
                if matches!(
                    Scalar::from_type(ty, &types),
                    Some(Scalar::Int | Scalar::Bool | Scalar::Index)
                ) =>
            {
                if matches!(Scalar::from_type(ty, &types), Some(Scalar::Index)) {
                    materializable = false;
                }
                stage_checks
                    .push(emit!(let #id = #support::Host(#support::KnownArg::require(#id)?);));
                emit!(#id: impl #support::KnownArg<#ty>)
            }
            Binding::Host(_) => {
                materializable = false;
                emit!(#id: #support::Host<#ty>)
            }
            Binding::Reg(Scalar::Int | Scalar::Bool, _) => emit!(#id: impl #support::RunArg),
            Binding::Buffer(_) => {
                materializable = false;
                emit!(#id: #hir::Buffer)
            }
            _ => return Vec::new(),
        });
        args.push(id);
    }
    let operand_types = params.iter().map(|p| p.ty);
    let name = &signature.ident;
    let generics = &signature.generics;
    let where_ = &generics.where_clause;
    let stage_name = entry_name("stage_method", name);
    let eval_name = entry_name("eval_method", name);
    let adapter_name = entry_name("trait_adapter", name);
    let mut items = Vec::new();
    let arity = args.len();
    let body = if materializable {
        let generic_args: Vec<_> = generics
            .params
            .iter()
            .filter_map(|p| match p {
                GenericParam::Type(p) => Some(&p.ident),
                GenericParam::Const(p) => Some(&p.ident),
                GenericParam::Lifetime(_) => None,
            })
            .collect();
        let func = (!generic_args.is_empty()).then(|| emit!(::<#(#generic_args),*>));
        let native_call = |words: &[TokenStream]| {
            let args = words
                .iter()
                .map(|word| emit!(#support::ImportWord::decode(#word)));
            emit!(Self::#name #func (#(#args),*))
        };
        let (adapter, call) = if arity <= 2 {
            let native = native_call(&[emit!(left), emit!(right)][..arity]);
            (
                emit! {
                    #[doc(hidden)]
                    #[allow(unused_variables)]
                    extern "C" fn #adapter_name #generics (left: i32, right: i32) -> i64 #where_ {
                        #support::invoke_import(|| #native)
                    }
                },
                emit! {
                    __splice_builder.rust_call(#hir::RustCall {
                        symbol: &#hir::RustSymbol {name: stringify!(#name), resolve: || Self::#adapter_name #func},
                        arity: #arity as u8,
                    }, &args)
                },
            )
        } else {
            let words = (0..arity)
                .map(|index| emit!(words[#index]))
                .collect::<Vec<_>>();
            let native = native_call(&words);
            (
                emit! {
                    #[doc(hidden)]
                    fn #adapter_name #generics (words: &[i32]) -> i32 #where_ {
                        #support::ImportWord::encode(#native)
                    }
                },
                emit! {
                    __splice_builder.rust_packet_call(&#hir::RustPacketSymbol {
                        name: stringify!(#name),
                        arity: #arity as u32,
                        resolve: || Self::#adapter_name #func,
                    }, &args)
                },
            )
        };
        items.push(syn::parse2(adapter).expect("a generated Rust trait adapter"));
        emit! {
            let args = [#(#support::RunArg::reg(#args, __splice_builder)),*];
            #call
        }
    } else {
        emit!(Err(
            "trait method requires Rust arguments that cannot cross the scalar ABI".into()
        ))
    };
    items.push(syn::parse2(emit! {
        #[doc(hidden)]
        #[allow(unused_variables, clippy::too_many_arguments)]
        fn #stage_name #generics (#(#declarations,)* __splice_builder: &mut dyn #root::frontend::KernelBuilder)
            -> #root::Result<#hir::Reg> #where_
        {
            #(#stage_checks)*
            #body
        }
    }).expect("a generated trait method stage"));
    items.push(syn::parse_quote! {
        #[doc(hidden)]
        #[allow(clippy::type_complexity)]
        fn #eval_name #generics (_: (#(Option<#operand_types>,)*)) -> #root::Result<#result_type> #where_ {
            Err("a trait method without a published implementation is opaque".into())
        }
    });
    items
}

/// A fixed signature allows a trait bound to select the companion without
/// recovering the declaration's argument types at a call site.
pub(crate) fn word_trait(root: &syn::Path, signature: &Signature) -> Vec<TraitItem> {
    let Some(kind) = trait_method(signature) else {
        return Vec::new();
    };
    let word = emit!(#root::frontend::scalar::Word);
    let encode = emit!(#root::frontend::scalar::ImportWord);
    let types: Vec<_> = signature
        .inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Typed(arg)
                if matches!(
                    Scalar::from_type(&arg.ty, &[]),
                    Some(Scalar::Int | Scalar::Bool)
                ) =>
            {
                Some(&arg.ty)
            }
            _ => None,
        })
        .collect();
    let mut items = trait_defaults(root, signature);
    if items.is_empty() {
        return items;
    }
    items.push(syn::parse2(stage_shape(signature)).expect("a generated trait stage shape"));
    let name = &signature.ident;
    let marker = kind.marker(name);
    let stage = entry_name("stage_method", name);
    let stage_words = entry_name("stage_trait_words", name);
    let eval = entry_name("eval_method", name);
    let eval_words = entry_name("eval_trait_words", name);
    let arguments: Vec<_> = (0..types.len())
        .map(|n| format_ident!("argument_{n}"))
        .collect();
    let decoded = arguments
        .iter()
        .zip(types)
        .map(|(arg, ty)| emit!(#arg.map(<#ty as #encode>::decode)));
    items.push(
        syn::parse2(emit! {
            #[doc(hidden)]
            fn #marker() -> bool where Self: Sized {
                true
            }
        })
        .expect("a generated trait publication marker"),
    );
    if kind != TraitMethod::Scalar {
        return items;
    }
    items.push(syn::parse2(emit! {
        #[doc(hidden)]
        fn #stage_words(__splice_words: &[#word], __splice_builder: &mut dyn #root::frontend::KernelBuilder)
            -> #root::Result<#root::hir::Reg> where Self: Sized
        {
            let [#(#arguments),*] = __splice_words else {
                return Err("wrong number of arguments for a published trait method".into());
            };
            Self::#stage(#(*#arguments,)* __splice_builder)
        }
    }).expect("a generated trait word stage"));
    items.push(
        syn::parse2(emit! {
            #[doc(hidden)]
            fn #eval_words(__splice_words: &[Option<i32>]) -> #root::Result<i32> where Self: Sized {
                let [#(#arguments),*] = __splice_words else {
                    return Err("wrong number of arguments for a published trait method".into());
                };
                Self::#eval((#(#decoded,)* )).map(#encode::encode)
            }
        })
        .expect("a generated trait word evaluator"),
    );
    items
}

pub(crate) fn individual_trait_supported(signature: &Signature) -> bool {
    trait_method(signature).is_some()
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TraitMethod {
    Scalar,
    Borrowed,
    Owned,
}

impl TraitMethod {
    pub(super) fn marker(self, name: &Ident) -> Ident {
        let kind = match self {
            Self::Scalar => "word_trait",
            Self::Borrowed => "borrowed_trait",
            Self::Owned => "owned_trait",
        };
        entry_name(kind, name)
    }
}

pub(super) fn trait_method(signature: &Signature) -> Option<TraitMethod> {
    let (kind, start) = match signature.receiver() {
        None => (TraitMethod::Scalar, 0),
        Some(receiver)
            if receiver.reference.is_some()
                && receiver.mutability.is_none()
                && receiver.colon_token.is_none()
                && receiver
                    .attrs
                    .iter()
                    .any(|attr| attr.path().is_ident("comptime")) =>
        {
            (TraitMethod::Borrowed, 1)
        }
        Some(receiver)
            if receiver.reference.is_none()
                && receiver.colon_token.is_none()
                && receiver
                    .attrs
                    .iter()
                    .any(|attr| attr.path().is_ident("comptime")) =>
        {
            (TraitMethod::Owned, 1)
        }
        _ => return None,
    };
    (signature.generics.params.is_empty()
        && signature.inputs.iter().skip(start).all(|arg| {
            matches!(arg, FnArg::Typed(arg) if matches!(Scalar::from_type(&arg.ty, &[]), Some(Scalar::Int | Scalar::Bool)))
        })
        && match &signature.output {
            ReturnType::Default => true,
            ReturnType::Type(_, ty) => {
                matches!(&**ty, Type::Tuple(tuple) if tuple.elems.is_empty())
                    || matches!(Scalar::from_type(ty, &[]), Some(Scalar::Int | Scalar::Bool))
            }
        })
    .then_some(kind)
}

pub(crate) fn stage_shape(signature: &Signature) -> TokenStream {
    let method = entry_name("stage_shape", &signature.ident);
    let stages = parameter_stages(signature);
    let masks = stage_masks(&stages);
    emit! {
        #[doc(hidden)]
        #[allow(clippy::type_complexity)]
        fn #method() -> (#([(); #stages],)*) where Self: Sized {
            (#(#masks,)*)
        }
    }
}

pub(super) fn stage_masks(stages: &[usize]) -> impl Iterator<Item = TokenStream> + '_ {
    stages
        .iter()
        .map(|stage| if *stage == 0 { emit!([]) } else { emit!([()]) })
}

pub(super) fn parameter_stages(signature: &Signature) -> Vec<usize> {
    signature
        .inputs
        .iter()
        .map(|arg| {
            let attrs = match arg {
                FnArg::Receiver(arg) => &arg.attrs,
                FnArg::Typed(arg) => &arg.attrs,
            };
            usize::from(attrs.iter().any(|attr| attr.path().is_ident("comptime")))
        })
        .collect()
}
