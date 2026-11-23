//! Default trait entries, scalar signatures and compile-time parameter masks.
use super::{Binding, Scalar, abi, entry_name, normalize, parameter};
use proc_macro2::TokenStream;
use syn::{FnArg, GenericParam, ReturnType, Signature, Type};

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum TraitMethodKind {
    Word,
    Bool,
    EvalOnly,
    Rust,
}

impl TraitMethodKind {
    pub(crate) fn for_result(output: &ReturnType) -> Self {
        match output {
            ReturnType::Default => Self::Word,
            ReturnType::Type(_, ty) => match Scalar::from_type(ty, &[]) {
                Some(Scalar::Int) => Self::Word,
                Some(Scalar::Bool) => Self::Bool,
                _ if matches!(ty.as_ref(), Type::Tuple(tuple) if tuple.elems.is_empty()) => {
                    Self::Word
                }
                _ if primitive_type(ty) => Self::EvalOnly,
                _ => Self::Rust,
            },
        }
    }
}

pub(in crate::scalar) fn primitive_type(ty: &Type) -> bool {
    matches!(ty, Type::Path(path) if path.qself.is_none() && path.path.get_ident().is_some_and(|name| {
        matches!(name.to_string().as_str(),
            "bool" | "char" | "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
            | "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "f32" | "f64")
    }))
}

pub(crate) fn trait_defaults(
    root: &syn::Path,
    signature: &Signature,
    trait_owner: &TokenStream,
) -> (TraitMethodKind, Vec<TokenStream>) {
    let kind = TraitMethodKind::for_result(&signature.output);
    if kind == TraitMethodKind::Rust {
        return (kind, Vec::new());
    }
    let mut signature = normalize(signature, None);
    signature
        .generics
        .make_where_clause()
        .predicates
        .push(syn::parse_quote!(Self: Sized));
    let types: Vec<_> = signature.generics.type_params().map(|t| &t.ident).collect();
    let result_type = match &signature.output {
        ReturnType::Default => emit!(()),
        ReturnType::Type(_, ty) => emit!(#ty),
    };
    let name = &signature.ident;
    let generics = &signature.generics;
    let where_ = &generics.where_clause;
    let eval_name = entry_name("eval_method", name);
    let operand_types = signature.inputs.iter().filter_map(|arg| match arg {
        FnArg::Typed(arg) => Some(arg.ty.as_ref()),
        FnArg::Receiver(_) => None,
    });
    let evaluator = emit! {
        #[doc(hidden)]
        #[allow(clippy::type_complexity)]
        fn #eval_name #generics (_: (#(::core::option::Option<#operand_types>,)*)) -> #root::Result<#result_type> #where_ {
            ::core::result::Result::Err(::core::convert::Into::into("a trait method without a published implementation is opaque"))
        }
    };
    if kind == TraitMethodKind::EvalOnly {
        return (kind, vec![evaluator]);
    }
    let Ok(params) = signature
        .inputs
        .iter()
        .zip(abi::parameter_names(&signature))
        .map(|(argument, name)| parameter(argument, &types, name))
        .collect::<syn::Result<Vec<_>>>()
    else {
        return (TraitMethodKind::Rust, Vec::new());
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
            _ => return (TraitMethodKind::Rust, Vec::new()),
        });
        args.push(id);
    }
    let stage_name = entry_name("stage_method", name);
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
            emit!(<Self as #trait_owner>::#name #func (#(#args),*))
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
                        symbol: &#hir::RustSymbol {name: ::core::stringify!(#name), resolve: || <Self as #trait_owner>::#adapter_name #func},
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
                        name: ::core::stringify!(#name),
                        arity: #arity as u32,
                        resolve: || <Self as #trait_owner>::#adapter_name #func,
                    }, &args)
                },
            )
        };
        items.push(adapter);
        emit! {
            let args = [#(#support::RunArg::reg(#args, __splice_builder)),*];
            #call
        }
    } else {
        emit!(::core::result::Result::Err(::core::convert::Into::into(
            "trait method requires Rust arguments that cannot cross the scalar ABI"
        )))
    };
    items.push(emit! {
        #[doc(hidden)]
        #[allow(unused_variables, clippy::too_many_arguments)]
        fn #stage_name #generics (#(#declarations,)* __splice_builder: &mut dyn #root::frontend::KernelBuilder)
            -> #root::Result<#hir::Reg> #where_
        {
            #(#stage_checks)*
            #body
        }
    });
    items.push(evaluator);
    (kind, items)
}

pub(crate) fn stage_shape(
    root: &syn::Path,
    signature: &Signature,
    body: Option<&syn::Block>,
    inherent_visibility: Option<&syn::Visibility>,
) -> [TokenStream; 2] {
    let (shape_kind, policy_kind, bound) = if inherent_visibility.is_some() {
        (
            "inherent_region_stage_shape",
            "inherent_region_needs_stages",
            TokenStream::new(),
        )
    } else {
        (
            "stage_shape",
            "region_needs_stages",
            emit!(where Self: Sized),
        )
    };
    let method = entry_name(shape_kind, &signature.ident);
    let policy = entry_name(policy_kind, &signature.ident);
    let needs_stages = body.map_or_else(
        || emit!(false),
        |body| crate::import::method_policy(root, signature, body),
    );
    let stages = parameter_stages(signature);
    let masks = stage_masks(&stages);
    [
        emit! {
            #[doc(hidden)]
            #[allow(dead_code, clippy::type_complexity)]
            #inherent_visibility fn #method() -> (#([(); #stages],)*) #bound {
                (#(#masks,)*)
            }
        },
        emit! {
            #[doc(hidden)]
            #[allow(dead_code)]
            #inherent_visibility fn #policy() -> bool #bound { #needs_stages }
        },
    ]
}

fn stage_masks(stages: &[usize]) -> impl Iterator<Item = TokenStream> + '_ {
    stages
        .iter()
        .map(|stage| if *stage == 0 { emit!([]) } else { emit!([()]) })
}

fn parameter_stages(signature: &Signature) -> Vec<usize> {
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
