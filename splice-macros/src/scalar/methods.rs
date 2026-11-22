//! Lower published inherent methods with represented operands in the caller's
//! builder. They use the same expression lowering as free functions.
use super::{Binding, Lower, Scalar, abi, flow, parameter};
use proc_macro2::{Ident, Span, TokenStream};
use quote::format_ident;
use std::collections::HashMap;
use syn::visit_mut::VisitMut;
use syn::{Block, FnArg, ImplItem, ReturnType, Signature, Type, Visibility};

mod declaration;
mod dispatch;
mod registration;

pub(crate) use declaration::{individual_trait_supported, stage_shape, trait_defaults, word_trait};
pub(crate) use dispatch::isolated_trait_call;
pub(crate) use registration::register_individual_trait;

/// Rust permits the same method name on unrelated types. Keep the hidden
/// entry points associated with the original type instead of exporting names.
pub(crate) fn entry_name(kind: &str, name: &Ident) -> Ident {
    format_ident!(
        "__splice_{kind}_{}",
        name.to_string().trim_start_matches("r#"),
        span = name.span()
    )
}

pub(crate) fn publish(
    root: &syn::Path,
    signature: &Signature,
    body: &Block,
    visibility: &Visibility,
    trait_impl: bool,
) -> Vec<ImplItem> {
    let runtime_receiver = !trait_impl
        && signature.receiver().is_some_and(|r| {
            r.reference.is_some()
                && r.mutability.is_some()
                && !r.attrs.iter().any(|a| a.path().is_ident("comptime"))
        });
    let mut body = body.clone();
    let mut signature = normalize(signature, Some(&mut body));
    if runtime_receiver && let Some(FnArg::Typed(receiver)) = signature.inputs.first_mut() {
        receiver.attrs.clear();
    }
    if trait_impl {
        signature
            .generics
            .make_where_clause()
            .predicates
            .push(syn::parse_quote!(Self: Sized));
    }

    let name = &signature.ident;
    let types: Vec<_> = signature
        .generics
        .type_params()
        .map(|ty| &ty.ident)
        .collect();
    let unit_result = match &signature.output {
        ReturnType::Default => true,
        ReturnType::Type(_, ty) => matches!(&**ty, Type::Tuple(tuple) if tuple.elems.is_empty()),
    };
    let result_type = match &signature.output {
        ReturnType::Type(_, ty) if !unit_result => match Scalar::from_type(ty, &types) {
            Some(result @ (Scalar::Int | Scalar::Bool)) => result,
            _ => return Vec::new(),
        },
        _ => Scalar::Int,
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

    let mut formals = Vec::new();
    let mut bindings = HashMap::new();
    let mut declarations = Vec::new();
    let mut scalar_args = Vec::new();
    let mut host_args = Vec::new();
    let mut debug_host_args = Vec::new();
    let mut mutable_args = Vec::new();
    let mut context = None;
    let support = emit!(#root::frontend::scalar);
    let hir = emit!(#root::hir);
    for param in &params {
        let Ok(pat) = super::named(param.pattern) else {
            return Vec::new();
        };
        let id = &pat.ident;
        let ty = param.ty;
        if matches!(param.binding, Binding::Host(_)) {
            debug_host_args.push(id);
        }
        let declaration = match param.binding {
            Binding::Host(_)
                if matches!(
                    Scalar::from_type(ty, &types),
                    Some(Scalar::Int | Scalar::Bool | Scalar::Index)
                ) =>
            {
                host_args.push(id);
                emit!(#id: impl #support::KnownArg<#ty>)
            }
            Binding::Host(_) => emit!(#support::Host(#id): #support::Host<#ty>),
            Binding::Reg(Scalar::Int | Scalar::Bool, write) => {
                scalar_args.push(id);
                if write {
                    mutable_args.push(id);
                }
                emit!(#id: impl #support::RunArg)
            }
            Binding::Buffer(_) => emit!(#id: #hir::Buffer),
            Binding::Context if runtime_receiver && context.is_none() => {
                context = Some(super::lower::Context::new(id.clone()));
                emit!(#id: #hir::Buffer)
            }
            _ => return Vec::new(),
        };
        declarations.push(declaration);
        formals.push(param.binding);
        bindings.insert(id.to_string(), param.binding);
    }

    let analysis = flow::analyze(&body, &bindings);
    let debug_names = params
        .iter()
        .map(|param| (param.name.to_string(), param.name.clone()))
        .collect();
    let lower = Lower {
        generics: &signature.generics,
        root,
        types: &types,
        name,
        result_type,
        unit_result,
        formals: &formals,
        bindings,
        aggregates: &Default::default(),
        debug_names,
        flow: &analysis,
        fish: TokenStream::new(),
        loop_kind: None,
        region_depth: 0,
        source_fault: None,
        evaluator_return: None,
        context: context.as_ref(),
    };
    let Ok(code) = lower.kernel_body(&body) else {
        return Vec::new();
    };
    let eval_name = entry_name(
        if trait_impl {
            "eval_method"
        } else {
            "eval_inherent"
        },
        name,
    );
    let stage_name = entry_name(
        if runtime_receiver {
            "stage_context"
        } else if trait_impl {
            "stage_method"
        } else {
            "stage_inherent"
        },
        name,
    );
    let witness = super::context_slice_types(root, &params, &types);
    let frame =
        runtime_receiver.then(|| emit!(_: #witness, __splice_context_frame: #root::ContextFrame,));
    let generics = &signature.generics;
    let where_ = &generics.where_clause;
    let value_type = result_type.value_type(root, &types);
    let arity = scalar_args.len();
    let indices = 0..scalar_args.len();
    let debug_arguments = debug_host_args.iter().map(|id| lower.capture_constant(id));
    let stage = emit! {
        #[doc(hidden)]
        #[allow(unreachable_code, clippy::too_many_arguments)]
        #visibility fn #stage_name #generics (#(#declarations,)* #frame __splice_builder: &mut dyn #root::frontend::KernelBuilder)
            -> #root::Result<#hir::Reg> #where_
        {
            let __splice_guard = #support::enter_expansion()?;
            let __splice_wants_debug = __splice_builder.debug_enabled();
            #(let #host_args = #support::KnownArg::require(#host_args)?;)*
            #(#debug_arguments)*
            let __splice_known: [Option<i32>; #arity] = [#(#support::RunArg::known(&#scalar_args)),*];
            #(let #scalar_args = #support::RunArg::reg(#scalar_args, __splice_builder);)*
            let __splice_inputs: &[(#hir::Reg, Option<i32>)] = &[#((#scalar_args, __splice_known[#indices])),*];
            #(let #mutable_args = __splice_builder.copy(#mutable_args);)*
            let __splice_result = __splice_builder.typed_local(#value_type);
            let __splice_return_label = __splice_builder.label();
            #code
            __splice_builder.mark(__splice_return_label);
            Ok(__splice_result)
        }
    };

    let mut items = vec![syn::parse2(stage).expect("a generated method stage")];
    if let Some(context) = context.as_ref() {
        let marker = entry_name("has_context", name);
        items.push(syn::parse_quote!(#[doc(hidden)] #visibility fn #marker() -> bool {true}));
        let signature_name = entry_name("context_signature", name);
        let dispatch_name = entry_name("dispatch_context", name);
        let signature = context.signature(root);
        let dispatch = context.dispatcher(root, emit!(Self::#signature_name), &types);
        items.push(syn::parse_quote!(
            #[doc(hidden)] #[allow(non_upper_case_globals)]
            #visibility const #signature_name: &'static #root::ContextSignature = &#signature;
        ));
        items.push(
            syn::parse2(emit!(
                #[doc(hidden)]
                #visibility fn #dispatch_name #generics (&mut self, __splice_operation:u32,
                    __splice_arguments:#root::ContextArguments<'_>, _: #witness) -> i32 #where_ {
                    let __splice_object = self;
                    #dispatch
                }
            ))
            .expect("generated method dispatcher"),
        );
        return items;
    }
    if !trait_impl {
        let marker = entry_name("published", name);
        items.push(syn::parse_quote!(#[doc(hidden)] #visibility fn #marker() -> bool { true }));
    }
    if let Ok(eval) = lower.method_evaluator(&signature, &body, visibility) {
        let mut eval: syn::ImplItemFn = syn::parse2(eval).expect("a generated method evaluator");
        eval.sig.ident = eval_name;
        items.push(ImplItem::Fn(eval));
    }
    items
}

fn normalize(signature: &Signature, body: Option<&mut Block>) -> Signature {
    let mut signature = signature.clone();
    if let Some(FnArg::Receiver(receiver)) = signature.inputs.first_mut() {
        let ty = &receiver.ty;
        let replacement = Ident::new("__splice_receiver", Span::mixed_site());
        let captured: FnArg = syn::parse_quote!(#[comptime] #replacement: #ty);
        signature.inputs[0] = captured;
        struct Receiver<'a>(&'a Ident);
        impl VisitMut for Receiver<'_> {
            fn visit_expr_path_mut(&mut self, path: &mut syn::ExprPath) {
                if path.path.is_ident("self") {
                    let replacement = self.0;
                    *path = syn::parse_quote!(#replacement);
                } else {
                    syn::visit_mut::visit_expr_path_mut(self, path);
                }
            }
        }
        if let Some(body) = body {
            Receiver(&replacement).visit_block_mut(body);
        }
    }
    signature
}

pub(crate) fn method_path(path: &syn::ExprPath) -> bool {
    path.qself.is_some()
        || path.path.segments.iter().rev().nth(1).is_some_and(|s| {
            s.ident == "Self" || s.ident.to_string().starts_with(char::is_uppercase)
        })
}

pub(crate) fn rename_path(path: &syn::ExprPath, kind: &str) -> syn::ExprPath {
    let mut path = path.clone();
    let segment = path.path.segments.last_mut().expect("a method path");
    segment.ident = entry_name(kind, &segment.ident);
    path
}

pub(crate) fn trait_path(path: &syn::ExprPath) -> Option<syn::Path> {
    let qualifier = path.qself.as_ref()?;
    qualifier.as_token?;
    let mut owner = path.path.clone();
    owner.segments = owner
        .segments
        .into_iter()
        .take(qualifier.position)
        .collect();
    Some(owner)
}
