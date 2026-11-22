//! Register adapters for individually published implementations at link time.
use super::declaration::{TraitMethod, parameter_stages, stage_masks, trait_method};
use super::dispatch::{
    receiver_adapter_types, receiver_family, receiver_type_parameters_are_static,
};
use super::{entry_name, normalize, publish};
use quote::format_ident;
use syn::{Block, FnArg, GenericParam, ImplItem, Signature, Type, Visibility};

pub(crate) fn register_individual_trait(
    root: &syn::Path,
    owner: &syn::Type,
    trait_path: &syn::Path,
    implementation: &syn::Generics,
    signature: &Signature,
    body: &Block,
) -> syn::Stmt {
    let registration = trait_registration(root, owner, trait_path, implementation, signature, body);
    let mut generics = implementation.clone();
    generics
        .params
        .push(syn::parse_quote!(__SpliceChecked: #trait_path));
    let where_ = &generics.where_clause;
    let shape = entry_name("stage_shape", &signature.ident);
    let stages = parameter_stages(signature);
    let shape_type = emit!((#([(); #stages],)*));
    let masks = stage_masks(&stages);
    syn::parse2(emit! {
        {
            trait __SpliceUncheckedTrait {
                fn #shape() -> #shape_type;
            }
            impl<T: ?Sized> __SpliceUncheckedTrait for T {
                fn #shape() -> #shape_type {
                    (#(#masks,)*)
                }
            }
            #[allow(dead_code)]
            fn __splice_check_stages #generics () #where_ {
                let _: #shape_type = __SpliceChecked::#shape();
            }
            #registration
        };
    })
    .expect("a nested trait implementation check")
}

fn trait_registration(
    root: &syn::Path,
    owner: &syn::Type,
    trait_path: &syn::Path,
    implementation: &syn::Generics,
    signature: &Signature,
    body: &Block,
) -> Option<syn::Stmt> {
    let kind = trait_method(signature)?;
    let has_receiver = kind != TraitMethod::Scalar;
    if has_receiver && !receiver_type_parameters_are_static(owner, implementation) {
        return None;
    }
    let mut body = body.clone();
    let mut staged_signature = if has_receiver {
        let mut normalized = normalize(signature, Some(&mut body));
        let FnArg::Typed(receiver) = &mut normalized.inputs[0] else {
            unreachable!("the normalized receiver is a typed parameter");
        };
        if let Type::Reference(reference) = receiver.ty.as_mut() {
            *reference.elem = owner.clone();
        } else {
            *receiver.ty = owner.clone();
        }
        normalized
    } else {
        signature.clone()
    };
    let mut uses_self = crate::declaration::UsesSelf::default();
    syn::visit::Visit::visit_block(&mut uses_self, &body);
    syn::visit::Visit::visit_generics(&mut uses_self, implementation);
    syn::visit::Visit::visit_signature(&mut uses_self, &staged_signature);
    if uses_self.0 {
        return None;
    }
    staged_signature.generics = implementation.clone();
    if let Some(where_clause) = &signature.generics.where_clause {
        staged_signature
            .generics
            .make_where_clause()
            .predicates
            .extend(where_clause.predicates.clone());
    }
    let receiver_type = if has_receiver {
        let FnArg::Typed(receiver) = &staged_signature.inputs[0] else {
            unreachable!("the normalized receiver is a typed parameter");
        };
        if crate::scalar::abi::has_early_bound_lifetime(&receiver.ty, &staged_signature.generics) {
            return None;
        }
        Some(receiver.ty.as_ref())
    } else {
        None
    };
    let generics = &staged_signature.generics;
    let where_ = &generics.where_clause;
    let type_args: Vec<_> = generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(item) => Some(&item.ident),
            GenericParam::Const(item) => Some(&item.ident),
            GenericParam::Lifetime(_) => None,
        })
        .collect();
    let fish = (!type_args.is_empty()).then(|| emit!(::<#(#type_args),*>));
    let name = &signature.ident;
    let stage = entry_name("stage_inherent", name);
    let eval = entry_name("eval_inherent", name);
    let helpers: Vec<_> = publish(
        root,
        &staged_signature,
        &body,
        &Visibility::Inherited,
        false,
    )
    .into_iter()
    .filter(|item| matches!(item, ImplItem::Fn(f) if f.sig.ident == stage || f.sig.ident == eval))
    .collect();
    if helpers.len() != 2 {
        return None;
    }
    let types = signature.inputs.iter().filter_map(|arg| match arg {
        FnArg::Typed(arg) => Some(&arg.ty),
        FnArg::Receiver(_) => None,
    });
    let arguments: Vec<_> = (0..signature.inputs.len() - usize::from(has_receiver))
        .map(|n| format_ident!("argument_{n}"))
        .collect();
    let scalar = emit!(#root::frontend::scalar);
    let decoded = arguments
        .iter()
        .zip(types)
        .map(|(arg, ty)| emit!(#arg.map(<#ty as #scalar::ImportWord>::decode)));
    let registry = emit!(#root::frontend::trait_methods);
    let adapter = if let Some(receiver) = receiver_type {
        let (family, lifetimes) = receiver_family(receiver, generics);
        let (eval, stage) = receiver_adapter_types(root, &emit!(#family), &lifetimes);
        emit!(#registry::Method::Receiver(&(
            __splice_eval_words #fish as #eval,
            __splice_stage_words #fish as #stage,
        )))
    } else {
        emit!(#registry::Method::Scalar(#registry::ScalarMethod {
            evaluate: __splice_eval_words #fish,
            stage: __splice_stage_words #fish,
        }))
    };
    let receiver_formal = receiver_type.map(|receiver| emit!(receiver: #receiver,));
    let stage_receiver = has_receiver.then(|| emit!(#scalar::Host(receiver),));
    let eval_receiver = has_receiver.then(|| emit!(Some(receiver),));

    // The CRT and ELF loader call the emitted slot with their zero-argument
    // C ABI. The block emits only a relocated function pointer in the loader
    // section, with no machine instructions in the Rust method. The adapter
    // and its descriptor live in the same executable as the function item.
    Some(syn::parse2(emit! {
        #[cfg(any(
            all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"),
            all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")),
        ))]
        {
            #(#helpers)*

            fn __splice_stage_words #generics (#receiver_formal words: &[#scalar::Word], builder: &mut dyn #root::frontend::KernelBuilder)
                -> #root::Result<#root::hir::Reg> #where_
            {
                let [#(#arguments),*] = words else {
                    return Err("wrong number of arguments for a published trait method".into());
                };
                #stage #fish (#stage_receiver #(*#arguments,)* builder)
            }

            fn __splice_eval_words #generics (#receiver_formal words: &[Option<i32>]) -> #root::Result<i32> #where_ {
                let [#(#arguments),*] = words else {
                    return Err("wrong number of arguments for a published trait method".into());
                };
                #eval #fish ((#eval_receiver #(#decoded,)* )).map(#scalar::ImportWord::encode)
            }

            extern "C" fn __splice_register #generics () #where_ {
                #registry::register(<#owner as #trait_path>::#name, #adapter);
            }

            #[cfg(all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"))]
            unsafe {
                ::core::arch::asm!(
                    ".pushsection .CRT$XCU,\"dr\"", ".p2align 3", ".quad {entry}", ".popsection",
                    entry = sym __splice_register #fish,
                    options(nostack, preserves_flags, nomem, att_syntax)
                );
            }
            #[cfg(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64")))]
            unsafe {
                ::core::arch::asm!(
                    ".pushsection .init_array,\"aw\",%init_array", ".p2align 3", ".quad {entry}", ".popsection",
                    entry = sym __splice_register #fish,
                    options(nostack, preserves_flags, nomem)
                );
            }
        };
    }).expect("a nested scalar trait registration"))
}
