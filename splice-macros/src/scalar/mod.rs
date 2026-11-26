//! Staged scalar functions. Rust executes host code while symbolic runtime
//! expressions append register operations and control-flow edges to the IR.
// Generated locals and labels share a hygiene context across fragments. Tokens
// interpolated from the kernel keep their original spans and lexical bindings.
macro_rules! emit {
    ($($tokens:tt)*) => { quote::quote_spanned!(proc_macro2::Span::mixed_site()=> $($tokens)*) };
}

pub(crate) mod abi;
pub(crate) mod lower;
pub(crate) mod methods;
pub(crate) use lower::simd::constructor_arity;
use lower::{Binding, BufferType, Lower, Scalar};
pub(crate) use lower::{Elem, select_free, uses_stage_requirements};
use proc_macro2::{Ident, TokenStream};
use std::collections::HashMap;
use syn::visit::Visit;
use syn::{Error, FnArg, GenericParam, ItemFn, Meta, Pat, Path, Result, ReturnType, Type};

// Only configuration gates belong on all generated items. A cfg_attr may
// also contain function-specific attributes such as inline or test.
pub(crate) fn configuration(meta: &Meta) -> Result<Option<TokenStream>> {
    if meta.path().is_ident("cfg") {
        return Ok(Some(emit!(#meta)));
    }
    let Meta::List(list) = meta else {
        return Ok(None);
    };
    if !list.path.is_ident("cfg_attr") {
        return Ok(None);
    }
    let args = list
        .parse_args_with(syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated)?;
    let mut args = args.iter();
    let condition = args
        .next()
        .ok_or_else(|| Error::new_spanned(list, "expected a cfg condition"))?;
    let gates = args
        .filter_map(|meta| configuration(meta).transpose())
        .collect::<Result<Vec<_>>>()?;
    Ok((!gates.is_empty()).then(|| emit!(cfg_attr(#condition, #(#gates),*))))
}

/// `print` (`#[kernel(print)]`) also publishes `print`, which renders the
/// specialized kernel as Rust; without it the kernel prints as a call.
pub(super) fn expand(root: Path, native: &mut ItemFn, print: bool) -> Result<TokenStream> {
    let mut generated_sig = native.sig.clone();
    crate::declaration::rebase_signature(&mut generated_sig);
    let sig = &generated_sig;
    let module_vis = &native.vis;
    let vis = &crate::declaration::nested_visibility(module_vis);
    let gates = native
        .attrs
        .iter()
        .filter_map(|a| configuration(&a.meta).transpose())
        .collect::<Result<Vec<_>>>()?;
    // `declaration::signature` has rejected async, unsafe, extern and
    // variadic signatures.
    let unit = syn::parse_quote!(());
    let ty = match &sig.output {
        ReturnType::Default => &unit,
        ReturnType::Type(_, ty) => &**ty,
    };
    let unit_result = matches!(ty, Type::Tuple(tuple) if tuple.elems.is_empty());
    let types: Vec<_> = sig.generics.type_params().map(|ty| &ty.ident).collect();
    let result_type = if unit_result {
        Scalar::Int
    } else {
        Scalar::from_type(ty, &types)
            .filter(|t| {
                matches!(
                    t,
                    Scalar::Int
                        | Scalar::Bool
                        | Scalar::Index
                        | Scalar::Vector(_)
                        | Scalar::Mask(_)
                )
            })
            .ok_or_else(|| {
                Error::new_spanned(ty, "expected (), i32, bool, usize, Lane<T> or Mask<T>")
            })?
    };
    if matches!(
        result_type,
        Scalar::Vector(Elem::Param(_)) | Scalar::Mask(Elem::Param(_))
    ) {
        return Err(Error::new_spanned(
            ty,
            "a kernel cannot return vectors or masks with generic elements",
        ));
    }
    let name = &sig.ident;
    let generic = &sig.generics;
    let where_ = &generic.where_clause;
    let outer_generic = &native.sig.generics;
    let outer_where = &outer_generic.where_clause;
    let args: Vec<_> = generic
        .params
        .iter()
        .filter_map(|p| match p {
            GenericParam::Type(t) => Some(&t.ident),
            GenericParam::Const(c) => Some(&c.ident),
            _ => None,
        })
        .collect();
    let fish = (!args.is_empty()).then(|| emit!(::<#(#args),*>));
    let hir = emit!(#root::hir);
    let factory = emit!(dyn #root::frontend::KernelBuilder);
    let support = emit!(#root::frontend::scalar);
    let mut bindings = HashMap::new();
    let mut debug_names = std::collections::BTreeMap::new();
    let mut formals = Vec::new();
    let mut host_types = Vec::new();
    let mut outer_host_types = Vec::new();
    let mut host_names = Vec::new();
    let mut helper_args = Vec::new();
    let mut inline_args = Vec::new();
    let mut reg_names = Vec::new();
    let mut buffer_names = Vec::new();
    let mut buffer_kinds = Vec::new();
    let mut columns = None;
    let mut column_parameter = None;
    let mut outer_call_decls = Vec::new();
    let mut nested_decls = Vec::new();
    let mut extract_known = Vec::new();
    let mut copy_mutable = Vec::new();
    let mut source_decls = Vec::new();
    let mut source_params = Vec::new();
    let mut source_args = Vec::new();
    let parameters = sig
        .inputs
        .iter()
        .zip(abi::parameter_names(sig))
        .map(|(arg, name)| parameter(arg, &types, name))
        .collect::<Result<Vec<_>>>()?;
    let mut pattern_decls = Vec::new();
    let mut mutable_host = Vec::new();
    let mut context = None;
    for ((arg, original), parameter) in sig.inputs.iter().zip(&native.sig.inputs).zip(&parameters) {
        let abi::Parameter {
            name: id,
            ty,
            pattern,
            binding: formal,
        } = parameter;
        let formal = *formal;
        let FnArg::Typed(original) = original else {
            unreachable!()
        };
        let original_ty = &original.ty;
        if matches!(formal, Binding::Host(_)) {
            match pattern {
                Pat::Ident(pat) if pat.by_ref.is_none() && pat.subpat.is_none() => {
                    if pat.mutability.is_some() {
                        mutable_host.push(emit!(let mut #id = #id;));
                    }
                }
                _ => {
                    let pattern = &original.pat;
                    pattern_decls.push(syn::parse_quote!(let #pattern = #id;));
                }
            }
        }
        let source = emit!(#root::frontend::source);
        if matches!(formal, Binding::Host(_)) {
            if matches!(
                Scalar::from_type(ty, &types),
                Some(Scalar::Int | Scalar::Bool | Scalar::Index)
            ) {
                source_decls.push(emit!(#id:impl #support::KnownArg<#ty>));
                extract_known.push(emit!(let #id=#support::KnownArg::require(#id)?;));
            } else {
                source_decls.push(emit!(#support::Host(#id): #support::Host<#ty>));
            }
        } else {
            source_decls.push(emit!(#id: impl #source::Argument));
            source_args.push(emit!(#source::Argument::source(#id)));
            let writable = matches!(formal, Binding::Reg(_, true));
            let prefix = if writable { "mut " } else { "" };
            let source_ty = if matches!(formal, Binding::Columns) {
                emit!(::core::concat!(
                    "&[",
                    ::core::stringify!(#root::BufferArg),
                    "<'_>]"
                ))
            } else {
                emit!(::core::any::type_name::<#ty>())
            };
            source_params.push(emit!(
                ::std::format!("{}{}: {}", #prefix, ::core::stringify!(#id), #source_ty)
            ));
        }
        if matches!(formal, Binding::Columns) {
            let FnArg::Typed(arg) = arg else {
                unreachable!()
            };
            let layout = arg.attrs[0].parse_args::<syn::Expr>()?;
            if columns.replace(layout).is_some() {
                return Err(Error::new_spanned(arg, "a kernel accepts one column table"));
            }
            column_parameter = Some(id.clone());
        }
        helper_args.push(if matches!(formal, Binding::Host(_)) {
            emit!(#support::Host(#id))
        } else {
            emit!(#id)
        });
        let decl = match formal {
            Binding::Aggregate(_) | Binding::Word(_) => {
                unreachable!("local used as a formal parameter")
            }
            Binding::Host(_) => {
                host_names.push(id);
                host_types.push(ty);
                outer_host_types.push(original_ty);
                inline_args.push(emit!(#support::Host(#id)));
                emit!(#support::Host(#id): #support::Host<#ty>)
            }
            Binding::Columns => {
                inline_args.push(emit!((__splice_columns, #hir::Buffer(__splice_column_base))));
                emit!(#id: (&[#hir::BufferKind], #hir::Buffer))
            }
            Binding::Buffer(writable) => {
                if columns.is_some() {
                    return Err(Error::new_spanned(
                        arg,
                        "declare the column table after other buffers",
                    ));
                }
                let index = buffer_names.len() as u32;
                inline_args.push(emit!(#hir::Buffer(#index)));
                buffer_names.push(id);
                let elem = writable.elem.tokens(&root, &types);
                let write = writable.write;
                buffer_kinds.push(emit!(#hir::BufferKind::new(#elem,#write)));
                emit!(#id: #hir::Buffer)
            }
            Binding::Context => {
                if columns.is_some() || context.is_some() {
                    return Err(Error::new_spanned(
                        arg,
                        "declare one Rust context before any column table",
                    ));
                }
                if !matches!(pattern, Pat::Ident(pat) if pat.by_ref.is_none() && pat.subpat.is_none())
                {
                    return Err(Error::new_spanned(
                        pattern,
                        "the Rust context parameter requires a named binding",
                    ));
                }
                let index = buffer_names.len() as u32;
                context = Some(lower::Context::for_function(id.clone()));
                inline_args.push(emit!(#hir::Buffer(#index)));
                buffer_names.push(id);
                buffer_kinds.push(emit!(#hir::BufferKind::context(&__SPLICE_CONTEXT_SIGNATURE)));
                emit!(#id: #hir::Buffer)
            }
            Binding::Reg(_, mutable) => {
                let index = reg_names.len();
                inline_args.push(emit!(__splice_builder.arg(#index)?));
                reg_names.push(id);
                if mutable {
                    copy_mutable.push(emit!(let #id = __splice_builder.copy(#id)?;));
                }
                emit!(#id: impl #support::RunArg)
            }
        };
        nested_decls.push(if matches!(formal, Binding::Host(_)) {
            source_decls.last().unwrap().clone()
        } else {
            decl.clone()
        });
        outer_call_decls.push(if matches!(formal, Binding::Host(_)) {
            emit!(#support::Host(#id): #support::Host<#original_ty>)
        } else {
            decl.clone()
        });
        bindings.insert(id.to_string(), formal);
        debug_names.insert(id.to_string(), id.clone());
        formals.push(formal);
    }
    if let Some(context) = &mut context {
        for parameter in &parameters {
            if matches!(parameter.binding, Binding::Host(_))
                && let Some(kind @ (Scalar::Int | Scalar::Bool)) =
                    Scalar::from_type(parameter.ty, &types)
            {
                context.register_host(parameter.name.clone(), kind);
            }
        }
    }
    let arity = reg_names.len();
    // Only a scalar kernel has a Signature: a SIMD helper expands into its caller.
    let scalar_signature =
        |t| lower::scalar_type(&root, t).unwrap_or_else(|| emit!(#hir::ScalarType::I32));
    let param_types = formals.iter().filter_map(|formal| match *formal {
        Binding::Reg(t, _) => Some(scalar_signature(t)),
        _ => None,
    });
    let params = emit!(&[#(#param_types),*]);
    let result = scalar_signature(result_type);
    let host_decls = emit!(#(#host_names: #host_types,)*);
    let outer_host_decls = emit!(#(#host_names: #outer_host_types,)*);
    let unpack = emit!(#(let #reg_names = #support::RunArg::reg(#reg_names, __splice_builder)?;)*);
    let mut lowered_block = native.block.clone();
    lowered_block.stmts.splice(0..0, pattern_decls);
    let body = crate::kast::Body::new(&lowered_block, &stage_params(&parameters))?;
    let lower = Lower {
        generics: generic,
        root: &root,
        types: &types,
        name,
        result_type,
        unit_result,
        formals: &formals,
        bindings,
        aggregates: &Default::default(),
        labels: &Default::default(),
        debug_names,
        body: &body,
        fish: emit!(#fish),
        loop_kind: None,
        region_depth: 0,
        source_fault: None,
        evaluator_return: None,
        context: context.as_ref(),
    };
    let evaluator = lower.evaluator(&native.sig, &lowered_block, module_vis, None)?;
    let code = lower.kernel_body(&lowered_block)?;
    let owned_scope = lower.owned_scope(native);
    let debug_arguments = host_names.iter().map(|id| lower.capture_constant(id));
    let source_support = emit!(#root::frontend::source);
    let overflow = lower::overflow_prologue(&root);
    if print && context.is_some() {
        return Err(Error::new_spanned(
            sig,
            "a Rust context kernel cannot be printed as Rust; inspect its portable IR with `ir()`",
        ));
    }
    let source_body = if print {
        let body = lower.source_body(&lowered_block)?;
        emit!(let __splice_printed = #body;)
    } else {
        TokenStream::new()
    };
    let source_aliases = types.iter().map(|ty| {
        emit!((
            ::std::borrow::ToOwned::to_owned(::core::stringify!(#ty)),
            ::std::format!("type {} = {};", ::core::stringify!(#ty), ::core::any::type_name::<#ty>())
        ))
    }).chain(generic.const_params().map(|param| {
        let name = &param.ident;
        let ty = &param.ty;
        emit!((::std::borrow::ToOwned::to_owned(::core::stringify!(#name)), ::std::format!("const {}: {} = {};", ::core::stringify!(#name), ::core::stringify!(#ty), #source_support::literal(#name))))
    }));
    let (source_entry, source_body_method) = if !print {
        // A caller's `print` shows the call as written.
        let nested_source = columns.is_none().then(|| emit! {
            #[doc(hidden)]
            #[allow(unused_variables, clippy::too_many_arguments)]
            #vis fn __splice_source_call #generic (#(#source_decls,)* __splice_source: &mut #source_support::Source) -> #root::Result<::core::option::Option<::std::string::String>> #where_ {
                ::core::result::Result::Ok(::core::option::Option::None)
            }
        });
        (emit!(#nested_source), TokenStream::new())
    } else {
        let source_layout = columns.as_ref().map(|layout| emit! {
            let __splice_layout = #layout;
            let __splice_column_layout: &[#root::BufferKind] = ::core::convert::AsRef::as_ref(&__splice_layout);
        });
        let nested_source = if columns.is_none() {
            emit! {
                #[doc(hidden)]
                #[allow(clippy::too_many_arguments)]
                #vis fn __splice_source_call #generic (#(#source_decls,)* __splice_source: &mut #source_support::Source) -> #root::Result<::core::option::Option<::std::string::String>> #where_ {
                    #(#extract_known)*
                    let name = __SpliceDefinition::__splice_print #fish (#(#host_names,)* __splice_source)?;
                    let arguments: &[::std::string::String] = &[#(#source_args),*];
                    ::core::result::Result::Ok(::core::option::Option::Some(::std::format!("{}({})", name, arguments.join(", "))))
                }
            }
        } else {
            emit!()
        };
        let entry = emit! {
            #vis fn print #generic (#host_decls) -> #root::Result<::std::string::String> #where_ {
                let mut source = <#source_support::Source as ::core::default::Default>::default();
                __SpliceDefinition::__splice_print #fish (#(#host_names,)* &mut source)?;
                ::core::result::Result::Ok(::std::format!("#[allow(unused_imports)]\nuse {}::simd::{{Lane, Mask}};\n\n{}", ::core::stringify!(#root), source.functions.join("\n")))
            }
            #nested_source
        };
        let body = emit! {
            #[allow(unused_variables, unreachable_code)]
            fn __splice_print #outer_generic (#outer_host_decls __splice_source: &mut #source_support::Source) -> #root::Result<::std::string::String> #outer_where {
                let _guard = #support::enter_expansion()?;
                #overflow
                let (id, name) = __splice_source.function(::core::stringify!(#name));
                let transfer = __splice_source.transfer.take();
                #source_layout
                #(#mutable_host)*
                let __splice_printed: ::std::string::String;
                #source_body
                __splice_source.transfer = transfer;
                __splice_source.functions[id] = #source_support::function(&name, &[#(#source_params),*], ::core::any::type_name::<#ty>(), &[#(#source_aliases),*], &__splice_printed);
                ::core::result::Result::Ok(name)
            }
        };
        (entry, body)
    };
    let kinds = emit!(const { &[#(#buffer_kinds),*] });
    let typed_helper = matches!(result_type, Scalar::Vector(_) | Scalar::Mask(_))
        || formals
            .iter()
            .any(|f| matches!(f, Binding::Reg(Scalar::Vector(_) | Scalar::Mask(_), _)));
    if typed_helper && context.is_some() {
        return Err(Error::new_spanned(
            sig,
            "a Rust context requires the scalar entry ABI",
        ));
    }
    let context_sign = context.as_ref().map(|context| {
        let signature = context.signature(&root);
        emit!(#[doc(hidden)] #vis static __SPLICE_CONTEXT_SIGNATURE: #root::ContextSignature = #signature;)
    });
    let root_context_args = context
        .as_ref()
        .map(|_| emit!(#root::ContextFrame::root(&__SPLICE_CONTEXT_SIGNATURE),));
    let context_decls = context
        .as_ref()
        .map(|_| emit!(__splice_context_frame: #root::ContextFrame,));
    let context_exports = context.as_ref().map(|context| {
        let dispatch = context.dispatcher(&root, emit!(&__SPLICE_CONTEXT_SIGNATURE), &types);
        let object = parameters.iter().find(|p| matches!(p.binding, Binding::Context)).expect("a context parameter").ty;
        let witness = context_slice_types(&root, &parameters, &types);
        emit! {
            #[doc(hidden)]
            #[allow(clippy::too_many_arguments)]
            #vis fn __splice_stage_context #generic (#(#nested_decls,)* _: #witness, __splice_context_frame: #root::ContextFrame,
                __splice_builder: &mut #factory) -> #root::Result<#hir::Reg> #where_ {
                #(#extract_known)*
                __SpliceDefinition::__splice_inline #fish (#(#helper_args,)* __splice_context_frame, __splice_builder)
            }

            #[doc(hidden)]
            #vis fn __splice_dispatch_context #generic (__splice_object: #object,
                __splice_operation: u32, __splice_arguments: #root::ContextArguments<'_>, _: #witness) -> i32 #where_ {
                #dispatch
            }
        }
    });
    let public_abi = abi::generate(&root, vis, sig, &parameters, unit_result, context.as_ref());
    // Rust requires lifetime parameters before the backend type; the backend
    // is still the first argument at the usual lifetime-elided turbofish.
    let mut compile_generics = public_abi.build_generics.clone();
    let backend = emit!(__SpliceCodegen);
    let lifetimes = compile_generics.lifetimes().count();
    compile_generics
        .params
        .insert(lifetimes, syn::parse_quote!(#backend: #root::Codegen));
    let callback = emit!(::std::boxed::Box::new(move |__splice_builder| __stage_body #fish (#(#host_names,)* __splice_builder)));
    let frontend = emit!(#root::frontend);
    let value_args = host_names
        .iter()
        .copied()
        .chain(generic.const_params().map(|p| &p.ident));
    let type_args = generic.type_params().map(|p| &p.ident);
    let specialize = emit! {
        #[allow(unused_imports)]
        use #frontend::MaybeKey as _;
        #[allow(unused_mut)]
        let mut __splice_key = ::core::option::Option::Some(#frontend::Specialization::new::<__SpliceIdentity>(#frontend::Signature::new(#params, #result, #kinds)));
        #((&#frontend::KeyArgument::<#type_args>::for_type()).add_type(&mut __splice_key);)*
        #((&#frontend::KeyArgument::for_value(&#value_args)).add_value(&#value_args, &mut __splice_key);)*
    };
    // A body that calls itself expands inline; the same specialization
    // inside itself is unbounded recursion.
    let recursion = calls_itself(&native.block, name).then(|| {
        let type_args = generic.type_params().map(|p| &p.ident);
        let value_args = host_names
            .iter()
            .copied()
            .chain(generic.const_params().map(|p| &p.ident));
        emit! {
            #[allow(unused_mut)]
            let mut __splice_key = ::core::option::Option::Some(#frontend::Specialization::new::<Self>(#frontend::Signature::new(#params, #result, #kinds)));
            {
                #[allow(unused_imports)]
                use #frontend::MaybeKey as _;
                #((&#frontend::KeyArgument::<#type_args>::for_type()).add_type(&mut __splice_key);)*
                #((&#frontend::KeyArgument::for_value(&#value_args)).add_value(&#value_args, &mut __splice_key);)*
            }
            let __splice_recursion = #support::enter_inline(__splice_key.map(|key| key.with_known(&__splice_input_values)))?;
        }
    });
    let result_value_type = result_type.value_type(&root, &types);
    let source = emit! {
        __splice_builder.set_source(&#root::frontend::KernelSource {
            name: ::core::stringify!(#name),
            scalars: &[#(::core::stringify!(#reg_names)),*],
            buffers: &[#(::core::stringify!(#buffer_names)),*],
        });
    };
    let nested_body = if context.is_some() {
        emit!(::core::result::Result::Err(#root::BuildError::staging(
            "a runtime object requires a compatible enclosing Rust context"
        )))
    } else {
        emit! {
            #(#extract_known)*
            if __SPLICE_NEEDS_STAGES && (false #(|| #support::RunArg::known(&#reg_names).is_some())*) {
                return __SpliceDefinition::__splice_inline #fish (#(#helper_args,)* __splice_builder);
            }
            #unpack
            #specialize
            let __splice_body: #frontend::Body<'_> = #callback;
            match __splice_key {
                ::core::option::Option::Some(key) => __splice_builder.call_specialized(key, &[#(#reg_names),*], &[#(#buffer_names),*], __splice_body),
                ::core::option::Option::None => __splice_builder.call_kernel(#frontend::Signature::new(#params, #result, #kinds), &[#(#reg_names),*], &[#(#buffer_names),*], __splice_body),
            }
        }
    };
    let nested_allow = context.as_ref().map(|_| emit!(#[allow(unused_variables)]));
    let standalone = emit! {
            fn __stage_body #generic (#host_decls __splice_builder: &mut #factory) -> #root::Result<()> #where_ {
                #source
                let output = __SpliceDefinition::__splice_inline #fish (#(#inline_args,)* #root_context_args __splice_builder)?;
                __splice_builder.ret(output)?;
                ::core::result::Result::Ok(())
            }
            #[doc(hidden)]
            #[allow(clippy::too_many_arguments)]
            #nested_allow
            #vis fn __splice_call #generic (#(#nested_decls,)* __splice_builder: &mut #factory)
                -> #root::Result<#hir::Reg> #where_ {
                #nested_body
            }

    };
    // The public kernel ABI transports i32 scalars and slices. SIMD helpers
    // expand into their caller so vectors and masks stay in typed registers.
    let (setup, inner_call) = if let Some(mut layout) = columns {
        if typed_helper {
            return Err(Error::new_spanned(
                sig,
                "a column-table entry point requires the scalar calling convention",
            ));
        }
        crate::declaration::rebase_expression(&mut layout, &sig.generics);
        let base = buffer_names.len() as u32;
        let setup = emit! {
            let __splice_layout = #layout;
            let __splice_columns: &[#hir::BufferKind] = ::core::convert::AsRef::as_ref(&__splice_layout);
            if __splice_columns.iter().any(|column| column.writable()) {
                return ::core::result::Result::Err(#root::BuildError::staging("a shared column table cannot contain writable columns"));
            }
            let __splice_column_base = #base;
            let mut __splice_kinds = (#kinds).to_vec();
            __splice_kinds.extend_from_slice(__splice_columns);
            let __splice_body: #frontend::Body<'_> = ::std::boxed::Box::new(move |__splice_builder| {
                #source
                let output = __SpliceDefinition::__splice_inline #fish (#(#inline_args,)* #root_context_args __splice_builder)?;
                __splice_builder.ret(output)?;
                ::core::result::Result::Ok(())
            });
            let __splice_kinds = &__splice_kinds;
        };
        (setup, emit!())
    } else if typed_helper {
        (
            emit!(),
            emit! {
                #[doc(hidden)]
                #[allow(clippy::too_many_arguments)]
                #vis fn __splice_call #generic (#(#nested_decls,)* __splice_builder:&mut #factory)->#root::Result<#hir::Reg> #where_{
                    #(#extract_known)*
                    __SpliceDefinition::__splice_inline #fish (#(#helper_args,)* __splice_builder)
                }
            },
        )
    } else {
        (
            emit! {
                let __splice_kinds = #kinds;
                let __splice_body: #frontend::Body<'_> = #callback;
            },
            standalone,
        )
    };
    let entry_points = if typed_helper {
        emit!(#inner_call)
    } else {
        let abi::Abi {
            build_type,
            signature,
            items,
            ..
        } = public_abi;
        emit! {
            #items

            struct __SpliceIdentity;

            #vis fn build #compile_generics (__splice_args: #build_type)
                -> ::core::result::Result<#root::TypedKernel<#signature, <#backend as #root::Backend>::Output>, #root::BuildError> #where_
            {
                build_with_options::<#backend #(, #args)*>(
                    #root::BuildOptions::host(#root::Mode::for_profile(::core::cfg!(debug_assertions)))?,
                    __splice_args,
                )
            }
            #vis fn build_with_options #compile_generics (
                __splice_options: #root::BuildOptions, __splice_args: #build_type,
            ) -> ::core::result::Result<#root::TypedKernel<#signature, <#backend as #root::Backend>::Output>, #root::BuildError> #where_ {
                #root::build_boundary(move || {
                    let BuildArgs { #(#host_names,)* } = __splice_args;
                    #setup
                    <#backend as #root::Codegen>::build_with_options(
                        #root::frontend::Signature::new(#params, #result, __splice_kinds), __splice_body, __splice_options,
                    ).map(|output| #root::TypedKernel::new(output, Signature(::core::marker::PhantomData)))
                })
            }
            #vis fn ir #generic (#host_decls)
                -> ::core::result::Result<#root::TypedIr<#signature>, #root::BuildError> #where_
            {
                ir_with_options #fish (
                    #root::BuildOptions::host(#root::Mode::for_profile(::core::cfg!(debug_assertions)))?,
                    #(#host_names,)*
                )
            }
            #vis fn ir_with_options #generic (__splice_options: #root::BuildOptions, #host_decls)
                -> ::core::result::Result<#root::TypedIr<#signature>, #root::BuildError> #where_
            {
                #root::build_boundary(move || {
                    #setup
                    #root::frontend::materialize(
                        #root::frontend::Signature::new(#params, #result, __splice_kinds), __splice_body, __splice_options,
                    ).map(|function| #root::TypedIr::new(function, Signature(::core::marker::PhantomData)))
                })
            }
            #inner_call
        }
    };
    let reg_indices = 0..reg_names.len();
    let needs_stages = uses_stage_requirements(&lowered_block);
    let region_policy = crate::import::publication(&root, vis, sig, needs_stages);
    let foldable = lower::can_fold(sig, &lowered_block);
    let fold_macro = if foldable {
        emit!(#root::__splice_choose_kernel)
    } else {
        emit!(#root::__splice_choose_rust)
    };
    let generated = emit! {
        #(#[#gates])*
        #module_vis mod #name {
            #[allow(unused_imports)]
            use super::*;

            #[doc(hidden)] #vis struct __SpliceDefinition;
            #[doc(hidden)]
            const __SPLICE_NEEDS_STAGES: bool = #needs_stages;
            #[doc(hidden)] #vis use #fold_macro as __splice_fold;
            #vis const __SPLICE_RETURN: #hir::ValueType = #result_value_type;
            #[doc(hidden)]
            #vis use #root::__splice_choose_kernel as __splice_dispatch;
            #region_policy
            #context_sign
            #context_exports
            #entry_points
            #source_entry
        }

        #(#[#gates])*
        impl #name::__SpliceDefinition {
            #evaluator
            #source_body_method
            // A staging error replaces the operation that would read its operands.
            #[allow(unreachable_code, unused_variables, clippy::too_many_arguments)]
            fn __splice_inline #outer_generic (#(#outer_call_decls,)* #context_decls __splice_builder:&mut #factory)
                -> #root::Result<#hir::Reg> #outer_where {
                let __splice_guard = #support::enter_expansion()?;
                #overflow
                let __splice_wants_debug = __splice_builder.debug_enabled();
                let __splice_input_values: [::core::option::Option<#hir::Constant>; #arity] = [#(#support::RunArg::known(&#reg_names)),*];
                #recursion
                #unpack
                #owned_scope;
                let __splice_inputs: &[(#hir::Reg, ::core::option::Option<#hir::Constant>)] = &[#((#reg_names, __splice_input_values[#reg_indices])),*];
                #(#copy_mutable)*
                #(#mutable_host)*
                #(#debug_arguments)*
                let __splice_result = __splice_builder.local(#result_value_type)?;
                let __splice_return_label = __splice_builder.label()?;
                #code
                __splice_builder.mark(__splice_return_label)?;
                ::core::result::Result::Ok(__splice_result)
            }
        }
    };
    for arg in &mut native.sig.inputs {
        if let FnArg::Typed(arg) = arg {
            arg.attrs.clear();
        }
    }
    crate::declaration::native_body(&mut native.block)?;
    if let Some(id) = column_parameter {
        native.block.stmts.insert(
            0,
            syn::parse_quote!(let #id: &[#root::BufferArg<'_>] = #id;),
        );
    }
    native.attrs.push(syn::parse_quote!(#[allow(dead_code)]));
    Ok(emit!(#native #generated))
}

/// Whether `block` calls the function `name` by its bare name.
fn calls_itself(block: &syn::Block, name: &Ident) -> bool {
    struct Calls<'a>(&'a Ident, bool);
    impl<'ast> Visit<'ast> for Calls<'_> {
        fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
            self.1 |= matches!(&*call.func, syn::Expr::Path(p) if p.qself.is_none() && p.path.is_ident(self.0));
            syn::visit::visit_expr_call(self, call);
        }
        fn visit_item(&mut self, _: &'ast syn::Item) {}
    }
    let mut calls = Calls(name, false);
    calls.visit_block(block);
    calls.1
}

/// The parameters of a kernel as its body's staging analysis sees them.
fn stage_params(parameters: &[abi::Parameter<'_>]) -> Vec<(Ident, crate::kast::Param)> {
    parameters
        .iter()
        .map(|parameter| (parameter.name.clone(), stage_param(parameter.binding)))
        .collect()
}

fn stage_param(binding: Binding) -> crate::kast::Param {
    use crate::kast::Param;
    match binding {
        Binding::Host(_) => Param::Known,
        Binding::Reg(t, _) | Binding::Word(t) => Param::Scalar {
            index: t.base() == Scalar::Index,
        },
        Binding::Buffer(_) => Param::Slice,
        Binding::Columns | Binding::Context | Binding::Aggregate(_) => Param::Object,
    }
}

fn context_slice_types(
    root: &Path,
    params: &[abi::Parameter<'_>],
    types: &[&Ident],
) -> TokenStream {
    let elements = params.iter().filter_map(|param| match param.binding {
        Binding::Buffer(buffer) => Some(buffer.elem.ty(root, types)),
        _ => None,
    });
    emit!(::core::marker::PhantomData<(#(#elements,)*)>)
}

fn parameter<'a>(arg: &'a FnArg, types: &[&Ident], name: Ident) -> Result<abi::Parameter<'a>> {
    let FnArg::Typed(arg) = arg else {
        unreachable!("callers pass a method receiver separately")
    };
    let mutable = matches!(&*arg.pat, Pat::Ident(pat) if pat.mutability.is_some());
    let binding = match arg.attrs.as_slice() {
        [attr] if matches!(&attr.meta, Meta::Path(p) if p.is_ident("comptime")) => Binding::Host(0),
        [attr] if matches!(&attr.meta, Meta::List(list) if list.path.is_ident("columns")) => {
            let valid = match &*arg.ty {
                Type::Reference(r) if r.mutability.is_none() => match &*r.elem {
                    Type::Slice(s) => {
                        matches!(&*s.elem, Type::Path(p) if p.path.segments.last().is_some_and(|s| s.ident == "BufferArg"))
                    }
                    _ => false,
                },
                _ => false,
            };
            if mutable || !valid {
                return Err(Error::new_spanned(
                    arg,
                    "expected columns: &[BufferArg<'_>]",
                ));
            }
            Binding::Columns
        }
        [] if matches!(
            Scalar::from_type(&arg.ty, types),
            Some(Scalar::Int | Scalar::Bool | Scalar::Index | Scalar::Vector(_) | Scalar::Mask(_))
        ) =>
        {
            Binding::Reg(Scalar::from_type(&arg.ty, types).unwrap(), mutable)
        }
        [] if matches!(&*arg.ty, Type::Reference(r) if r.mutability.is_some())
            && slice_type(&arg.ty, types).is_none() =>
        {
            Binding::Context
        }
        [] => {
            let writable = slice_type(&arg.ty, types).ok_or_else(|| {
                Error::new_spanned(
                    &arg.ty,
                    "runtime parameters must be i32, bool, usize or a slice of u16, u32, i32, i64, f32, or f64",
                )
            })?;
            if mutable {
                return Err(Error::new_spanned(
                    &arg.pat,
                    "use &mut [i32] to write to a slice",
                ));
            }
            Binding::Buffer(writable)
        }
        _ => {
            return Err(Error::new_spanned(
                arg,
                "expected #[comptime], #[columns(layout)], or an unannotated runtime argument",
            ));
        }
    };
    Ok(abi::Parameter {
        name,
        ty: &arg.ty,
        pattern: &arg.pat,
        binding,
    })
}

pub(super) fn compound_op(op: &syn::BinOp) -> Option<&'static str> {
    use syn::BinOp::*;
    match op {
        AddAssign(_) => Some("Add"),
        SubAssign(_) => Some("Sub"),
        MulAssign(_) => Some("Mul"),
        DivAssign(_) => Some("Div"),
        RemAssign(_) => Some("Rem"),
        BitAndAssign(_) => Some("And"),
        BitOrAssign(_) => Some("Or"),
        BitXorAssign(_) => Some("Xor"),
        ShlAssign(_) => Some("Shl"),
        ShrAssign(_) => Some("Shr"),
        _ => None,
    }
}

fn bind_pattern<T: Copy>(names: &mut HashMap<String, T>, pat: &Pat, binding: T) {
    visit_bindings(pat, &mut |pat| {
        names.insert(pat.ident.to_string(), binding);
    });
}

fn visit_bindings(pat: &Pat, visit: &mut impl FnMut(&syn::PatIdent)) {
    struct Bind<'a, F>(&'a mut F);
    impl<'ast, F: FnMut(&syn::PatIdent)> Visit<'ast> for Bind<'_, F> {
        fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
            self.0(pat);
            syn::visit::visit_pat_ident(self, pat);
        }
    }
    Bind(visit).visit_pat(pat);
}

fn named(pat: &Pat) -> Result<&syn::PatIdent> {
    match pat {
        Pat::Ident(p) if p.by_ref.is_none() && p.subpat.is_none() => Ok(p),
        _ => Err(Error::new_spanned(pat, "expected a named binding")),
    }
}

fn slice_type(ty: &syn::Type, types: &[&Ident]) -> Option<BufferType> {
    let syn::Type::Reference(reference) = ty else {
        return None;
    };
    let syn::Type::Slice(slice) = &*reference.elem else {
        return None;
    };
    Some(BufferType {
        elem: Elem::parse(&slice.elem, types)?,
        write: reference.mutability.is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::{Item, parse_quote};

    #[test]
    fn declaration_markers_reject_invalid_positions_and_duplicates() {
        let examples: [(syn::Item, &str); 5] = [
            (
                parse_quote!(
                    #[comptime]
                    fn f() {}
                ),
                "belongs on a kernel parameter",
            ),
            (
                parse_quote!(
                    fn f(
                        #[comptime]
                        #[comptime]
                        x: i32,
                    ) -> i32 {
                        x
                    }
                ),
                "unannotated runtime argument",
            ),
            (
                parse_quote!(impl Plan { fn read(#[comtime] &self) {} }),
                "expected #[comptime]",
            ),
            (
                parse_quote!(
                    #[kernel]
                    fn read() {}
                ),
                "duplicate #[kernel]",
            ),
            (
                parse_quote!(
                    trait Plan {
                        #[intrinsic("x")]
                        fn read(&self);
                    }
                ),
                "method with a body",
            ),
        ];
        for (item, diagnostic) in examples {
            let error = crate::declaration::expand(
                parse_quote!(::splice),
                crate::import::Lowering::Ir,
                false,
                emit!(#item),
            )
            .unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
        }
    }

    #[test]
    fn writes_require_a_mutable_slice() {
        let mut f = parse_quote! {
            fn edit(column: &[i32]) -> i32 { column[0] = 1; 0 }
        };
        let error = expand(parse_quote!(::splice), &mut f, false).unwrap_err();
        assert!(error.to_string().contains("requires &mut [i32]"));
    }

    #[test]
    fn all_generated_items_follow_the_kernel_configuration() {
        let mut f = parse_quote! {
            #[cfg(feature = "kernels")]
            #[cfg_attr(all(), inline, cfg(target_arch = "x86_64"))]
            fn sum(x: i32, y: i32) -> i32 { x + y }
        };
        let items = syn::parse2::<syn::File>(expand(parse_quote!(::splice), &mut f, true).unwrap())
            .unwrap()
            .items;
        assert_eq!(items.len(), 3);
        for item in items {
            let attrs = match item {
                Item::Fn(f) => f.attrs,
                Item::Mod(m) => m.attrs,
                Item::Impl(i) => i.attrs,
                _ => panic!("unexpected item in scalar kernel expansion"),
            };
            assert!(attrs.iter().any(|a| a.path().is_ident("cfg")));
            let nested = attrs
                .iter()
                .find(|a| a.path().is_ident("cfg_attr"))
                .unwrap();
            assert!(emit!(#nested).to_string().contains("target_arch"));
        }
    }
}
