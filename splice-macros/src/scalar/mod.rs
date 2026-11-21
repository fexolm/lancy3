//! Staged scalar functions. Rust executes host code while symbolic runtime
//! expressions append register operations and control-flow edges to the IR.
// Generated locals and labels share a hygiene context across fragments. Tokens
// interpolated from the kernel keep their original spans and lexical bindings.
macro_rules! emit {
    ($($tokens:tt)*) => { quote::quote_spanned!(proc_macro2::Span::mixed_site()=> $($tokens)*) };
}

mod flow;
mod lower;
use lower::{Binding, BufferType, Elem, Lower, Scalar};
use proc_macro2::{Ident, TokenStream};
use std::collections::HashMap;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{Error, FnArg, GenericParam, ItemFn, Meta, Pat, Path, Result, ReturnType, Type};

struct Native;
impl VisitMut for Native {
    fn visit_expr_mut(&mut self, e: &mut syn::Expr) {
        match e {
            syn::Expr::Macro(m) if m.mac.path.is_ident("comptime") => {
                if let Ok(expr) = m.mac.parse_body() {
                    *e = expr;
                    self.visit_expr_mut(e);
                }
            }
            _ => syn::visit_mut::visit_expr_mut(self, e),
        }
    }
}

// Only configuration gates belong on all generated items. A cfg_attr may
// also contain function-specific attributes such as inline or test.
fn configuration(meta: &Meta) -> Result<Option<TokenStream>> {
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

pub(super) fn expand(root: Path, mut native: ItemFn) -> Result<TokenStream> {
    let sig = &native.sig;
    let vis = &native.vis;
    let gates = native
        .attrs
        .iter()
        .filter_map(|a| configuration(&a.meta).transpose())
        .collect::<Result<Vec<_>>>()?;
    if sig.asyncness.is_some()
        || sig.unsafety.is_some()
        || sig.abi.is_some()
        || sig.variadic.is_some()
    {
        return Err(Error::new_spanned(
            sig,
            "expected an ordinary kernel function",
        ));
    }
    let ReturnType::Type(_, ty) = &sig.output else {
        return Err(Error::new_spanned(
            &sig.output,
            "expected a kernel return type",
        ));
    };
    let types: Vec<_> = sig.generics.type_params().map(|ty| &ty.ident).collect();
    let result_type = Scalar::from_type(ty, &types)
        .filter(|t| matches!(t, Scalar::Int | Scalar::Vector(_) | Scalar::Mask(_)))
        .ok_or_else(|| Error::new_spanned(ty, "expected i32, Lane<T> or Mask<T>"))?;
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
    let mut host_names = Vec::new();
    let mut helper_args = Vec::new();
    let mut inline_args = Vec::new();
    let mut reg_names = Vec::new();
    let mut buffer_names = Vec::new();
    let mut buffer_kinds = Vec::new();
    let mut columns = None;
    let mut column_parameter = None;
    let mut call_decls = Vec::new();
    let mut copy_mutable = Vec::new();
    let mut source_decls = Vec::new();
    let mut source_params = Vec::new();
    let mut source_args = Vec::new();
    for arg in &sig.inputs {
        let (id, ty, formal) = parameter(arg, &types)?;
        let source = emit!(#root::frontend::source);
        if matches!(formal, Binding::Host(_)) {
            source_decls.push(emit!(#support::Host(#id): #support::Host<#ty>));
        } else {
            source_decls.push(emit!(#id: impl #source::Argument));
            source_args.push(emit!(#source::Argument::source(#id)));
            let writable = matches!(formal, Binding::Reg(_, true));
            let prefix = if writable { "mut " } else { "" };
            let source_ty = if matches!(formal, Binding::Columns) {
                emit!(concat!("&[", stringify!(#root::BufferArg), "<'_>]"))
            } else {
                emit!(::core::any::type_name::<#ty>())
            };
            source_params.push(emit!(
                format!("{}{}: {}", #prefix, stringify!(#id), #source_ty)
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
            Binding::Host(_) => {
                host_names.push(id);
                host_types.push(ty);
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
            Binding::Reg(_, mutable) => {
                let index = reg_names.len();
                inline_args.push(emit!(__splice_builder.arg(#index)));
                reg_names.push(id);
                if mutable {
                    copy_mutable.push(emit!(let #id = __splice_builder.copy(#id);));
                }
                emit!(#id: impl #support::RunArg)
            }
        };
        call_decls.push(decl);
        bindings.insert(id.to_string(), formal);
        debug_names.insert(id.to_string(), id.clone());
        formals.push(formal);
    }
    let arity = reg_names.len();
    let host_decls = emit!(#(#host_names: #host_types,)*);
    let unpack = emit!(#(let #reg_names = #support::RunArg::reg(#reg_names, __splice_builder);)*);
    let flow = flow::analyze(&native.block, &bindings);
    let lower = Lower {
        root: &root,
        types: &types,
        name,
        result_type,
        formals: &formals,
        bindings,
        debug_names,
        flow: &flow,
        fish: emit!(#fish),
        loop_kind: None,
        region_depth: 0,
        source_fault: None,
    };
    let code = lower.kernel_body(&native.block)?;
    let debug_arguments = host_names.iter().map(|id| lower.capture_constant(id));
    let source_support = emit!(#root::frontend::source);
    let source_body = match lower.source_body(&native.block) {
        Ok(body) => emit!(let __splice_printed = #body;),
        Err(error) => {
            let error = error.to_string();
            emit!(return Err(#error.to_owned());)
        }
    };
    let source_aliases = types.iter().map(|ty| {
        emit!((
            stringify!(#ty).to_owned(),
            format!("type {} = {};", stringify!(#ty), ::core::any::type_name::<#ty>())
        ))
    }).chain(generic.const_params().map(|param| {
        let name = &param.ident;
        let ty = &param.ty;
        emit!((stringify!(#name).to_owned(), format!("const {}: {} = {};", stringify!(#name), stringify!(#ty), #source_support::literal(#name))))
    }));
    let source_entry = {
        let source_layout = columns.as_ref().map(|layout| emit! {
            let __splice_layout = #layout;
            let __splice_column_layout: &[#root::BufferKind] = ::core::convert::AsRef::as_ref(&__splice_layout);
        });
        let nested_source = if columns.is_none() {
            emit! {
                #[doc(hidden)]
                #[allow(clippy::too_many_arguments)]
                #vis fn __splice_source_call #generic (#(#source_decls,)* __splice_source: &mut #source_support::Source) -> #root::Result<String> #where_ {
                    let name = Self::__splice_print #fish (#(#host_names,)* __splice_source)?;
                    let arguments: &[String] = &[#(#source_args),*];
                    Ok(format!("{}({})", name, arguments.join(", ")))
                }
            }
        } else {
            emit!()
        };
        emit! {
            #vis fn print #generic (#host_decls) -> #root::Result<String> #where_ {
                let mut source = #source_support::Source::default();
                Self::__splice_print #fish (#(#host_names,)* &mut source)?;
                Ok(format!("#[allow(unused_imports)]\nuse {}::simd::{{Lane, Mask}};\n\n{}", stringify!(#root), source.functions.join("\n")))
            }
            #[allow(unused_variables, unreachable_code)]
            fn __splice_print #generic (#host_decls __splice_source: &mut #source_support::Source) -> #root::Result<String> #where_ {
                let _guard = #support::enter_expansion()?;
                let (id, name) = __splice_source.function(stringify!(#name));
                let transfer = __splice_source.transfer.take();
                #source_layout
                let __splice_printed: String;
                #source_body
                __splice_source.transfer = transfer;
                __splice_source.functions[id] = #source_support::function(&name, &[#(#source_params),*], ::core::any::type_name::<#ty>(), &[#(#source_aliases),*], &__splice_printed);
                Ok(name)
            }
            #nested_source
        }
    };
    let kinds = emit!(const { &[#(#buffer_kinds),*] });
    // Rust requires lifetime parameters before the backend type; the backend
    // is still the first argument at the usual lifetime-elided turbofish.
    let mut compile_generics = generic.clone();
    let backend = emit!(__SpliceCodegen);
    let lifetimes = generic.lifetimes().count();
    compile_generics
        .params
        .insert(lifetimes, syn::parse_quote!(#backend: #root::Codegen));
    let callback = emit!(::std::boxed::Box::new(move |__splice_builder| Self::__stage_body #fish (#(#host_names,)* __splice_builder)));
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
        let mut __splice_key = Some(#frontend::Specialization::new::<Self>(#arity, #kinds));
        #((&#frontend::KeyArgument::<#type_args>::for_type()).add_type(&mut __splice_key);)*
        #((&#frontend::KeyArgument::for_value(&#value_args)).add_value(&#value_args, &mut __splice_key);)*
    };
    let typed_helper = result_type != Scalar::Int
        || formals
            .iter()
            .any(|f| matches!(f, Binding::Reg(Scalar::Vector(_) | Scalar::Mask(_), _)));
    let result_value_type = result_type.value_type(&root, &types);
    let source = emit! {
        __splice_builder.set_source(&#root::frontend::KernelSource {
            name: stringify!(#name),
            scalars: &[#(stringify!(#reg_names)),*],
            buffers: &[#(stringify!(#buffer_names)),*],
        });
    };
    let standalone = emit! {
            #vis fn build #compile_generics (#host_decls) -> #root::Result<<#backend as #root::Codegen>::Output> #where_ {
                Self::build_with_options::<#backend #(, #args)*>(
                    #root::BuildOptions::for_profile(cfg!(debug_assertions)), #(#host_names,)*
                )
            }
            #vis fn build_with_options #compile_generics (__splice_options: #root::BuildOptions, #host_decls) -> #root::Result<<#backend as #root::Codegen>::Output> #where_ {
                <#backend as #root::Codegen>::build_with_options(#arity, #kinds, #callback, __splice_options)
            }
            #vis fn ir #generic (#host_decls) -> #root::Result<#hir::Function> #where_ {
                Self::ir_with_options #fish (#root::BuildOptions::for_profile(cfg!(debug_assertions)), #(#host_names,)*)
            }
            #vis fn ir_with_options #generic (__splice_options: #root::BuildOptions, #host_decls) -> #root::Result<#hir::Function> #where_ {
                #root::frontend::materialize_with_options(#arity, #kinds, #callback, __splice_options)
            }
            fn __stage_body #generic (#host_decls __splice_builder: &mut #factory) -> #root::Result<()> #where_ {
                #source
                let output = Self::__splice_inline #fish (#(#inline_args,)* __splice_builder)?;
                __splice_builder.ret(output);
                Ok(())
            }
            #[doc(hidden)]
            #[allow(clippy::too_many_arguments)]
            #vis fn __splice_call #generic (#(#call_decls,)* __splice_builder: &mut #factory)
                -> #root::Result<#hir::Reg> #where_ {
                #unpack
                #specialize
                let __splice_body: #frontend::Body<'_> = #callback;
                match __splice_key {
                    Some(key) => __splice_builder.call_specialized(key, &[#(#reg_names),*], &[#(#buffer_names),*], __splice_body),
                    None => __splice_builder.call_kernel(#arity, #kinds, &[#(#reg_names),*], &[#(#buffer_names),*], __splice_body),
                }
            }

    };
    // The public kernel ABI transports i32 scalars and slices. SIMD helpers
    // expand into their caller so vectors and masks stay in typed registers.
    let entry_points = if let Some(layout) = columns {
        if typed_helper {
            return Err(Error::new_spanned(
                sig,
                "a column-table entry point requires the scalar calling convention",
            ));
        }
        let base = buffer_names.len() as u32;
        let setup = emit! {
            let __splice_layout = #layout;
            let __splice_columns: &[#hir::BufferKind] = ::core::convert::AsRef::as_ref(&__splice_layout);
            let __splice_column_base = #base;
            let mut __splice_kinds = (#kinds).to_vec();
            __splice_kinds.extend_from_slice(__splice_columns);
            let __splice_body: #frontend::Body<'_> = ::std::boxed::Box::new(move |__splice_builder| {
                #source
                let output = Self::__splice_inline #fish (#(#inline_args,)* __splice_builder)?;
                __splice_builder.ret(output);
                Ok(())
            });
        };
        emit! {
            #vis fn build #compile_generics (#host_decls) -> #root::Result<<#backend as #root::Codegen>::Output> #where_ {
                Self::build_with_options::<#backend #(, #args)*>(
                    #root::BuildOptions::for_profile(cfg!(debug_assertions)), #(#host_names,)*
                )
            }
            #vis fn build_with_options #compile_generics (__splice_options: #root::BuildOptions, #host_decls) -> #root::Result<<#backend as #root::Codegen>::Output> #where_ {
                #setup
                <#backend as #root::Codegen>::build_with_options(#arity, &__splice_kinds, __splice_body, __splice_options)
            }
            #vis fn ir #generic (#host_decls) -> #root::Result<#hir::Function> #where_ {
                Self::ir_with_options #fish (#root::BuildOptions::for_profile(cfg!(debug_assertions)), #(#host_names,)*)
            }
            #vis fn ir_with_options #generic (__splice_options: #root::BuildOptions, #host_decls) -> #root::Result<#hir::Function> #where_ {
                #setup
                #root::frontend::materialize_with_options(#arity, &__splice_kinds, __splice_body, __splice_options)
            }
        }
    } else if typed_helper {
        emit! {
            #[doc(hidden)]
            #[allow(clippy::too_many_arguments)]
            #vis fn __splice_call #generic (#(#call_decls,)* __splice_builder:&mut #factory)->#root::Result<#hir::Reg> #where_{
                Self::__splice_inline #fish (#(#helper_args,)* __splice_builder)
            }
        }
    } else {
        standalone
    };
    let generated = emit! {
        #(#[#gates])*
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        #vis enum #name {}
        #(#[#gates])*
        impl #name {
            #[doc(hidden)]
            #vis const __SPLICE_RETURN: #hir::ValueType = #result_value_type;
            #entry_points
            #source_entry
            #[allow(unreachable_code, clippy::too_many_arguments)]
            fn __splice_inline #generic (#(#call_decls,)* __splice_builder:&mut #factory)
                -> #root::Result<#hir::Reg> #where_ {
                let __splice_guard = #support::enter_expansion()?;
                let __splice_wants_debug = __splice_builder.debug_enabled();
                #unpack
                #(#copy_mutable)*
                #(#debug_arguments)*
                let __splice_result = __splice_builder.typed_local(#result_value_type);
                let __splice_return_label = __splice_builder.label();
                #code
                __splice_builder.mark(__splice_return_label);
                Ok(__splice_result)
            }
        }
    };
    for arg in &mut native.sig.inputs {
        if let FnArg::Typed(arg) = arg {
            arg.attrs.clear();
        }
    }
    Native.visit_block_mut(&mut native.block);
    if let Some(id) = column_parameter {
        native.block.stmts.insert(
            0,
            syn::parse_quote!(let #id: &[#root::BufferArg<'_>] = #id;),
        );
    }
    native.attrs.push(syn::parse_quote!(#[allow(dead_code)]));
    Ok(emit!(#native #generated))
}

fn parameter<'a>(arg: &'a FnArg, types: &[&Ident]) -> Result<(&'a Ident, &'a Type, Binding)> {
    let FnArg::Typed(arg) = arg else {
        return Err(Error::new_spanned(arg, "kernel methods are unsupported"));
    };
    let pat = named(&arg.pat)?;
    let mutable = pat.mutability.is_some();
    let binding = match arg.attrs.as_slice() {
        [attr] if matches!(&attr.meta, Meta::Path(p) if p.is_ident("comptime")) => {
            if mutable {
                return Err(Error::new_spanned(pat, "comptime arguments are immutable"));
            }
            Binding::Host(0)
        }
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
            Some(Scalar::Int | Scalar::Vector(_) | Scalar::Mask(_))
        ) =>
        {
            Binding::Reg(Scalar::from_type(&arg.ty, types).unwrap(), mutable)
        }
        [] => {
            let writable = slice_type(&arg.ty, types).ok_or_else(|| {
                Error::new_spanned(
                    &arg.ty,
                    "runtime parameters must be i32 or a slice of u16, u32, i32, i64, f32, or f64",
                )
            })?;
            if mutable {
                return Err(Error::new_spanned(
                    pat,
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
    Ok((&pat.ident, &arg.ty, binding))
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
    fn a_load_cannot_be_used_as_a_recursive_comptime_argument() {
        let f = parse_quote! {
            fn walk(#[comptime] depth: i32, column: &[i32]) -> i32 {
                if depth == 0 { 1 } else { walk(column[0], column) }
            }
        };
        let error = expand(parse_quote!(::splice), f).unwrap_err();
        assert!(error.to_string().contains("expected a comptime value"));
    }

    #[test]
    fn standard_library_cannot_consume_runtime_values() {
        for (body, diagnostic) in [
            (
                parse_quote!({
                    let slots = vec![x; 2];
                    slots[0]
                }),
                "runtime value used in a comptime expression",
            ),
            (
                parse_quote!({ std::vec![0, x][0] }),
                "runtime value used in a comptime expression",
            ),
            (
                parse_quote!({
                    let mut seen = 0;
                    let _slots = vec![
                        {
                            if x > 0 {
                                seen = 1;
                            }
                            5
                        };
                        2
                    ];
                    seen
                }),
                "runtime value used in a comptime expression",
            ),
            (
                parse_quote!({ vec![0; x as usize].len() as i32 }),
                "runtime value used in a comptime expression",
            ),
            (
                parse_quote!({
                    let slots = vec![0; 2];
                    slots[x as usize]
                }),
                "index a runtime",
            ),
            (
                parse_quote!({
                    let mut slots = vec![0; 2];
                    slots[0] = x;
                    slots[0]
                }),
                "index a runtime",
            ),
            (
                parse_quote!({
                    let mut slots = Vec::new();
                    slots.push(x);
                    0
                }),
                "host method cannot consume runtime values",
            ),
            (
                parse_quote!({ std::cmp::max(x, 3) }),
                "standard library calls require comptime arguments",
            ),
            (
                parse_quote!({ comptime!(vec![x; 2].len()) as i32 }),
                "runtime value used in a comptime expression",
            ),
            (
                parse_quote!({
                    print!("{x}");
                    x
                }),
                "expected a let binding or scalar kernel expression",
            ),
        ] {
            let mut f: ItemFn = parse_quote!(
                fn example(x: i32) -> i32 {
                    x
                }
            );
            f.block = body;
            let error = expand(parse_quote!(::splice), f).unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
        }
    }

    #[test]
    fn writes_require_a_mutable_slice() {
        let f = parse_quote! {
            fn edit(column: &[i32]) -> i32 { column[0] = 1; 0 }
        };
        let error = expand(parse_quote!(::splice), f).unwrap_err();
        assert!(error.to_string().contains("requires &mut [i32]"));
    }

    #[test]
    fn all_generated_items_follow_the_kernel_configuration() {
        let f = parse_quote! {
            #[cfg(feature = "kernels")]
            #[cfg_attr(all(), inline, cfg(target_arch = "x86_64"))]
            fn sum(x: i32, y: i32) -> i32 { x + y }
        };
        let items = syn::parse2::<syn::File>(expand(parse_quote!(::splice), f).unwrap())
            .unwrap()
            .items;
        assert_eq!(items.len(), 3);
        for item in items {
            let attrs = match item {
                Item::Fn(f) => f.attrs,
                Item::Enum(e) => e.attrs,
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
