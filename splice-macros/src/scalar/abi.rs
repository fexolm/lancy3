//! The generated Rust interface keeps the declared types and borrows. The
//! scalar backend still receives the usual independent register and buffer lists.
use super::{Binding, lower::Context};
use proc_macro2::{Ident, Span, TokenStream};
use std::collections::HashSet;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{
    FnArg, GenericParam, Generics, Lifetime, LifetimeParam, Pat, Path, ReturnType, Signature, Type,
};

pub(super) struct Parameter<'a> {
    pub(super) name: Ident,
    pub(super) ty: &'a Type,
    pub(super) pattern: &'a Pat,
    pub(super) binding: Binding,
}

pub(crate) fn parameter_names(sig: &Signature) -> impl Iterator<Item = Ident> {
    let mut used: HashSet<_> = sig
        .inputs
        .iter()
        .filter_map(explicit_name)
        .map(Ident::unraw_name)
        .collect();
    sig.inputs.iter().enumerate().map(move |(index, arg)| {
        if let Some(name) = explicit_name(arg) {
            return name.clone();
        }
        let mut name = format!("arg{}", index + 1);
        while used.contains(&name) {
            name.push('_');
        }
        used.insert(name.clone());
        Ident::new(&name, Span::mixed_site())
    })
}

fn explicit_name(arg: &FnArg) -> Option<&Ident> {
    let FnArg::Typed(arg) = arg else { return None };
    let Pat::Ident(pat) = &*arg.pat else {
        return None;
    };
    Some(&pat.ident)
}

trait UnrawName {
    fn unraw_name(&self) -> String;
}

impl UnrawName for Ident {
    fn unraw_name(&self) -> String {
        self.to_string().trim_start_matches("r#").to_owned()
    }
}

pub(super) struct Abi {
    pub(super) build_type: TokenStream,
    pub(super) signature: TokenStream,
    pub(super) build_generics: Generics,
    pub(super) items: TokenStream,
}

pub(super) fn generate(
    root: &Path,
    vis: &syn::Visibility,
    sig: &Signature,
    params: &[Parameter<'_>],
    unit_result: bool,
    context: Option<&Context>,
) -> Abi {
    let mut elision = Elision::new(&sig.generics);
    let mut host = Vec::new();
    let mut runtime = Vec::new();
    let mut written_buffers = Vec::new();
    let mut registers = Vec::new();
    let mut columns = None;
    let mut context_param = None;
    for param in params {
        let name = &param.name;
        let mut ty = param.ty.clone();
        elision.visit_type_mut(&mut ty);
        match param.binding {
            Binding::Aggregate(_) => unreachable!("aggregate local used as a formal parameter"),
            Binding::Host(_) => host.push((name, ty)),
            Binding::Reg(..) | Binding::Word(_) => {
                registers.push(emit!(#root::KernelScalar::to_word(self.#name)));
                runtime.push((name, ty));
            }
            Binding::Buffer(buffer) => {
                written_buffers.push((name, ty.clone(), Some(buffer.elem)));
                runtime.push((name, ty));
            }
            Binding::Context => {
                context_param = Some(name);
                written_buffers.push((name, ty.clone(), None));
                runtime.push((name, ty));
            }
            Binding::Columns => {
                columns = Some(name);
                runtime.push((name, ty));
            }
        }
    }
    let mut result_ty = match &sig.output {
        ReturnType::Type(_, ty) => (**ty).clone(),
        ReturnType::Default => syn::parse_quote!(()),
    };
    elision.visit_result(&mut result_ty);
    let api =
        crate::api::PublicApi::new(root, vis, &sig.generics, &elision.generics, &host, &runtime);
    let api_items = &api.items;
    let (_, host_args, _) = api.host.split_for_impl();
    let build_type = emit!(BuildArgs #host_args);
    let call_generics = api.call.clone();
    let (call_decls, call_args, _) = call_generics.split_for_impl();
    let mut signature_generics = call_generics.clone();
    signature_generics.params = signature_generics
        .params
        .into_iter()
        .filter(|p| !matches!(p, GenericParam::Lifetime(_)))
        .map(|mut p| {
            if let GenericParam::Type(t) = &mut p {
                t.bounds.clear();
            }
            p
        })
        .collect();
    signature_generics.where_clause = None;
    let (_, signature_args, _) = signature_generics.split_for_impl();
    let type_params = signature_generics.type_params().map(|t| &t.ident);
    let signature = emit!(Signature #signature_args);
    let marker = emit! {
        #[doc(hidden)]
        #vis struct Signature #signature_generics (::core::marker::PhantomData<fn(#(*const #type_params),*)>);
    };

    let mut invoke_generics = call_generics.clone();
    for (_, ty, elem) in &written_buffers {
        if matches!(elem, Some(super::Elem::Param(_))) {
            let Type::Reference(reference) = ty else {
                unreachable!("buffer parameters are references")
            };
            let Type::Slice(slice) = &*reference.elem else {
                unreachable!("buffer parameters are slices")
            };
            let element = &slice.elem;
            invoke_generics
                .make_where_clause()
                .predicates
                .push(syn::parse_quote!(
                    #element: #root::BufferElement
                ));
        }
    }
    let (_, _, invoke_where) = invoke_generics.split_for_impl();
    let buffers = written_buffers.iter().map(|(name, ty, elem)| {
        if elem.is_none() {
            return emit!(#root::BufferArg::context(&mut __splice_runtime_context));
        }
        let Type::Reference(reference) = ty else {
            unreachable!("buffer parameters are references")
        };
        if reference.mutability.is_some() {
            emit!(#root::BufferArg::write(self.#name))
        } else {
            emit!(#root::BufferArg::read(self.#name))
        }
    });
    let context_setup = context.zip(context_param).map(|(context, parameter)| {
        let types = sig.generics.type_params().map(|ty| &ty.ident).collect::<Vec<_>>();
        let invoke = context.dispatcher(root, emit!(&__SPLICE_CONTEXT_SIGNATURE), &types);
        emit! {
            let __splice_object = self.#parameter;
            let mut __splice_dispatch = |__splice_operation: u32, __splice_arguments: #root::__private::ContextArguments<'_>| -> u64 {
                #invoke
            };
            let mut __splice_runtime_context = #root::__private::RuntimeContext::new(&__SPLICE_CONTEXT_SIGNATURE, &mut __splice_dispatch);
        }
    });
    let buffers = if let Some(columns) = columns {
        emit! {
            let mut __splice_buffers = ::std::vec![#(#buffers),*];
            let __splice_fixed = __splice_buffers.len();
            for (__splice_index, __splice_column) in self.#columns.iter().enumerate() {
                // A context is not a column: report it at its position.
                __splice_buffers.push(__splice_column.as_read_only().ok_or(
                    #root::SignatureError::Context { buffer: __splice_fixed + __splice_index },
                )?);
            }
        }
    } else {
        emit!(let mut __splice_buffers = [#(#buffers),*];)
    };
    let decode = if unit_result {
        emit!({
            let _ = __splice_result;
        })
    } else {
        emit!(<#result_ty as #root::KernelScalar>::from_word(__splice_result))
    };
    let implementation = emit! {
        impl #call_decls #root::CallArgs for CallArgs #call_args #invoke_where {
            type Signature = #signature;
            type Output = #result_ty;
            fn invoke<__SpliceExecutable: #root::Executable>(self, executable: &__SpliceExecutable)
                -> ::core::result::Result<Self::Output, #root::CallError>
            {
                #root::__private::call_boundary(move || {
                    let __splice_registers = [#(#registers),*];
                    #context_setup
                    #buffers
                    let __splice_result = #root::Executable::call(
                        executable, &__splice_registers, &mut __splice_buffers,
                    )?;
                    ::core::result::Result::Ok(#decode)
                })
            }
        }
    };
    Abi {
        build_type,
        signature,
        build_generics: api.entry,
        items: emit!(#api_items #marker #implementation),
    }
}

struct GenericUses(HashSet<String>);

impl GenericUses {
    fn add_tokens(&mut self, tokens: TokenStream) {
        for token in tokens {
            match token {
                proc_macro2::TokenTree::Ident(ident) => {
                    self.0.insert(ident.to_string());
                }
                proc_macro2::TokenTree::Group(group) => self.add_tokens(group.stream()),
                _ => (),
            }
        }
    }
}

impl<'ast> Visit<'ast> for GenericUses {
    fn visit_path(&mut self, path: &'ast Path) {
        if path.leading_colon.is_none()
            && let Some(first) = path.segments.first()
        {
            self.0.insert(first.ident.to_string());
        }
        syn::visit::visit_path(self, path);
    }

    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime) {
        self.0.insert(lifetime.to_string());
    }

    fn visit_macro(&mut self, invocation: &'ast syn::Macro) {
        self.add_tokens(invocation.tokens.clone());
    }
}

pub(crate) fn param_name(param: &GenericParam) -> String {
    match param {
        GenericParam::Lifetime(p) => p.lifetime.to_string(),
        GenericParam::Type(p) => p.ident.to_string(),
        GenericParam::Const(p) => p.ident.to_string(),
    }
}

pub(crate) fn used_generics<'a>(
    generics: &Generics,
    types: impl Iterator<Item = &'a Type>,
) -> Generics {
    let mut used = GenericUses(HashSet::new());
    for ty in types {
        used.visit_type(ty);
    }
    select_generics(generics, used)
}

fn select_generics(generics: &Generics, used: GenericUses) -> Generics {
    let names: HashSet<_> = generics.params.iter().map(param_name).collect();
    let mut result = generics.clone();
    result.params = result
        .params
        .into_iter()
        .filter(|p| used.0.contains(&param_name(p)))
        .collect();
    let all_in_scope = |refs: GenericUses| {
        refs.0
            .intersection(&names)
            .all(|name| used.0.contains(name))
    };
    for param in &mut result.params {
        match param {
            GenericParam::Type(p) => {
                p.bounds = std::mem::take(&mut p.bounds)
                    .into_iter()
                    .filter(|bound| {
                        let mut refs = GenericUses(HashSet::new());
                        refs.visit_type_param_bound(bound);
                        all_in_scope(refs)
                    })
                    .collect()
            }
            GenericParam::Lifetime(p) => {
                p.bounds = std::mem::take(&mut p.bounds)
                    .into_iter()
                    .filter(|lifetime| {
                        !names.contains(&lifetime.to_string())
                            || used.0.contains(&lifetime.to_string())
                    })
                    .collect()
            }
            GenericParam::Const(_) => (),
        }
    }
    if let Some(clause) = &mut result.where_clause {
        clause.predicates = std::mem::take(&mut clause.predicates)
            .into_iter()
            .filter(|predicate| {
                let mut refs = GenericUses(HashSet::new());
                refs.visit_where_predicate(predicate);
                all_in_scope(refs)
            })
            .collect();
        if clause.predicates.is_empty() {
            result.where_clause = None;
        }
    }
    result
}

pub(crate) struct Elision {
    pub(crate) generics: Generics,
    inputs: HashSet<Lifetime>,
    next: usize,
}

impl Elision {
    pub(crate) fn new(generics: &Generics) -> Self {
        Self {
            generics: generics.clone(),
            inputs: HashSet::new(),
            next: 0,
        }
    }

    fn fresh(&mut self) -> Lifetime {
        loop {
            let lifetime =
                Lifetime::new(&format!("'__splice_arg_{}", self.next), Span::mixed_site());
            self.next += 1;
            if !self
                .generics
                .lifetimes()
                .any(|param| param.lifetime == lifetime)
            {
                self.generics
                    .params
                    .insert(0, LifetimeParam::new(lifetime.clone()).into());
                return lifetime;
            }
        }
    }

    pub(crate) fn visit_result(&self, ty: &mut Type) {
        if self.inputs.len() != 1 {
            return;
        }
        struct ResultElision(Lifetime);
        impl VisitMut for ResultElision {
            fn visit_type_reference_mut(&mut self, reference: &mut syn::TypeReference) {
                if reference.lifetime.is_none() {
                    reference.lifetime = Some(self.0.clone());
                }
                syn::visit_mut::visit_type_reference_mut(self, reference);
            }
            fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
                if lifetime.ident == "_" {
                    *lifetime = self.0.clone();
                }
            }
            fn visit_type_bare_fn_mut(&mut self, _: &mut syn::TypeBareFn) {}
        }
        ResultElision(self.inputs.iter().next().unwrap().clone()).visit_type_mut(ty);
    }
}

impl VisitMut for Elision {
    fn visit_type_reference_mut(&mut self, reference: &mut syn::TypeReference) {
        if reference.lifetime.is_none() {
            reference.lifetime = Some(self.fresh());
        }
        syn::visit_mut::visit_type_reference_mut(self, reference);
    }

    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        if lifetime.ident == "_" {
            *lifetime = self.fresh();
        }
        self.inputs.insert(lifetime.clone());
    }

    fn visit_type_bare_fn_mut(&mut self, _: &mut syn::TypeBareFn) {}
}
