//! Select the marked Rust trait implementation at a qualified call site.
use super::declaration::TraitMethod;
use super::{entry_name, trait_path};
use proc_macro2::{Ident, Span, TokenStream};
use syn::ext::IdentExt;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{GenericParam, Generics, Lifetime, Type};

pub(crate) fn isolated_trait_call(
    root: &syn::Path,
    generics: &syn::Generics,
    call: &syn::ExprPath,
    arguments: &[TokenStream],
    opaque: TokenStream,
    required: bool,
) -> TokenStream {
    let qualifier = call.qself.as_ref().expect("a trait-qualified call");
    let trait_path = trait_path(call).expect("an explicit trait owner");
    if !matches!(
        call.path.segments.last().unwrap().arguments,
        syn::PathArguments::None
    ) {
        return opaque;
    }
    let owner = &qualifier.ty;
    let mut uses_self = crate::declaration::UsesSelf::default();
    uses_self.visit_path(&trait_path);
    if uses_self.0 {
        return opaque;
    }
    let Some(captured) = captured_generics(&trait_path, generics) else {
        return opaque;
    };
    uses_self.visit_generics(&captured);
    if uses_self.0 {
        return opaque;
    }
    let owner_param = probe_ident(&captured, "__SpliceOwner");
    let probe_arguments = probe_ident(&captured, "__splice_probe_arguments");
    let probe_builder = probe_ident(&captured, "__splice_probe_builder");
    let arguments_for_probe = captured.params.iter().map(generic_argument);
    let mut probe = captured.clone();
    probe.params.push(syn::parse_quote!(#owner_param: ?Sized));
    let witnesses = probe.params.iter().filter_map(|parameter| match parameter {
        GenericParam::Lifetime(parameter) => {
            let lifetime = &parameter.lifetime;
            Some(emit!(&#lifetime ()))
        }
        GenericParam::Type(parameter) => {
            let ident = &parameter.ident;
            Some(emit!(*const #ident))
        }
        GenericParam::Const(_) => None,
    });
    let (probe_params, probe_type, probe_where) = probe.split_for_impl();
    let mut marked = probe.clone();
    marked
        .make_where_clause()
        .predicates
        .push(syn::parse_quote!(#owner_param: #trait_path + Sized));
    let (marked_params, _, marked_where) = marked.split_for_impl();
    let name = &call.path.segments.last().unwrap().ident;
    let word_marker = TraitMethod::Scalar.marker(name);
    let borrowed_marker = TraitMethod::Borrowed.marker(name);
    let owned_marker = TraitMethod::Owned.marker(name);
    let selector = entry_name(
        if required {
            "eval_trait_words"
        } else {
            "stage_trait_words"
        },
        name,
    );
    let support = emit!(#root::frontend::scalar);
    let registry = emit!(#root::frontend::trait_methods);
    let (params, forward, output, materialize, result, invoke_receiver) = if required {
        (
            emit!(#probe_arguments: &[Option<i32>]),
            emit!(#probe_arguments),
            emit!(i32),
            emit! {
                trait __SpliceWordOperand {
                    fn word(self) -> #root::Result<Option<i32>>;
                }
                impl<T: #support::ScalarConstant + Copy> __SpliceWordOperand for &Option<T> {
                    fn word(self) -> #root::Result<Option<i32>> {
                        Ok(self.map(#support::ScalarConstant::bits))
                    }
                }
                impl<T> __SpliceWordOperand for &&Option<T> {
                    fn word(self) -> #root::Result<Option<i32>> {
                        Err("a trait method requires a scalar comptime operand".into())
                    }
                }
            },
            emit!(#support::ImportWord::decode(match registered {
                Some(#registry::Method::Scalar(method)) => (method.evaluate)(&operands)?,
                _ => (&probe).run(&operands)?,
            })),
            emit!(#support::ImportWord::decode((&mut receiver).evaluate_receiver(adapter, &operands)?)),
        )
    } else {
        (
            emit!(#probe_arguments: &[#support::Word], #probe_builder: &mut dyn #root::frontend::KernelBuilder),
            emit!(#probe_arguments, #probe_builder),
            emit!(#root::hir::Reg),
            emit! {
                struct __SpliceWordOperand<T>(T);
                trait __SpliceReadWord {
                    fn word(self, b: &mut dyn #root::frontend::KernelBuilder) -> #root::Result<#support::Word>;
                }
                impl<T: #support::RunArg + Copy> __SpliceReadWord for &__SpliceWordOperand<T> {
                    fn word(self, b: &mut dyn #root::frontend::KernelBuilder) -> #root::Result<#support::Word> {
                        if let Some(bits) = self.0.known() {
                            Ok(#support::Word::Known(bits))
                        } else {
                            Ok(#support::Word::Runtime(self.0.reg(b)))
                        }
                    }
                }
                impl<T> __SpliceReadWord for &&__SpliceWordOperand<T> {
                    fn word(self, _: &mut dyn #root::frontend::KernelBuilder) -> #root::Result<#support::Word> {
                        Err("a trait method requires a scalar operand".into())
                    }
                }
            },
            emit!(match registered {
                Some(#registry::Method::Scalar(method)) => (method.stage)(&operands, __splice_builder)?,
                _ => (&probe).run(&operands, __splice_builder)?,
            }),
            emit!((&mut receiver).stage_receiver(adapter, &operands, __splice_builder)?),
        )
    };
    let convert = |index: usize| {
        let index = syn::Index::from(index);
        if required {
            emit!((&values.#index).word()?)
        } else {
            emit!((&__SpliceWordOperand(values.#index)).word(__splice_builder)?)
        }
    };
    let operands = (0..arguments.len()).map(convert);
    let select_receiver = if !arguments.is_empty()
        && receiver_type_parameters_are_static(owner, generics)
    {
        let (family, lifetimes) = receiver_family(owner, generics);
        let general = relax_static_lifetimes(&family);
        let variants = [
            (emit!((&probe).borrowed()), emit!(&)),
            (emit!((&probe).owned()), TokenStream::new()),
        ];
        variants.into_iter().map(|(marker, reference)| {
            let (eval, stage) = receiver_adapter_types(root, &emit!(#reference #family), &lifetimes);
            let entry = syn::Index::from(usize::from(!required));
            let pointer = if required { &eval } else { &stage };
            let general = general.as_ref().map(|family| {
                let (eval, stage) = receiver_adapter_types(root, &emit!(#reference #family), &lifetimes);
                emit! {
                    .or_else(|| {
                        let adapters = registered.as_ref()?.receiver_adapter::<(#eval, #stage)>()?;
                        let adapter: #pointer = adapters.#entry;
                        Some(adapter)
                    })
                }
            });
            let operands = (1..arguments.len()).map(convert);
            emit! {
                else if #marker
                    && let Some(adapter) = registered.as_ref()
                        .and_then(#registry::Method::receiver_adapter::<(#eval, #stage)>)
                        .map(|adapters| adapters.#entry)
                        #general
                {
                    use #registry::InvokeReceiver as _;
                    let values = (#(#arguments,)*);
                    let mut receiver = #registry::ReceiverArgument::<#reference #owner, _>::new(values.0);
                    let operands = {
                        #materialize
                        [#(#operands),*]
                    };
                    #invoke_receiver
                }
            }
        }).collect::<TokenStream>()
    } else {
        TokenStream::new()
    };
    let callable_parameters = arguments.iter().map(|_| emit!(_));
    emit!({
        struct __SpliceTraitOwner #probe_params (::core::marker::PhantomData<(#(#witnesses,)*)>) #probe_where;
        trait __SpliceSelectTrait {
            fn words(self) -> bool;
            fn borrowed(self) -> bool;
            fn owned(self) -> bool;
            fn run(self, #params) -> #root::Result<#output>;
        }
        impl #marked_params __SpliceSelectTrait for &__SpliceTraitOwner #probe_type #marked_where {
            fn words(self) -> bool {
                trait __SpliceOpaqueTrait {
                    fn #word_marker() -> bool { false }
                }
                impl<U: ?Sized> __SpliceOpaqueTrait for U {}
                #owner_param::#word_marker()
            }
            fn borrowed(self) -> bool {
                trait __SpliceOpaqueTrait {
                    fn #borrowed_marker() -> bool { false }
                }
                impl<U: ?Sized> __SpliceOpaqueTrait for U {}
                #owner_param::#borrowed_marker()
            }
            fn owned(self) -> bool {
                trait __SpliceOpaqueTrait {
                    fn #owned_marker() -> bool { false }
                }
                impl<U: ?Sized> __SpliceOpaqueTrait for U {}
                #owner_param::#owned_marker()
            }
            fn run(self, #params) -> #root::Result<#output> {
                trait __SpliceOpaqueTrait {
                    #[allow(unused_variables)]
                    fn #selector(#params) -> #root::Result<#output> {
                        Err("this trait method has no Splice implementation".into())
                    }
                }
                impl<U: ?Sized> __SpliceOpaqueTrait for U {}
                #owner_param::#selector(#forward)
            }
        }
        impl #probe_params __SpliceSelectTrait for &&__SpliceTraitOwner #probe_type #probe_where {
            fn words(self) -> bool { false }
            fn borrowed(self) -> bool { false }
            fn owned(self) -> bool { false }
            #[allow(unused_variables)]
            fn run(self, #params) -> #root::Result<#output> {
                Err("this trait method has no Splice implementation".into())
            }
        }
        let probe = __SpliceTraitOwner::<#(#arguments_for_probe,)* #owner>(::core::marker::PhantomData);
        let registered = if (&probe).words() || (&probe).borrowed() || (&probe).owned() {
            // Taking the address instantiates the selected method, including its
            // loader record when the implementation is individually annotated.
            ::core::hint::black_box(#call as fn(#(#callable_parameters),*) -> _);
            #registry::lookup(#call)
        } else {
            None
        };
        if (&probe).words() {
            let values = (#(#arguments,)*);
            let operands = {
                #materialize
                [#(#operands),*]
            };
            #result
        } #select_receiver else {
            #opaque
        }
    })
}

fn relax_static_lifetimes(owner: &Type) -> Option<Type> {
    let Type::Path(mut owner) = owner.clone() else {
        return None;
    };
    let mut changed = false;
    for segment in &mut owner.path.segments {
        if let syn::PathArguments::AngleBracketed(arguments) = &mut segment.arguments {
            for argument in &mut arguments.args {
                if let syn::GenericArgument::Lifetime(lifetime) = argument
                    && lifetime.ident == "static"
                {
                    *lifetime = Lifetime::new("'_", lifetime.span());
                    changed = true;
                }
            }
        }
    }
    changed.then_some(Type::Path(owner))
}

fn generic_argument(parameter: &GenericParam) -> TokenStream {
    match parameter {
        GenericParam::Lifetime(parameter) => {
            let lifetime = &parameter.lifetime;
            emit!(#lifetime)
        }
        GenericParam::Type(parameter) => {
            let ident = &parameter.ident;
            emit!(#ident)
        }
        GenericParam::Const(parameter) => {
            let ident = &parameter.ident;
            emit!(#ident)
        }
    }
}

fn probe_ident(generics: &Generics, base: &str) -> Ident {
    let mut name = base.to_owned();
    while generics
        .type_params()
        .any(|param| param.ident.unraw() == name)
        || generics
            .const_params()
            .any(|param| param.ident.unraw() == name)
    {
        name.push('_');
    }
    Ident::new(&name, Span::mixed_site())
}

fn captured_generics(trait_path: &syn::Path, generics: &Generics) -> Option<Generics> {
    struct Capture<'a> {
        generics: &'a Generics,
        captures: bool,
        inferred: bool,
    }

    impl<'ast> Visit<'ast> for Capture<'_> {
        fn visit_path(&mut self, path: &'ast syn::Path) {
            if path.leading_colon.is_none()
                && let Some(first) = path.segments.first()
            {
                let ident = first.ident.unraw();
                self.captures |= self
                    .generics
                    .type_params()
                    .any(|param| param.ident.unraw() == ident)
                    || self
                        .generics
                        .const_params()
                        .any(|param| param.ident.unraw() == ident);
            }
            syn::visit::visit_path(self, path);
        }

        fn visit_lifetime(&mut self, lifetime: &'ast syn::Lifetime) {
            self.inferred |= lifetime.ident == "_";
            self.captures |= self
                .generics
                .lifetimes()
                .any(|param| param.lifetime == *lifetime);
        }

        fn visit_type_reference(&mut self, reference: &'ast syn::TypeReference) {
            self.inferred |= reference.lifetime.is_none();
            syn::visit::visit_type_reference(self, reference);
        }

        fn visit_type_infer(&mut self, _: &'ast syn::TypeInfer) {
            self.inferred = true;
        }
    }

    let mut capture = Capture {
        generics,
        captures: false,
        inferred: false,
    };
    capture.visit_path(trait_path);
    if capture.inferred {
        // Preserve the original call when a type or lifetime is only
        // inferred at that expression. An impl bound needs explicit arguments.
        None
    } else if capture.captures {
        // Keep the original bounds, which may tie a trait argument to the
        // other parameters or to projections in the caller's where clause.
        Some(generics.clone())
    } else {
        Some(Generics::default())
    }
}

pub(super) fn receiver_adapter_types(
    root: &syn::Path,
    receiver: &TokenStream,
    lifetimes: &[Lifetime],
) -> (TokenStream, TokenStream) {
    let higher_ranked = (!lifetimes.is_empty()).then(|| emit!(for<#(#lifetimes),*>));
    (
        emit!(#higher_ranked fn(#receiver, &[Option<i32>]) -> #root::Result<i32>),
        emit!(#higher_ranked fn(#receiver, &[#root::frontend::scalar::Word], &mut dyn #root::frontend::KernelBuilder) -> #root::Result<#root::hir::Reg>),
    )
}

pub(super) fn receiver_family(owner: &Type, generics: &Generics) -> (Type, Vec<Lifetime>) {
    struct Family<'a> {
        generics: &'a Generics,
        lifetimes: Vec<(Lifetime, Lifetime)>,
        bound: Vec<Ident>,
    }

    impl Family<'_> {
        fn enter(&mut self, lifetimes: Option<&syn::BoundLifetimes>) -> usize {
            let depth = self.bound.len();
            self.bound.extend(
                lifetimes
                    .into_iter()
                    .flat_map(|bounds| bounds.lifetimes.iter())
                    .filter_map(|parameter| match parameter {
                        GenericParam::Lifetime(parameter) => Some(parameter.lifetime.ident.clone()),
                        _ => None,
                    }),
            );
            depth
        }
    }

    impl VisitMut for Family<'_> {
        fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
            if lifetime.ident == "static"
                || lifetime.ident == "_"
                || self.bound.contains(&lifetime.ident)
            {
                return;
            }
            if let Some((_, existing)) = self
                .lifetimes
                .iter()
                .find(|(name, _)| name.ident == lifetime.ident)
            {
                *lifetime = existing.clone();
                return;
            }
            let mut name = format!("__splice_receiver_lifetime_{}", self.lifetimes.len());
            while self
                .generics
                .lifetimes()
                .any(|param| param.lifetime.ident == name)
                || self.bound.iter().any(|bound| *bound == name)
            {
                name.push('_');
            }
            let replacement = Lifetime::new(&format!("'{name}"), Span::mixed_site());
            self.lifetimes.push((lifetime.clone(), replacement.clone()));
            *lifetime = replacement;
        }

        fn visit_type_bare_fn_mut(&mut self, function: &mut syn::TypeBareFn) {
            let depth = self.enter(function.lifetimes.as_ref());
            syn::visit_mut::visit_type_bare_fn_mut(self, function);
            self.bound.truncate(depth);
        }

        fn visit_trait_bound_mut(&mut self, bound: &mut syn::TraitBound) {
            let depth = self.enter(bound.lifetimes.as_ref());
            syn::visit_mut::visit_trait_bound_mut(self, bound);
            self.bound.truncate(depth);
        }
    }

    let mut family = Family {
        generics,
        lifetimes: Vec::new(),
        bound: Vec::new(),
    };
    let mut owner = owner.clone();
    family.visit_type_mut(&mut owner);
    (
        owner,
        family
            .lifetimes
            .into_iter()
            .map(|(_, bound)| bound)
            .collect(),
    )
}

pub(super) fn receiver_type_parameters_are_static(owner: &Type, generics: &Generics) -> bool {
    fn has_static_bound(
        bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>,
    ) -> bool {
        bounds.iter().any(|bound| matches!(bound, syn::TypeParamBound::Lifetime(lifetime) if lifetime.ident == "static"))
    }

    struct ReceiverType<'a> {
        generics: &'a syn::Generics,
        allowed: bool,
    }

    impl<'ast> syn::visit::Visit<'ast> for ReceiverType<'_> {
        fn visit_type_path(&mut self, ty: &'ast syn::TypePath) {
            self.allowed &= !ty.path.is_ident("Self");
            if let Some(first) = ty.path.segments.first()
                && let Some(param) = self
                    .generics
                    .type_params()
                    .find(|param| param.ident == first.ident)
            {
                self.allowed &= ty.path.segments.len() == 1
                    && (has_static_bound(&param.bounds)
                        || self.generics.where_clause.iter().flat_map(|clause| &clause.predicates).any(|predicate| {
                            matches!(predicate, syn::WherePredicate::Type(predicate)
                                if matches!(&predicate.bounded_ty, Type::Path(path) if path.path.is_ident(&param.ident))
                                    && has_static_bound(&predicate.bounds))
                        }));
            }
            self.allowed &= ty.qself.is_none();
            syn::visit::visit_type_path(self, ty);
        }

        fn visit_type_infer(&mut self, _: &'ast syn::TypeInfer) {
            self.allowed = false;
        }
    }

    let mut check = ReceiverType {
        generics,
        allowed: true,
    };
    syn::visit::Visit::visit_type(&mut check, owner);
    check.allowed
}
