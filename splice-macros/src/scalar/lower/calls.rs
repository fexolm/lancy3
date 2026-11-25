//! Resolve scalar calls against the published kernel or the Rust function.
use super::{Binding, Lower, Scalar, Stage, Value, err};
use proc_macro2::TokenStream;
use syn::{Expr, Result};

impl Lower<'_> {
    pub(super) fn call(
        &self,
        c: &syn::ExprCall,
        hint: Option<Scalar>,
        cast: Option<&syn::Type>,
    ) -> Result<Value> {
        let Expr::Path(p) = &*c.func else {
            return err(&c.func, "expected a kernel function path");
        };
        if p.path
            .get_ident()
            .is_some_and(|name| self.bindings.contains_key(&name.to_string()))
        {
            return err(
                &c.func,
                "a Splice kernel calls functions by path, not through a local function value",
            );
        }
        let root = self.root;
        let support = emit!(#root::frontend::scalar);
        let values = c
            .args
            .iter()
            .map(|a| self.expr(a))
            .collect::<Result<Vec<_>>>()?;
        let trait_owner = crate::scalar::methods::trait_path(p);
        let method = trait_owner.is_none() && crate::scalar::methods::method_path(p);
        // A free item can be imported by path in both the value namespace
        // (Rust function) and the type namespace (published kernel module).
        let named = p.qself.is_none()
            && !crate::scalar::methods::method_path(p)
            && !p.path.is_ident(self.name);
        let recursive = p.path.is_ident(self.name);
        if recursive && self.formals.iter().any(|p| matches!(p, Binding::Columns)) {
            return err(c, "a column-table entry point cannot call itself");
        }
        let output = if recursive {
            self.result_type
        } else {
            hint.filter(|t| matches!(t, Scalar::Bool | Scalar::Vector(_) | Scalar::Mask(_)))
                .unwrap_or(Scalar::Int)
        };
        let mut check = TokenStream::new();
        let target = if recursive {
            if !self.formals.iter().any(|p| matches!(p, Binding::Host(_))) {
                return err(
                    c,
                    "runtime recursion is unsupported; recursive expansion needs a terminating #[comptime] plan",
                );
            }
            if values.len() != self.formals.len() {
                return err(c, "wrong number of recursive arguments");
            }
            let fish = &self.fish;
            emit!(Self::__splice_inline #fish)
        } else if trait_owner.is_some() {
            let stage = crate::scalar::methods::rename_path(p, "stage_method");
            emit!(#stage)
        } else if method {
            let stage = crate::scalar::methods::rename_path(p, "stage_inherent");
            let name = &stage.path.segments.last().unwrap().ident;
            let published = crate::scalar::methods::entry_name(
                "published",
                &p.path.segments.last().unwrap().ident,
            );
            let parameters = (0..values.len())
                .map(|i| quote::format_ident!("__SpliceArg{i}"))
                .collect::<Vec<_>>();
            check = emit! {
                trait __SpliceOpaqueMethod {
                    fn #published() -> bool { false }
                    fn #name<#(#parameters),*>(#(_: #parameters,)* _: &mut dyn #root::frontend::KernelBuilder)
                        -> #root::Result<#root::hir::Reg>
                    {
                        ::core::result::Result::Err(#root::BuildError::staging("method does not have a scalar Splice implementation"))
                    }
                }
                impl<T: ?Sized> __SpliceOpaqueMethod for T {}
            };
            emit!(#stage)
        } else {
            let mut path = p.path.clone();
            let fish = std::mem::take(&mut path.segments.last_mut().unwrap().arguments);
            let expected = output.value_type(root, self.types);
            check = emit!(const {::core::assert!(::core::matches!(#path::__SPLICE_RETURN,#expected),"annotate the kernel helper result with its Lane<T> or Mask<T> type")};);
            emit!(#path::__splice_call #fish)
        };
        let all_known = named && values.iter().all(|v| v.stage.is_host());
        let original_args = values.iter().map(|v| v.code.clone()).collect::<Vec<_>>();
        let mut args = Vec::new();
        let mut trait_stages = Vec::new();
        for (i, (expr, value)) in c.args.iter().zip(values).enumerate() {
            let formal = recursive.then(|| self.formals[i]);
            let stage = value.stage;
            let argument = match (formal, stage) {
                (Some(Binding::Reg(t, _)), _) => self.reg(value, t, expr)?,
                (Some(Binding::Buffer(write)), Stage::Buffer(actual))
                    if write.elem == actual.elem && (!write.write || actual.write) =>
                {
                    value.code
                }
                (Some(Binding::Buffer(_)), _) => {
                    return err(expr, "expected a runtime slice of the required mutability");
                }
                (Some(Binding::Host(_)), Stage::Word(_)) => {
                    emit!(#support::Host(#support::KnownArg::require(#value)?))
                }
                (_, Stage::Word(_)) => value.code,
                (_, stage) if stage.is_host() => emit!(#support::Host(#value)),
                (Some(Binding::Host(_)), _) => emit!({
                    return ::core::result::Result::Err(#root::BuildError::staging(
                        "runtime value passed to a comptime parameter",
                    ));
                }),
                (
                    None,
                    Stage::Reg(
                        Scalar::Int
                        | Scalar::Bool
                        | Scalar::Inferred
                        | Scalar::Vector(_)
                        | Scalar::Mask(_),
                    )
                    | Stage::Buffer(_),
                ) => value.code,
                _ => {
                    return err(
                        expr,
                        "expected a comptime value, a runtime i32, or a runtime slice",
                    );
                }
            };
            if trait_owner.is_some() {
                trait_stages.push(stage);
            }
            args.push(argument);
        }
        if all_known {
            let params = (0..args.len())
                .map(|i| quote::format_ident!("__splice_param_{i}"))
                .collect::<Vec<_>>();
            let bound = params
                .iter()
                .map(|name| emit!(#support::Host(#name)))
                .collect::<Vec<_>>();
            let mut item = p.path.clone();
            let generics = std::mem::take(&mut item.segments.last_mut().unwrap().arguments);
            let kernel = emit!({
                let (#(#params,)*)=(#(#original_args,)*);
                let mut runtime=||{::core::result::Result::Ok::<_,#root::BuildError>(#support::Word::Runtime(#target(#(#bound,)* __splice_builder)?))};
                #item::__splice_fold!(
                    match #root::build_boundary(|| #item::__SpliceDefinition::__splice_eval #generics ((#(::core::option::Option::Some(#params),)*))) {
                        ::core::result::Result::Ok(value)=>#support::Word::known(value),
                        ::core::result::Result::Err(_)=>runtime()?,
                    },
                    runtime()?
                )
            });
            let native = self.import_call(p, &args);
            let code = self.dispatch_call(p, kernel, emit!(#support::Word::Runtime(#native)));
            return Ok(Value::new(Stage::Word(output), code));
        }
        let staged_call =
            |args: &[TokenStream]| emit!({#check #target(#(#args,)* __splice_builder)?});
        let code = if let Some(owner) = trait_owner {
            let method = &p.path.segments.last().unwrap().ident;
            let names: Vec<_> = (0..args.len())
                .map(|index| {
                    quote::format_ident!(
                        "__splice_call_argument_{index}",
                        span = proc_macro2::Span::mixed_site()
                    )
                })
                .collect();
            let bound: Vec<_> = names.iter().map(|name| emit!(#name)).collect();
            let opaque_args: Vec<_> = bound
                .iter()
                .zip(trait_stages)
                .map(|(argument, stage)| {
                    if matches!(stage, Stage::Reg(_)) {
                        // The inlining factory records known formals before
                        // assigning their registers to the method body.
                        emit!(#support::Word::input(#argument, __splice_inputs))
                    } else {
                        argument.clone()
                    }
                })
                .collect();
            let mut shape = crate::scalar::methods::rename_path(p, "stage_shape");
            let mut policy = crate::scalar::methods::rename_path(p, "region_needs_stages");
            for metadata in [&mut shape, &mut policy] {
                metadata.path.segments.last_mut().unwrap().arguments = syn::PathArguments::None;
            }
            let opaque = self.import_result(p, &opaque_args, cast, Some((&shape, &policy)));
            let ordinary = self.import_result(p, &bound, cast, None);
            let kernel = staged_call(&bound);
            let boolean = if cast.is_some() {
                emit!(#support::RunArg::word(#kernel, __splice_builder)?)
            } else {
                kernel.clone()
            };
            let dispatch = crate::declaration::trait_method_dispatch(
                root, &owner, method, kernel, boolean, opaque, ordinary,
            );
            emit!({ let (#(#names,)*) = (#(#args,)*); #dispatch })
        } else if method {
            let mut published = crate::scalar::methods::rename_path(p, "published");
            published.path.segments.last_mut().unwrap().arguments = syn::PathArguments::None;
            let native = self.import_call(p, &args);
            emit!({ #check if #published() { #target(#(#args,)* __splice_builder)? } else { #native } })
        } else if named {
            let native = self.import_call(p, &args);
            self.dispatch_call(p, staged_call(&args), native)
        } else {
            staged_call(&args)
        };
        Ok(Value::new(Stage::Reg(output), code))
    }
}

impl Lower<'_> {
    pub(super) fn dispatch_call(
        &self,
        path: &syn::ExprPath,
        kernel: TokenStream,
        rust: TokenStream,
    ) -> TokenStream {
        dispatch_free(self.root, path, kernel, rust)
    }
    pub(super) fn import_call(&self, path: &syn::ExprPath, args: &[TokenStream]) -> TokenStream {
        self.import_result(path, args, None, None)
    }

    fn import_result(
        &self,
        path: &syn::ExprPath,
        args: &[TokenStream],
        cast: Option<&syn::Type>,
        stage_checks: Option<(&syn::ExprPath, &syn::ExprPath)>,
    ) -> TokenStream {
        let root = self.root;
        let native = emit!(#root::frontend::native);
        let arity = args.len();
        let slots: Vec<_> = (0..arity)
            .map(|index| quote::format_ident!("__SpliceOperand{index}"))
            .collect();
        let (candidate, prepare) = if let Some(ty) = cast {
            (
                emit!(__splice_candidate(#path, |result| result as #ty)),
                emit! {
                    fn __splice_candidate<#(#slots,)* R, F: ::core::ops::Fn(R) -> i32>(
                        entry: fn(#(#slots),*) -> R, project: F,
                    ) -> #native::Candidate<fn(#(#slots),*) -> R, #native::MapResult<F>> {
                        #native::Candidate(entry, #native::MapResult(project))
                    }
                },
            )
        } else {
            (
                emit!(__splice_candidate(#path)),
                emit! {
                    fn __splice_candidate<#(#slots,)* R>(entry: fn(#(#slots),*) -> R)
                        -> #native::Candidate<fn(#(#slots),*) -> R, #native::EncodeResult>
                    {
                        #native::Candidate(entry, #native::EncodeResult)
                    }
                },
            )
        };
        let (generics, _, where_clause) = self.generics.split_for_impl();
        let type_args = self
            .generics
            .params
            .iter()
            .filter_map(|p| match p {
                syn::GenericParam::Type(t) => Some(&t.ident),
                syn::GenericParam::Const(c) => Some(&c.ident),
                _ => None,
            })
            .collect::<Vec<_>>();
        let fish = (!type_args.is_empty()).then(|| emit!(::<#(#type_args),*>));
        // Nested Rust items cannot reference an outer impl's Self. Published
        // methods with that spelling still take the native staging branch.
        use syn::visit::Visit;
        let mut uses_self = crate::declaration::UsesSelf::default();
        uses_self.visit_expr_path(path);
        uses_self.visit_generics(self.generics);
        if uses_self.0 {
            return emit!(::core::result::Result::Err::<#root::hir::Reg,#root::BuildError>(#root::BuildError::staging("unpublished Self method requires its owning Rust context"))?);
        }
        let indices = 0..arity;
        let word = emit!(#root::frontend::scalar::ImportWord);
        let words = arity as u32;
        let adapter = emit! {
            trait __SplicePacketAdapter {
                fn supported(self) -> bool;
                fn invoke(self, args: #root::ContextArguments<'_>) -> i32;
            }
            impl<#(#slots: #word,)* R, P: #native::ProjectResult<R>> __SplicePacketAdapter for &#native::Candidate<fn(#(#slots),*) -> R, P> {
                fn supported(self) -> bool { true }
                #[allow(unused_variables)]
                fn invoke(self, args: #root::ContextArguments<'_>) -> i32 {
                    #native::ProjectResult::project(&self.1, (self.0)(#(<#slots as #word>::decode(args.word(#indices))),*))
                }
            }
            impl<F, P> __SplicePacketAdapter for &&#native::Candidate<F, P> {
                fn supported(self) -> bool { false }
                fn invoke(self, _: #root::ContextArguments<'_>) -> i32 { ::core::unreachable!("verified scalar packet adapter") }
            }
            fn adapter #generics (args: #root::ContextArguments<'_>) -> i32 #where_clause {
                (&#candidate).invoke(args)
            }
        };
        let call = emit! {
            __splice_builder.rust_call(
                #root::hir::RustTarget::Static(const {
                    &#root::StaticRustSymbol::new(::core::stringify!(#path), #words, 0, adapter #fish)
                }),
                &operands,
                &[],
            )?
        };
        let operands = if let Some((shape, needs_stages)) = stage_checks {
            let indices: Vec<_> = (0..arity).map(syn::Index::from).collect();
            emit! {
                let __splice_import_values = (#(Operand(#args),)*);
                let __splice_import_stages = #shape();
                #(if !__splice_import_stages.#indices.is_empty()
                    && !(&__splice_import_values.#indices).known()
                {
                    return ::core::result::Result::Err(#root::BuildError::staging("runtime value passed to a comptime parameter"));
                })*
                if #needs_stages() {
                    return ::core::result::Result::Err(#root::BuildError::staging("a published kernel call cannot be specialized inside this Rust region"));
                }
                let operands = [#((&__splice_import_values.#indices).materialize(__splice_builder)?),*];
            }
        } else {
            emit!(let operands = [#((&Operand(#args)).materialize(__splice_builder)?),*];)
        };
        emit!({
            #prepare
            #adapter
            use #root::frontend::native::{Materialize as _, Operand};
            if !(&#candidate).supported() {
                return ::core::result::Result::Err(#root::BuildError::staging("Rust import requires a non-scalar argument or result"));
            }
            #operands
            #call
        })
    }
}

pub(super) fn dispatch_free(
    root: &syn::Path,
    path: &syn::ExprPath,
    kernel: TokenStream,
    rust: TokenStream,
) -> TokenStream {
    select_free(
        root,
        path,
        &quote::format_ident!("__splice_dispatch"),
        kernel,
        rust,
    )
}

pub(crate) fn select_free(
    root: &syn::Path,
    path: &syn::ExprPath,
    selector: &proc_macro2::Ident,
    kernel: TokenStream,
    rust: TokenStream,
) -> TokenStream {
    let span = path.path.segments.last().unwrap().ident.span();
    let alias = crate::declaration::selector_alias("callee", span);
    let mut item = path.path.clone();
    item.segments.last_mut().unwrap().arguments = syn::PathArguments::None;
    emit!({
        mod __splice_default {
            pub mod #alias {
                pub use #root::__splice_choose_rust as #selector;
            }
        }
        #[allow(unused_imports)] use __splice_default::*;
        {
            #[allow(unused_imports)] use #item as #alias;
            #alias::#selector!(#kernel, #rust)
        }
    })
}
