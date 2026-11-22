//! Generate build-time checks for calls inside Rust regions.
use crate::scalar::{methods, select_free};
use proc_macro2::TokenStream;
use syn::{Expr, ExprPath, Path, Type};

pub(in crate::import) enum Check {
    Function(ExprPath, Vec<bool>),
    Method {
        path: ExprPath,
        trait_method: bool,
        known: Vec<bool>,
    },
    Constructor(ExprPath, Type),
    Branch {
        condition: Expr,
        yes: Vec<Check>,
        no: Vec<Check>,
    },
}

impl Check {
    pub(in crate::import) fn emit(
        self,
        root: &Path,
        resolve: fn(ExprPath) -> ExprPath,
    ) -> TokenStream {
        match self {
            Self::Branch { condition, yes, no } => {
                let yes = yes.into_iter().map(|check| check.emit(root, resolve));
                let no = no.into_iter().map(|check| check.emit(root, resolve));
                emit!(if #condition { #(#yes)* } else { #(#no)* })
            }
            Self::Function(path, known) => {
                let path = resolve(path);
                let mut module = path.path.clone();
                module.segments.last_mut().unwrap().arguments = syn::PathArguments::None;
                let selector = quote::format_ident!("__splice_region_dispatch");
                let staged_call = emit!({
                    let known: &[bool] = &[#(#known,)*];
                    #module::__SPLICE_REGION_NEEDS_STAGES ||
                        #module::__SPLICE_REGION_STAGED.iter().zip(known).any(|(&staged, &known)| staged && !known)
                });
                let guard = select_free(root, &path, &selector, staged_call, emit!(false));
                let message =
                    "a published kernel call cannot be specialized inside this Rust region";
                emit!(if #guard { return Err(#message.into()); })
            }
            Self::Method {
                path,
                trait_method,
                known,
            } => {
                let path = resolve(path);
                let entry = |trait_kind, inherent_kind| {
                    let mut entry = methods::rename_path(
                        &path,
                        if trait_method {
                            trait_kind
                        } else {
                            inherent_kind
                        },
                    );
                    entry.path.segments.last_mut().unwrap().arguments = syn::PathArguments::None;
                    entry
                };
                let shape = entry("stage_shape", "inherent_region_stage_shape");
                let needs_stages = entry("region_needs_stages", "inherent_region_needs_stages");
                let masks = (0..known.len())
                    .map(|index| quote::format_ident!("__splice_stage_mask_{index}"))
                    .collect::<Vec<_>>();
                let checks = emit! {
                    let (#(#masks,)*) = #shape();
                    if false #(|| (!#masks.is_empty() && !#known))* {
                        return Err("runtime value passed to a comptime parameter".into());
                    }
                    if #needs_stages() {
                        return Err("a published kernel call cannot be specialized inside this Rust region".into());
                    }
                };
                if trait_method {
                    let owner = methods::trait_path(&path).expect("a qualified trait method");
                    crate::declaration::trait_dispatch(root, &owner, emit!({ #checks }), emit!(()))
                } else {
                    let shape_name = &shape.path.segments.last().unwrap().ident;
                    let policy_name = &needs_stages.path.segments.last().unwrap().ident;
                    let arity = known.iter().map(|_| emit!([(); 0]));
                    let masks = known.iter().map(|_| emit!([]));
                    emit!({
                        trait __SpliceOrdinaryMethod {
                            fn #shape_name() -> (#(#arity,)*) { (#(#masks,)*) }
                            fn #policy_name() -> bool { false }
                        }
                        impl<T: ?Sized> __SpliceOrdinaryMethod for T {}
                        #checks
                    })
                }
            }
            Self::Constructor(path, ty) => {
                let path = resolve(path);
                // Constructors also occupy the type namespace. Verify the
                // variant through a pattern if its name shadows the prelude.
                emit!(let _ = |value: #ty| ::core::matches!(value, #path(..));)
            }
        }
    }
}
