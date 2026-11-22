//! Parse the item under `kernel` and validate staging attributes before Rust
//! checks the original signature and body.
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{Attribute, Block, Error, FnArg, Item, Meta, Path, Result, Signature, Visibility};

mod container;
use container::{Container, method_container};

pub(crate) fn expand(root: Path, tokens: TokenStream) -> Result<TokenStream> {
    let item = match syn::parse2::<Item>(tokens.clone()) {
        Ok(item) => item,
        Err(item_error) => {
            let Ok(method) = syn::parse2::<syn::TraitItemFn>(tokens) else {
                return Err(item_error);
            };
            return isolated_trait_item(&root, method);
        }
    };
    match item {
        Item::Verbatim(tokens) => {
            let Ok(method) = syn::parse2::<syn::TraitItemFn>(tokens.clone()) else {
                return Err(Error::new_spanned(
                    tokens,
                    "expected a function, method, impl or trait",
                ));
            };
            isolated_trait_item(&root, method)
        }
        Item::Fn(mut function) => {
            let container = if function.sig.receiver().is_some()
                && !matches!(function.vis, Visibility::Inherited)
            {
                // A visible receiver method also proves inherent context
                // when its tokens originated in another macro.
                Some(Container::Inherent)
            } else {
                method_container(&function.sig)
            };
            if matches!(container, Some(Container::Trait)) {
                return isolated_trait_item(&root, syn::parse2(function.into_token_stream())?);
            }
            if matches!(
                container,
                Some(Container::Inherent | Container::TraitImpl(_) | Container::UnprovenMethod)
            ) || method_context(&function.attrs, &function.sig)
            {
                let companions = if matches!(container, Some(Container::Inherent)) {
                    crate::scalar::methods::publish(
                        &root,
                        &function.sig,
                        &function.block,
                        &function.vis,
                        false,
                    )
                } else {
                    Vec::new()
                };
                let gates = configuration(&function.attrs)?;
                let registration = match &container {
                    Some(Container::TraitImpl(implementation)) => {
                        Some(crate::scalar::methods::register_individual_trait(
                            &root,
                            &implementation.owner,
                            &implementation.trait_path,
                            &implementation.generics,
                            &function.sig,
                            &function.block,
                        ))
                    }
                    _ => None,
                };
                prepare_method(
                    &mut function.attrs,
                    &mut function.sig,
                    Some(&mut function.block),
                    false,
                )?;
                if let Some(registration) = registration {
                    function.block.stmts.insert(0, registration);
                }
                let companions = companions.iter().map(|item| quote!(#(#[#gates])* #item));
                Ok(quote!(#function #(#companions)*))
            } else {
                signature(&function.sig)?;
                item_markers(&mut function.attrs, false, false)?;
                crate::import::expand_function(root, function)
            }
        }
        Item::Impl(mut implementation) => {
            item_markers(&mut implementation.attrs, false, false)?;
            let mut specialized = Vec::new();
            let mut registrations = Vec::new();
            let trait_path = implementation
                .trait_
                .as_ref()
                .map(|(_, path, _)| path.clone());
            for member in &mut implementation.items {
                if let syn::ImplItem::Fn(method) = member {
                    let gates = configuration(&method.attrs)?;
                    if let Some(path) = &trait_path {
                        let shape = crate::scalar::methods::stage_shape(&method.sig);
                        specialized.push(syn::parse2(quote!(#(#[#gates])* #shape))?);
                        registrations.push(crate::scalar::methods::register_individual_trait(
                            &root,
                            &implementation.self_ty,
                            path,
                            &implementation.generics,
                            &method.sig,
                            &method.block,
                        ));
                    }
                    for hidden in crate::scalar::methods::publish(
                        &root,
                        &method.sig,
                        &method.block,
                        &method.vis,
                        trait_path.is_some(),
                    ) {
                        specialized.push(syn::parse2(quote!(#(#[#gates])* #hidden))?);
                    }
                    prepare_method(
                        &mut method.attrs,
                        &mut method.sig,
                        Some(&mut method.block),
                        true,
                    )?;
                }
            }
            if let Some(path) = trait_path {
                let gates = configuration(&implementation.attrs)?;
                let mut published = implementation.clone();
                published.items.extend(specialized);
                let methods = implementation
                    .items
                    .iter_mut()
                    .filter_map(|item| match item {
                        syn::ImplItem::Fn(method) => Some(method),
                        _ => None,
                    });
                for (method, registration) in methods.zip(registrations) {
                    method.block.stmts.insert(0, registration);
                }
                let dispatch = trait_dispatch(
                    &root,
                    &path,
                    quote!({ #published }),
                    quote!({ #implementation }),
                );
                return Ok(quote! {
                    #(#[#gates])*
                    #[allow(non_local_definitions)]
                    const _: () = #dispatch;
                });
            }
            implementation.items.extend(specialized);
            Ok(implementation.into_token_stream())
        }
        Item::Trait(mut declaration) => {
            item_markers(&mut declaration.attrs, false, false)?;
            let mut specialized = Vec::new();
            for member in &mut declaration.items {
                if let syn::TraitItem::Fn(method) = member {
                    let gates = configuration(&method.attrs)?;
                    let hidden = if crate::scalar::methods::individual_trait_supported(&method.sig)
                    {
                        crate::scalar::methods::word_trait(&root, &method.sig)
                    } else {
                        let shape = crate::scalar::methods::stage_shape(&method.sig);
                        specialized.push(syn::parse2(quote!(#(#[#gates])* #shape))?);
                        crate::scalar::methods::trait_defaults(&root, &method.sig)
                    };
                    for hidden in hidden {
                        specialized.push(syn::parse2(quote!(#(#[#gates])* #hidden))?);
                    }
                    prepare_method(
                        &mut method.attrs,
                        &mut method.sig,
                        method.default.as_mut(),
                        true,
                    )?;
                }
            }
            declaration.items.extend(specialized);
            let gates = configuration(&declaration.attrs)?;
            let name = &declaration.ident;
            let vis = &declaration.vis;
            Ok(quote! {
                #declaration

                #(#[#gates])*
                #[doc(hidden)]
                #[allow(unused_imports)]
                #vis use #root::__splice_choose_kernel as #name;
            })
        }
        other => Err(Error::new_spanned(
            other,
            "expected a function, method, impl or trait",
        )),
    }
}

fn isolated_trait_item(root: &Path, mut method: syn::TraitItemFn) -> Result<TokenStream> {
    let gates = configuration(&method.attrs)?;
    let mut companions = crate::scalar::methods::word_trait(root, &method.sig);
    if companions.is_empty() {
        companions.push(syn::parse2(crate::scalar::methods::stage_shape(
            &method.sig,
        ))?);
    }
    prepare_method(
        &mut method.attrs,
        &mut method.sig,
        method.default.as_mut(),
        false,
    )?;
    let companions = companions.iter().map(|item| quote!(#(#[#gates])* #item));
    Ok(quote!(#method #(#companions)*))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_trait_requirement_preserves_the_native_receiver_signature() {
        let tokens = super::expand(
            syn::parse_quote!(::splice),
            quote!(
                fn limit(#[comptime] &self, value: i32) -> i32;
            ),
        )
        .unwrap();
        let declaration: syn::ItemTrait = syn::parse_quote!(trait Test { #tokens });
        let syn::TraitItem::Fn(method) = &declaration.items[0] else {
            panic!("the original trait method is the first item");
        };
        let receiver = method.sig.receiver().unwrap();
        assert!(receiver.reference.is_some());
        assert!(receiver.attrs.is_empty());
        assert!(method.default.is_none());
    }

    #[test]
    fn isolated_trait_requirement_rejects_an_explicit_object_parameter() {
        let error = super::expand(
            syn::parse_quote!(::splice),
            quote!(
                fn limit(&self, value: &dyn AsRef<i32>) -> i32;
            ),
        )
        .unwrap_err();
        assert!(error.to_string().contains("explicit trait object"));
    }
}

fn configuration(attrs: &[Attribute]) -> Result<Vec<TokenStream>> {
    attrs
        .iter()
        .filter_map(|attr| crate::scalar::configuration(&attr.meta).transpose())
        .collect()
}

/// An ordinary trait occupies only the type namespace. A published trait
/// additionally imports a selector in the macro namespace.
pub(crate) fn trait_dispatch(
    root: &Path,
    path: &Path,
    published: TokenStream,
    ordinary: TokenStream,
) -> TokenStream {
    let mut path = path.clone();
    for part in &mut path.segments {
        part.arguments = syn::PathArguments::None;
    }
    let alias = selector_alias("trait", path.segments.last().unwrap().ident.span());
    quote::quote_spanned!(proc_macro2::Span::mixed_site()=> {
        mod __splice_default_trait {
            pub use #root::__splice_choose_rust as #alias;
        }
        use __splice_default_trait::*;
        {
            #[allow(unused_imports)]
            use #path as #alias;
            #alias!(#published, #ordinary)
        }
    })
}

// A nested call selects its own fallback even inside another selector's imports.
pub(crate) fn selector_alias(kind: &str, span: proc_macro2::Span) -> proc_macro2::Ident {
    let (line, column) = if proc_macro::is_available() {
        (span.unwrap().line(), span.unwrap().column())
    } else {
        (0, 0)
    };
    quote::format_ident!("__splice_{kind}_{line}_{column}")
}

#[derive(Default)]
pub(crate) struct UsesSelf(pub(crate) bool);
impl<'ast> Visit<'ast> for UsesSelf {
    fn visit_path(&mut self, path: &'ast Path) {
        self.0 |= path
            .segments
            .first()
            .is_some_and(|segment| segment.ident == "Self");
        syn::visit::visit_path(self, path);
    }
}

fn method_context(attrs: &[Attribute], sig: &Signature) -> bool {
    let mut uses = UsesSelf(false);
    uses.visit_signature(sig);
    sig.receiver().is_some() || uses.0 || attrs.iter().any(|attr| marker(attr, "intrinsic"))
}

fn prepare_method(
    attrs: &mut Vec<Attribute>,
    sig: &mut Signature,
    body: Option<&mut Block>,
    inside_annotated_item: bool,
) -> Result<()> {
    signature(sig)?;
    item_markers(attrs, inside_annotated_item, body.is_some())?;
    for arg in &mut sig.inputs {
        let attrs = match arg {
            FnArg::Typed(arg) => &mut arg.attrs,
            FnArg::Receiver(arg) => &mut arg.attrs,
        };
        let mut comptime = false;
        for attr in attrs.iter() {
            if !marker(attr, "comptime") {
                return Err(Error::new_spanned(
                    attr,
                    "expected #[comptime] on a method parameter",
                ));
            }
            if comptime || !matches!(attr.meta, Meta::Path(_)) {
                return Err(Error::new_spanned(
                    attr,
                    "expected one #[comptime] marker without arguments",
                ));
            }
            comptime = true;
        }
        attrs.clear();
    }
    if let Some(body) = body {
        native_body(body)?;
    }
    Ok(())
}

fn marker(attr: &Attribute, name: &str) -> bool {
    attr.path()
        .segments
        .last()
        .is_some_and(|segment| segment.ident == name)
}

fn item_markers(
    attrs: &mut Vec<Attribute>,
    nested_kernel: bool,
    intrinsic_body: bool,
) -> Result<()> {
    let mut kernel = false;
    let mut intrinsic = false;
    for attr in attrs.iter() {
        if marker(attr, "kernel") {
            if !nested_kernel || kernel || !matches!(attr.meta, Meta::Path(_)) {
                return Err(Error::new_spanned(
                    attr,
                    "expected one #[kernel] marker without arguments",
                ));
            }
            kernel = true;
        } else if marker(attr, "comptime") {
            return Err(Error::new_spanned(
                attr,
                "#[comptime] belongs on a kernel parameter",
            ));
        } else if marker(attr, "intrinsic") {
            if !intrinsic_body || intrinsic {
                return Err(Error::new_spanned(
                    attr,
                    "one #[intrinsic(\"name\")] is allowed on a kernel method with a body",
                ));
            }
            let name: syn::LitStr = attr.parse_args()?;
            if name.value().is_empty() {
                return Err(Error::new_spanned(
                    name,
                    "expected a nonempty intrinsic name",
                ));
            }
            intrinsic = true;
        }
    }
    attrs.retain(|attr| !marker(attr, "kernel") && !marker(attr, "intrinsic"));
    Ok(())
}

fn signature(sig: &Signature) -> Result<()> {
    if sig.asyncness.is_some()
        || sig.unsafety.is_some()
        || sig.abi.is_some()
        || sig.variadic.is_some()
    {
        return Err(Error::new_spanned(
            sig,
            "expected a safe synchronous kernel with Rust ABI and fixed arity",
        ));
    }
    struct DynType(Option<Error>);
    impl<'ast> Visit<'ast> for DynType {
        fn visit_type_trait_object(&mut self, ty: &'ast syn::TypeTraitObject) {
            self.0 = Some(Error::new_spanned(
                ty,
                "an explicit trait object is unsupported in a kernel signature",
            ));
        }
    }
    let mut find = DynType(None);
    find.visit_signature(sig);
    find.0.map_or(Ok(()), Err)
}

pub(crate) fn native_body(block: &mut Block) -> Result<()> {
    struct Native(Option<Error>);
    impl Native {
        fn parse(&mut self, mac: &syn::Macro) -> Option<syn::Expr> {
            if !mac.path.is_ident("comptime") {
                return None;
            }
            match mac.parse_body() {
                Ok(expr) => Some(expr),
                Err(error) => {
                    self.0 = Some(error);
                    None
                }
            }
        }
    }
    impl VisitMut for Native {
        fn visit_stmt_mut(&mut self, stmt: &mut syn::Stmt) {
            if let syn::Stmt::Macro(mac) = stmt
                && let Some(expr) = self.parse(&mac.mac)
            {
                *stmt = syn::Stmt::Expr(expr, mac.semi_token);
            }
            syn::visit_mut::visit_stmt_mut(self, stmt);
        }
        fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
            if let syn::Expr::Macro(mac) = expr
                && let Some(inner) = self.parse(&mac.mac)
            {
                let attrs = &mac.attrs;
                *expr = syn::parse_quote!(#(#attrs)* (#inner));
            }
            syn::visit_mut::visit_expr_mut(self, expr);
        }
        fn visit_item_mut(&mut self, _: &mut Item) {}
    }
    let mut native = Native(None);
    native.visit_block_mut(block);
    native.0.map_or(Ok(()), Err)
}

pub(crate) fn nested_visibility(visibility: &Visibility) -> Visibility {
    match visibility {
        Visibility::Inherited => syn::parse_quote!(pub(super)),
        Visibility::Restricted(restriction) if restriction.path.is_ident("self") => {
            syn::parse_quote!(pub(super))
        }
        Visibility::Restricted(restriction) if restriction.path.segments[0].ident == "super" => {
            let path = &restriction.path;
            syn::parse_quote!(pub(in super::#path))
        }
        _ => visibility.clone(),
    }
}

struct RebasePaths;

impl VisitMut for RebasePaths {
    fn visit_path_mut(&mut self, path: &mut Path) {
        syn::visit_mut::visit_path_mut(self, path);
        if path.segments.len() < 2 {
            return;
        }
        if path.segments[0].ident == "self" {
            path.segments[0].ident = syn::parse_quote!(super);
        } else if path.segments[0].ident == "super" {
            *path = syn::parse_quote!(super::#path);
        }
    }
}

pub(crate) fn rebase_expression(expr: &mut syn::Expr) {
    RebasePaths.visit_expr_mut(expr);
}

pub(crate) fn rebase_signature(sig: &mut Signature) {
    RebasePaths.visit_signature_mut(sig);
}
