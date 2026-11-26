//! Parse the item under `kernel` and validate staging attributes before Rust
//! checks the original signature and body.
use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{Attribute, Block, Error, FnArg, Item, Meta, Path, Result, Signature, Visibility};

use crate::import::Lowering;
mod selector;
pub(crate) use selector::{trait_method_dispatch, trait_method_eval_dispatch};

/// `print` (`#[kernel(print)]`) applies to a free function lowered to IR.
pub(crate) fn expand(
    root: Path,
    lowering: Lowering,
    print: bool,
    tokens: TokenStream,
) -> Result<TokenStream> {
    let method_error = "put #[kernel] on the enclosing impl or trait to publish its methods";
    let item = match syn::parse2::<Item>(tokens.clone()) {
        Ok(Item::Verbatim(_)) | Err(_)
            if syn::parse2::<syn::TraitItemFn>(tokens.clone()).is_ok() =>
        {
            return Err(Error::new_spanned(tokens, method_error));
        }
        item => item?,
    };
    match item {
        Item::Fn(mut function) => {
            if method_context(&function.attrs, &function.sig) {
                return Err(Error::new_spanned(&function.sig, method_error));
            }
            hoist_lint_attributes(&mut function.attrs, &mut function.sig);
            signature(&function.sig)?;
            item_markers(&mut function.attrs, false)?;
            crate::import::expand_function(root, function, lowering, print)
        }
        Item::Impl(_) | Item::Trait(_) if print => Err(Error::new_spanned(
            tokens,
            "#[kernel(print)] publishes `print` for a free function",
        )),
        Item::Impl(_) | Item::Trait(_) if lowering == Lowering::Rust => Err(Error::new_spanned(
            tokens,
            "put #[kernel(rust)] on individual methods of a #[kernel] impl or trait",
        )),
        Item::Impl(mut implementation) => {
            item_markers(&mut implementation.attrs, false)?;
            let mut specialized = Vec::new();
            let mut errors = Errors::default();
            let trait_path = implementation.trait_.as_ref().map(|(_, path, _)| path);
            for member in &mut implementation.items {
                if let syn::ImplItem::Fn(method) = member {
                    let lowering = method_lowering(&mut method.attrs)?;
                    let gates = configuration(&method.attrs)?;
                    let shape = crate::scalar::methods::stage_shape(
                        &root,
                        &method.sig,
                        Some(&method.block),
                        trait_path.is_none().then_some(&method.vis),
                    );
                    for item in shape {
                        specialized.push(syn::ImplItem::Verbatim(quote!(#(#[#gates])* #item)));
                    }
                    let intrinsic = intrinsic_name(&method.attrs)?;
                    let generated = match lowering {
                        Lowering::Ir => crate::scalar::methods::publish(
                            &root,
                            &method.sig,
                            &method.block,
                            &method.vis,
                            trait_path.is_some(),
                            intrinsic.as_ref(),
                        )
                        .unwrap_or_else(|error| {
                            errors.push(error);
                            Vec::new()
                        }),
                        Lowering::Rust => Vec::new(),
                    };
                    if let Some(path) = trait_path
                        && !generated.is_empty()
                    {
                        // An impl may spell an associated type as its concrete
                        // result. Only override entries declared by the trait.
                        let alias =
                            selector_alias("trait", path.segments.last().unwrap().ident.span());
                        let name = &method.sig.ident;
                        let kind =
                            crate::scalar::methods::TraitMethodKind::for_result(&method.sig.output);
                        let provided = if kind == crate::scalar::methods::TraitMethodKind::EvalOnly
                        {
                            quote!(eval)
                        } else {
                            quote!(stage)
                        };
                        specialized.push(syn::ImplItem::Verbatim(quote! {
                            #(#[#gates])* #alias! { @items #name, __splice_method_case, (items #provided #(#generated)*) }
                        }));
                    } else {
                        for hidden in generated {
                            specialized
                                .push(syn::ImplItem::Verbatim(quote!(#(#[#gates])* #hidden)));
                        }
                    }
                    prepare_method(&mut method.attrs, &mut method.sig, Some(&mut method.block))?;
                }
            }
            errors.finish()?;
            if let Some(path) = trait_path {
                let gates = configuration(&implementation.attrs)?;
                implementation.items.extend(specialized);
                let dispatch =
                    select_trait(&root, path, quote!(@implementation { #implementation }));
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
            item_markers(&mut declaration.attrs, false)?;
            let name = &declaration.ident;
            let (_, arguments, _) = declaration.generics.split_for_impl();
            let trait_owner = quote!(#name #arguments);
            let mut specialized = Vec::new();
            let mut selectors = Vec::new();
            let mut errors = Errors::default();
            for member in &mut declaration.items {
                if let syn::TraitItem::Fn(method) = member {
                    let lowering = method_lowering(&mut method.attrs)?;
                    let gates = configuration(&method.attrs)?;
                    let (kind, defaults) = match lowering {
                        Lowering::Ir => {
                            crate::scalar::methods::trait_defaults(&root, &method.sig, &trait_owner)
                                .unwrap_or_else(|error| {
                                    errors.push(error);
                                    (crate::scalar::methods::TraitMethodKind::Rust, Vec::new())
                                })
                        }
                        Lowering::Rust => {
                            (crate::scalar::methods::TraitMethodKind::Rust, Vec::new())
                        }
                    };
                    selectors.push((method.sig.ident.clone(), kind, gates.clone()));
                    for hidden in
                        crate::scalar::methods::stage_shape(&root, &method.sig, None, None)
                            .into_iter()
                            .chain(defaults)
                    {
                        specialized.push(syn::TraitItem::Verbatim(quote!(#(#[#gates])* #hidden)));
                    }
                    prepare_method(&mut method.attrs, &mut method.sig, method.default.as_mut())?;
                }
            }
            errors.finish()?;
            declaration.items.extend(specialized);
            let gates = configuration(&declaration.attrs)?;
            let vis = &declaration.vis;
            let selector = selector::definition(name, vis, &gates, selectors);
            Ok(quote! {
                #declaration
                #selector
            })
        }
        other => Err(Error::new_spanned(
            other,
            "expected a free function, impl or trait",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebased_use_trees_keep_their_targets() {
        let mut body: syn::Expr = syn::parse_quote!({
            use self::inner::{a, b};
            use super::offset;
            use crate::root;
            offset() + self::f(1) + super::g()
        });
        rebase_expression(&mut body, &syn::Generics::default());
        let text = quote::ToTokens::to_token_stream(&body).to_string();
        assert!(text.contains("use super :: super :: offset ;"), "{text}");
        assert!(text.contains("use super :: inner :: { a , b } ;"), "{text}");
        assert!(text.contains("use crate :: root ;"), "{text}");
        assert!(text.contains("super :: f (1)"), "{text}");
        assert!(text.contains("super :: super :: g ()"), "{text}");
    }

    /// MAC-11: generated names do not depend on memory addresses or on how
    /// many items the process expanded before.
    #[test]
    fn expansion_is_deterministic() {
        let items = [
            quote!(
                fn total(#[comptime] k: i32, xs: &[i32], flag: bool) -> i32 {
                    let mut sum = 0;
                    for &x in xs {
                        if flag {
                            sum += x * k;
                        }
                    }
                    sum + comptime!(k + 1)
                }
            ),
            quote!(
                pub trait Rule {
                    fn apply(value: i32) -> i32;
                }
            ),
        ];
        for item in items {
            let expand = || {
                expand(
                    syn::parse_quote!(::splice),
                    Lowering::Ir,
                    false,
                    item.clone(),
                )
                .unwrap()
                .to_string()
            };
            assert_eq!(expand(), expand());
        }
    }

    /// MAC-10: the expansion of a small kernel pair stays within a token
    /// budget, so growth of the generated code is a reviewed change. The
    /// probe is the review's `size.rs`: a helper and a loop calling it.
    #[test]
    fn expansion_size_stays_within_its_budget() {
        fn tokens(stream: TokenStream) -> usize {
            stream
                .into_iter()
                .map(|tree| match tree {
                    proc_macro2::TokenTree::Group(group) => 1 + tokens(group.stream()),
                    _ => 1,
                })
                .sum()
        }
        let probe = [
            quote!(
                fn helper(x: i32) -> i32 {
                    x * 2 + 1
                }
            ),
            quote!(
                fn body(#[comptime] k: i32, xs: &[i32]) -> i32 {
                    let mut s = 0;
                    for &x in xs {
                        s += helper(x) + k * 3;
                    }
                    s
                }
            ),
        ];
        let total: usize = probe
            .into_iter()
            .map(|item| {
                tokens(expand(syn::parse_quote!(::splice), Lowering::Ir, false, item).unwrap())
            })
            .sum();
        // 7 213 tokens at the commit that introduced the budget.
        const BUDGET: usize = 7_400;
        assert!(total <= BUDGET, "{total} tokens, budget {BUDGET}");
    }

    /// MAC-09: a marker is recognized by its whole path.
    #[test]
    fn foreign_attributes_are_not_splice_markers() {
        let item = quote!(impl Plan { fn read(#[x::comptime] value: i32) -> i32 { value } });
        let error = expand(syn::parse_quote!(::splice), Lowering::Ir, false, item).unwrap_err();
        assert!(
            error.to_string().contains("expected #[comptime]"),
            "{error}"
        );
        let item =
            quote!(impl Plan { #[splice::kernel(rust)] fn read(value: i32) -> i32 { value } });
        assert!(expand(syn::parse_quote!(::splice), Lowering::Ir, false, item).is_ok());
    }

    #[test]
    fn methods_must_be_published_by_their_block() {
        for item in [
            quote!(
                fn value(&self) -> i32 {
                    1
                }
            ),
            quote!(
                fn value() -> i32;
            ),
            quote!(
                trait Rule {
                    #[kernel]
                    fn value() -> i32;
                }
            ),
            quote!(impl Plan { #[kernel] fn value() -> i32 { 1 } }),
        ] {
            let error = expand(syn::parse_quote!(::splice), Lowering::Ir, false, item).unwrap_err();
            assert!(
                error.to_string().contains("enclosing impl or trait"),
                "{error}"
            );
        }
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
    select_trait(root, path, quote!(#published, #ordinary))
}

fn select_trait(root: &Path, path: &Path, input: TokenStream) -> TokenStream {
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
            #[allow(unused_imports)] use #root::__splice_method_case;
            #alias!(#input)
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

/// Report every unpublishable method of a block, not only the first.
#[derive(Default)]
struct Errors(Option<Error>);

impl Errors {
    fn push(&mut self, error: Error) {
        match &mut self.0 {
            Some(first) => first.combine(error),
            None => self.0 = Some(error),
        }
    }

    fn finish(self) -> Result<()> {
        self.0.map_or(Ok(()), Err)
    }
}

/// A method of a `#[kernel]` block is lowered unless it carries
/// `#[kernel(rust)]`; the marker is consumed here.
fn method_lowering(attrs: &mut Vec<Attribute>) -> Result<Lowering> {
    let Some(index) = attrs.iter().position(|attr| marker(attr, "kernel")) else {
        return Ok(Lowering::Ir);
    };
    let attr = attrs.remove(index);
    match &attr.meta {
        Meta::List(list) if list.parse_args::<syn::Ident>().is_ok_and(|id| id == "rust") => {
            Ok(Lowering::Rust)
        }
        _ => Err(Error::new_spanned(
            attr,
            "put #[kernel] only on the enclosing impl or trait; a method accepts #[kernel(rust)]",
        )),
    }
}

fn prepare_method(
    attrs: &mut Vec<Attribute>,
    sig: &mut Signature,
    body: Option<&mut Block>,
) -> Result<()> {
    hoist_lint_attributes(attrs, sig);
    signature(sig)?;
    item_markers(attrs, body.is_some())?;
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

/// Move lint attributes (`#[allow(..)]` and friends) from parameters to the
/// function. They only affect diagnostics, and the parameter keeps its Splice
/// meaning from the remaining markers.
fn hoist_lint_attributes(attrs: &mut Vec<Attribute>, sig: &mut Signature) {
    const LINTS: [&str; 5] = ["allow", "expect", "warn", "deny", "forbid"];
    for arg in &mut sig.inputs {
        let params = match arg {
            FnArg::Typed(arg) => &mut arg.attrs,
            FnArg::Receiver(arg) => &mut arg.attrs,
        };
        let (lints, markers) = std::mem::take(params)
            .into_iter()
            .partition(|attr| LINTS.iter().any(|lint| attr.path().is_ident(lint)));
        *params = markers;
        attrs.extend::<Vec<_>>(lints);
    }
}

/// Whether `attr` is the Splice marker `name`. `comptime`, `columns` and
/// `intrinsic` are inert names only a `#[kernel]` item reads; `kernel` is
/// also the attribute macro, which a method can name by its path in Splice.
/// `#[cuda::kernel]` or `#[x::comptime]` is someone else's attribute.
fn marker(attr: &Attribute, name: &str) -> bool {
    let path = attr.path();
    path.is_ident(name)
        || (name == "kernel"
            && path.segments.len() == 2
            && path.segments[0].ident == "splice"
            && path.segments[1].ident == name)
}

/// The name of the method's `#[intrinsic("name")]`, validated by
/// [`item_markers`].
fn intrinsic_name(attrs: &[Attribute]) -> Result<Option<syn::LitStr>> {
    attrs
        .iter()
        .find(|attr| marker(attr, "intrinsic"))
        .map(Attribute::parse_args)
        .transpose()
}

fn item_markers(attrs: &mut Vec<Attribute>, intrinsic_body: bool) -> Result<()> {
    let mut intrinsic = false;
    for attr in attrs.iter() {
        if marker(attr, "kernel") {
            return Err(Error::new_spanned(attr, "duplicate #[kernel] attribute"));
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
    attrs.retain(|attr| !marker(attr, "intrinsic"));
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

/// Rewrites paths of a user signature or expression moved into the
/// generated `mod <fn>`: `self::` and `super::` gain a level, and a type
/// the module itself declares (`BuildArgs`, `CallArgs`, `Signature`,
/// `OpaqueKernel`) is the user's item outside it, unless the name is a
/// generic parameter or an item declared inside the expression.
struct RebasePaths {
    local: std::collections::HashSet<String>,
}

/// Types the generated module declares; they shadow its `use super::*`.
const MODULE_TYPES: [&str; 4] = ["BuildArgs", "CallArgs", "Signature", "OpaqueKernel"];

impl RebasePaths {
    fn new(generics: &syn::Generics) -> Self {
        let local = generics
            .params
            .iter()
            .filter_map(|param| match param {
                syn::GenericParam::Type(ty) => Some(ty.ident.to_string()),
                syn::GenericParam::Const(c) => Some(c.ident.to_string()),
                syn::GenericParam::Lifetime(_) => None,
            })
            .collect();
        Self { local }
    }

    /// Also keep the names of items declared inside `expr`.
    fn declared_in(mut self, expr: &syn::Expr) -> Self {
        struct Items<'a>(&'a mut std::collections::HashSet<String>);
        impl<'ast> Visit<'ast> for Items<'_> {
            fn visit_item(&mut self, item: &'ast Item) {
                let name = match item {
                    Item::Struct(i) => Some(&i.ident),
                    Item::Enum(i) => Some(&i.ident),
                    Item::Union(i) => Some(&i.ident),
                    Item::Type(i) => Some(&i.ident),
                    Item::Trait(i) => Some(&i.ident),
                    _ => None,
                };
                self.0.extend(name.map(ToString::to_string));
                if let Item::Use(item) = item {
                    self.visit_use_tree(&item.tree);
                }
            }
            fn visit_use_name(&mut self, name: &'ast syn::UseName) {
                self.0.insert(name.ident.to_string());
            }
            fn visit_use_rename(&mut self, rename: &'ast syn::UseRename) {
                self.0.insert(rename.rename.to_string());
            }
        }
        Items(&mut self.local).visit_expr(expr);
        self
    }
}

impl VisitMut for RebasePaths {
    fn visit_path_mut(&mut self, path: &mut Path) {
        syn::visit_mut::visit_path_mut(self, path);
        let first = &path.segments[0].ident;
        if path.leading_colon.is_none()
            && MODULE_TYPES.iter().any(|name| first == name)
            && !self.local.contains(&first.to_string())
        {
            *path = syn::parse_quote!(super::#path);
            return;
        }
        if path.segments.len() < 2 {
            return;
        }
        if path.segments[0].ident == "self" {
            path.segments[0].ident =
                proc_macro2::Ident::new("super", path.segments[0].ident.span());
        } else if path.segments[0].ident == "super" {
            *path = syn::parse_quote!(super::#path);
        }
    }

    fn visit_item_use_mut(&mut self, item: &mut syn::ItemUse) {
        if item.leading_colon.is_none()
            && let syn::UseTree::Path(root) = &mut item.tree
        {
            if root.ident == "self" {
                root.ident = proc_macro2::Ident::new("super", root.ident.span());
            } else if root.ident == "super" {
                let placeholder = syn::UseTree::Glob(syn::UseGlob {
                    star_token: Default::default(),
                });
                let tree = std::mem::replace(&mut item.tree, placeholder);
                item.tree = syn::UseTree::Path(syn::UsePath {
                    ident: proc_macro2::Ident::new("super", proc_macro2::Span::call_site()),
                    colon2_token: Default::default(),
                    tree: Box::new(tree),
                });
            }
        }
    }
}

/// Rebase `expr` of a function with `generics` into its generated module.
pub(crate) fn rebase_expression(expr: &mut syn::Expr, generics: &syn::Generics) {
    RebasePaths::new(generics)
        .declared_in(expr)
        .visit_expr_mut(expr);
}

pub(crate) fn rebase_signature(sig: &mut Signature) {
    RebasePaths::new(&sig.generics).visit_signature_mut(sig);
}
