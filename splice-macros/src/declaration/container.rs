//! Recover the immediate source container when attribute input is ambiguous.
use quote::ToTokens;
use syn::visit::Visit;
use syn::{Attribute, Path, Signature};

pub(super) enum Container {
    Inherent,
    TraitImpl(Implementation),
    Trait,
    Free,
    UnprovenMethod,
}

pub(super) struct Implementation {
    pub(super) owner: Box<syn::Type>,
    pub(super) trait_path: Path,
    pub(super) generics: syn::Generics,
}

/// An attribute on a method receives the same tokens inside either kind of
/// impl. The source span can prove the immediate container for ordinary source
/// items. Macro-generated and unavailable source cannot prove it.
pub(super) fn method_container(signature: &Signature) -> Option<Container> {
    if !proc_macro::is_available() {
        return None;
    }
    let name = &signature.ident;
    let span = name.span().unwrap();
    let path = span.local_file()?;
    if !std::fs::metadata(&path).ok()?.is_file() {
        return None;
    }
    let token = span.source_text()?;
    if *name != token {
        return None;
    }
    let mut source = std::fs::read_to_string(path).ok()?;
    let offset = source_offset(&source, &token, span.line(), span.column())?;
    let sentinel = fresh_identifier(&source);
    source.replace_range(offset..offset + token.len(), &sentinel);
    let file = syn::parse_file(&source).ok()?;
    let mut expected = signature.clone();
    expected.ident = proc_macro2::Ident::new(&sentinel, proc_macro2::Span::call_site());
    let mut find = FindContainer {
        name: &sentinel,
        signature: expected.to_token_stream().to_string(),
        matches: Vec::new(),
        trusted_module: true,
    };
    find.visit_file(&file);
    (find.matches.len() == 1)
        .then(|| find.matches.pop())
        .flatten()
}

fn fresh_identifier(source: &str) -> String {
    let prefix = "__splice_item_";
    let extra = source
        .match_indices(prefix)
        .map(|(at, _)| {
            source.as_bytes()[at + prefix.len()..]
                .iter()
                .take_while(|&&byte| byte == b'_')
                .count()
                + 1
        })
        .max()
        .unwrap_or_default();
    format!("{prefix}{}", "_".repeat(extra))
}

fn source_offset(source: &str, token: &str, line: usize, column: usize) -> Option<usize> {
    let column = column.checked_sub(1)?;
    let mut lines = source.split_inclusive('\n');
    let line_start = lines
        .by_ref()
        .take(line.checked_sub(1)?)
        .map(str::len)
        .sum::<usize>();
    let text = lines.next()?;
    let character_offset = text.char_indices().nth(column).map(|(offset, _)| offset);
    let matches = |offset: usize| {
        let Some(prefix) = text.get(..offset) else {
            return false;
        };
        let Some(rest) = text
            .get(offset..)
            .and_then(|suffix| suffix.strip_prefix(token))
        else {
            return false;
        };
        prefix
            .chars()
            .next_back()
            .is_none_or(|ch| !ch.is_alphanumeric() && ch != '_' && ch != '#')
            && rest
                .chars()
                .next()
                .is_none_or(|ch| !ch.is_alphanumeric() && ch != '_')
    };
    match (
        matches(column),
        character_offset.filter(|offset| *offset != column && matches(*offset)),
    ) {
        (true, None) => Some(line_start + column),
        (false, Some(offset)) => Some(line_start + offset),
        _ => None,
    }
}

struct FindContainer<'a> {
    name: &'a str,
    signature: String,
    matches: Vec<Container>,
    trusted_module: bool,
}

impl FindContainer<'_> {
    fn is_target(&self, signature: &Signature) -> bool {
        signature.ident == self.name && signature.to_token_stream().to_string() == self.signature
    }
}

impl<'source> Visit<'source> for FindContainer<'_> {
    fn visit_item_mod(&mut self, module: &'source syn::ItemMod) {
        let enclosing = self.trusted_module;
        self.trusted_module &= inert_attributes(&module.attrs);
        syn::visit::visit_item_mod(self, module);
        self.trusted_module = enclosing;
    }

    fn visit_item_fn(&mut self, item: &'source syn::ItemFn) {
        if self.is_target(&item.sig) {
            self.matches.push(Container::Free);
        }
        syn::visit::visit_item_fn(self, item);
    }

    fn visit_item_impl(&mut self, item: &'source syn::ItemImpl) {
        for member in &item.items {
            if let syn::ImplItem::Fn(method) = member
                && self.is_target(&method.sig)
            {
                let container = if !self.trusted_module || !inert_attributes(&item.attrs) {
                    Container::UnprovenMethod
                } else if let Some((_, trait_path, _)) = &item.trait_ {
                    Container::TraitImpl(Implementation {
                        owner: item.self_ty.clone(),
                        trait_path: trait_path.clone(),
                        generics: item.generics.clone(),
                    })
                } else {
                    Container::Inherent
                };
                self.matches.push(container);
            }
        }
        syn::visit::visit_item_impl(self, item);
    }

    fn visit_trait_item_fn(&mut self, item: &'source syn::TraitItemFn) {
        if self.is_target(&item.sig) {
            self.matches.push(Container::Trait);
        }
        syn::visit::visit_trait_item_fn(self, item);
    }
}

fn inert_attributes(attrs: &[Attribute]) -> bool {
    attrs.iter().all(|attr| {
        [
            "allow",
            "deny",
            "deprecated",
            "doc",
            "expect",
            "forbid",
            "inline",
            "warn",
            "cfg",
        ]
        .iter()
        .any(|name| attr.path().is_ident(name))
    })
}
