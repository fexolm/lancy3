//! Keep scalar fields in their construction order and give them the stage
//! of the entire aggregate before exposing any projection to the lowerer.
use super::{Binding, Lower, Scalar, Stage, err};
use proc_macro2::TokenStream;
use syn::{Expr, ExprField, ExprStruct, ItemFn, Member, Pat, Path, Result};

pub(in crate::scalar) struct Layout {
    fields: Vec<(Member, Scalar)>,
    constructor: TokenStream,
}

// A function item fixes the Rust type without constructing or dropping it.
fn constructor(root: &Path, pat: &Pat, literal: &ExprStruct) -> TokenStream {
    let mut witness = literal.clone();
    for field in &mut witness.fields {
        field.colon_token = Some(Default::default());
        field.expr = syn::parse_quote!(#root::frontend::scalar::ImportWord::decode(0));
    }
    let result = match pat {
        Pat::Type(typed) => {
            let ty = &typed.ty;
            emit!(-> #ty)
        }
        _ => TokenStream::new(),
    };
    emit!(|| #result { #witness })
}

impl Lower<'_> {
    /// Rust owns the lexical scope when a local's destructor is observable.
    /// A dropless aggregate continues through the ordinary field lowering.
    pub(in crate::scalar) fn owned_scope(&self, function: &ItemFn) -> TokenStream {
        let root = self.root;
        let layouts = self.aggregates.borrow();
        if layouts.is_empty() {
            return TokenStream::new();
        }
        let owners = layouts.iter().map(|layout| &layout.constructor);
        let Some(checks) = crate::import::scope_checks(root, function) else {
            return TokenStream::new();
        };
        let Some(call) = self.owned_call(&function.sig) else {
            return TokenStream::new();
        };
        emit!({
            if false #(|| #root::frontend::scalar::has_destructor(#owners))* {
                #checks
                return Ok(#call);
            }
        })
    }

    pub(super) fn aggregate_local(
        &mut self,
        pat: &Pat,
        literal: &ExprStruct,
    ) -> Result<TokenStream> {
        let pattern = match pat {
            Pat::Type(p) => &*p.pat,
            p => p,
        };
        let name = super::super::named(pattern)?;
        let id = &name.ident;
        if literal.rest.is_some() {
            return err(literal, "struct update requires an owned Rust value");
        }
        let root = self.root;
        let mut fields = Vec::with_capacity(literal.fields.len());
        let mut words = Vec::with_capacity(literal.fields.len());
        for field in &literal.fields {
            let value = self.expr(&field.expr)?;
            let kind = match value.stage {
                Stage::Word(kind) | Stage::Reg(kind) | Stage::Literal(kind) => kind,
                Stage::Host => Scalar::Inferred,
                _ => return err(field, "expected a scalar field initializer"),
            };
            fields.push((field.member.clone(), kind));
            // A formal register can carry a known scalar from an enclosing call.
            let word = if self.source_fault.is_some() {
                if value.stage.is_host() {
                    emit!(Some(#root::frontend::scalar::ScalarConstant::bits(#value)))
                } else {
                    emit!(None)
                }
            } else if matches!(value.stage, Stage::Reg(_)) {
                emit!({
                    let reg = #value;
                    #root::frontend::scalar::Word::input(reg, __splice_inputs)
                })
            } else {
                self.word(value, &field.expr)?
            };
            words.push(word);
        }
        let constructor = constructor(root, pat, literal);
        let mut layouts = self.aggregates.borrow_mut();
        self.bindings
            .insert(id.to_string(), Binding::Aggregate(layouts.len()));
        layouts.push(Layout {
            fields,
            constructor: constructor.clone(),
        });
        self.debug_names.remove(&id.to_string());
        if self.source_fault.is_some() {
            let count = words.len();
            return Ok(emit!(
                let mut #id: [Option<i32>; #count] = [#(#words),*];
                if #id.iter().any(Option::is_none) { #id.fill(None); }
            ));
        }
        Ok(emit!(
            let #id = #root::frontend::scalar::aggregate(
                [#(#words),*], #constructor, __splice_builder
            )?;
        ))
    }

    pub(super) fn aggregate_field(
        &self,
        field: &ExprField,
    ) -> Option<(TokenStream, Scalar, TokenStream)> {
        let Expr::Path(path) = &*field.base else {
            return None;
        };
        let Binding::Aggregate(layout) = self.binding(path)? else {
            return None;
        };
        let layouts = self.aggregates.borrow();
        let (index, (_, kind)) = layouts[layout]
            .fields
            .iter()
            .enumerate()
            .find(|(_, (member, _))| member == &field.member)?;
        Some((
            emit!(#path[#index]),
            *kind,
            layouts[layout].constructor.clone(),
        ))
    }
}
