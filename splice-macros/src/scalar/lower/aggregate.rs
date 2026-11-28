//! Keep scalar fields in their construction order and give them the stage
//! of the entire aggregate before exposing any projection to the lowerer.
use super::{Binding, Lower, Scalar, Stage, err};
use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{Expr, ExprField, Member, Pat, Path, Result};

pub(in crate::scalar) struct Layout {
    fields: Vec<(Member, Scalar)>,
    constructor: TokenStream,
}

/// The initializer of an aggregate local: a struct literal, or a call of a
/// tuple struct's constructor whose fields the body reads.
pub(super) struct Literal<'e> {
    fields: Vec<(Member, &'e Expr)>,
    /// The literal with every field replaced by `value`.
    shape: Box<dyn Fn(TokenStream) -> TokenStream + 'e>,
    /// Requires the callee of a constructor call to be a tuple struct.
    tuple: Option<&'e Expr>,
    at: &'e Expr,
}

impl<'e> Literal<'e> {
    pub(super) fn of(lower: &Lower<'_>, local: &syn::Local, init: &'e Expr) -> Option<Self> {
        match init {
            Expr::Struct(literal) => Some(Self {
                fields: literal
                    .fields
                    .iter()
                    .map(|field| (field.member.clone(), &field.expr))
                    .collect(),
                shape: Box::new(move |value| {
                    let mut witness = literal.clone();
                    for field in &mut witness.fields {
                        field.colon_token = Some(Default::default());
                        field.expr = Expr::Verbatim(value.clone());
                    }
                    witness.rest = None;
                    witness.into_token_stream()
                }),
                tuple: None,
                at: init,
            }),
            // A field read of the result means rustc typed the call as a
            // struct; the pattern check below rejects any callee other than
            // a tuple struct.
            Expr::Call(call)
                if matches!(&*call.func, Expr::Path(path) if path.qself.is_none())
                    && lower.body.projected_local(local) =>
            {
                Some(Self {
                    fields: call
                        .args
                        .iter()
                        .enumerate()
                        .map(|(index, arg)| (Member::Unnamed(index.into()), arg))
                        .collect(),
                    shape: Box::new(move |value| {
                        let (func, values) = (&call.func, call.args.iter().map(|_| &value));
                        emit!(#func(#(#values),*))
                    }),
                    tuple: Some(&call.func),
                    at: init,
                })
            }
            _ => None,
        }
    }

    pub(super) fn has_rest(&self) -> bool {
        matches!(self.at, Expr::Struct(literal) if literal.rest.is_some())
    }

    // A function item fixes the Rust type without constructing or dropping it.
    fn constructor(&self, root: &Path, pat: &Pat) -> TokenStream {
        let witness = (self.shape)(emit!(#root::frontend::scalar::ImportWord::decode(0)));
        let result = match pat {
            Pat::Type(typed) => {
                let ty = &typed.ty;
                emit!(-> #ty)
            }
            _ => TokenStream::new(),
        };
        let tuple = self.tuple.map(|func| {
            let fields = self.fields.iter().map(|_| emit!(_));
            emit!(let #func(#(#fields),*) = __splice_witness;)
        });
        emit!(|| #result {
            let __splice_witness = #witness;
            #tuple
            __splice_witness
        })
    }
}

impl Lower<'_> {
    /// A kernel's aggregate locals have no destructor: the IR cannot run
    /// Rust drop glue (a `const` assertion, so `cargo build` reports it).
    pub(in crate::scalar) fn dropless_aggregates(&self) -> TokenStream {
        let root = self.root;
        let layouts = self.aggregates.borrow();
        let owners = layouts.iter().map(|layout| &layout.constructor);
        emit!(#(#root::frontend::scalar::dropless_aggregate(#owners);)*)
    }

    pub(super) fn aggregate_local(&mut self, pat: &Pat, literal: &Literal) -> Result<TokenStream> {
        let root = self.root;
        let mut fields = Vec::with_capacity(literal.fields.len());
        let mut words = Vec::with_capacity(literal.fields.len());
        for (member, expr) in &literal.fields {
            let value = self.expr(expr)?;
            let kind = match value.stage {
                Stage::Word(kind) | Stage::Reg(kind) | Stage::Literal(kind) => kind,
                Stage::Host => Scalar::Inferred,
                _ => return err(expr, "expected a scalar field initializer"),
            };
            fields.push((member.clone(), kind));
            // A formal register can carry a known scalar from an enclosing call.
            words.push(if matches!(value.stage, Stage::Reg(_)) {
                emit!({
                    let reg = #value;
                    #root::frontend::scalar::Word::input(reg, __splice_inputs)
                })
            } else {
                self.word(value, expr)?
            });
        }
        let (id, constructor) = self.bind_aggregate(pat, literal, fields)?;
        Ok(emit!(
            let #id = #root::frontend::scalar::aggregate(
                [#(#words),*], #constructor, __splice_builder
            )?;
        ))
    }

    /// The aggregate local of the printer: the constants of its fields, all
    /// of them if the generator knows every field and none otherwise.
    pub(super) fn source_aggregate(&mut self, pat: &Pat, literal: &Literal) -> Result<TokenStream> {
        let root = self.root;
        let mut fields = Vec::with_capacity(literal.fields.len());
        let mut words = Vec::with_capacity(literal.fields.len());
        for (member, expr) in &literal.fields {
            fields.push((member.clone(), Scalar::Inferred));
            words.push(if self.body.known(expr) {
                let value = self.known_operand(expr, None)?;
                emit!(::core::option::Option::Some(#root::frontend::scalar::ScalarConstant::constant(#value)))
            } else {
                emit!(::core::option::Option::None)
            });
        }
        let (id, _) = self.bind_aggregate(pat, literal, fields)?;
        let count = words.len();
        Ok(emit!(
            let mut #id: [::core::option::Option<#root::hir::Constant>; #count] = [#(#words),*];
            if #id.iter().any(::core::option::Option::is_none) { #id.fill(::core::option::Option::None); }
        ))
    }

    /// Bind the aggregate local `pat` to the layout of `fields`: its name
    /// and the constructor that fixes its Rust type.
    fn bind_aggregate(
        &mut self,
        pat: &Pat,
        literal: &Literal,
        fields: Vec<(Member, Scalar)>,
    ) -> Result<(syn::Ident, TokenStream)> {
        let pattern = match pat {
            Pat::Type(p) => &*p.pat,
            p => p,
        };
        let name = super::super::named(pattern)?;
        if literal.has_rest() {
            return err(literal.at, "struct update requires an owned Rust value");
        }
        let constructor = literal.constructor(self.root, pat);
        let mut layouts = self.aggregates.borrow_mut();
        self.bindings
            .insert(self.body.binding(name), Binding::Aggregate(layouts.len()));
        layouts.push(Layout {
            fields,
            constructor: constructor.clone(),
        });
        self.debug_names.remove(&name.ident.to_string());
        Ok((name.ident.clone(), constructor))
    }

    /// The fields of aggregate local `layout`, in construction order.
    pub(super) fn aggregate_fields(&self, layout: usize) -> Vec<(Member, Scalar)> {
        self.aggregates.borrow()[layout].fields.clone()
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
