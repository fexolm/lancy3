//! Struct values a kernel reads from memory (`frontend::record::Record`):
//! struct parameters, elements of slices of structs, their fields and the
//! locals bound to them. Only the builder knows a struct's layout, so a
//! field the macro reads is a register of the type the builder gives it.
use super::{Binding, Elem, Lower, Scalar, Stage, Value, err};
use proc_macro2::TokenStream;
use syn::{BinOp, Expr, ExprIndex, Member, Result, UnOp};

/// Where a struct value comes from: a binding, or an element of a slice.
enum Root<'e> {
    Binding(&'e syn::ExprPath, bool),
    Element(&'e ExprIndex),
}

fn ungroup(e: &Expr) -> &Expr {
    match e {
        Expr::Paren(x) => ungroup(&x.expr),
        Expr::Group(x) => ungroup(&x.expr),
        e => e,
    }
}

fn member_name(member: &Member) -> String {
    match member {
        Member::Named(ident) => ident.to_string().trim_start_matches("r#").to_owned(),
        Member::Unnamed(index) => index.index.to_string(),
    }
}

impl Lower<'_> {
    /// The root of a struct place `e` and the fields it names from there.
    fn record_place<'e>(&self, e: &'e Expr) -> Result<Option<(Root<'e>, Vec<String>)>> {
        let mut path = Vec::new();
        let mut at = ungroup(e);
        loop {
            match at {
                Expr::Field(field) => {
                    path.push(member_name(&field.member));
                    at = ungroup(&field.base);
                }
                Expr::Unary(unary) if matches!(unary.op, UnOp::Deref(_)) => {
                    at = ungroup(&unary.expr);
                }
                _ => break,
            }
        }
        path.reverse();
        let root = match at {
            Expr::Path(p) => match self.binding(p) {
                Some(Binding::Record(mutable)) => Root::Binding(p, mutable),
                _ => return Ok(None),
            },
            Expr::Index(index) if !matches!(&*index.index, Expr::Range(_)) => {
                // Any other operand reports its own errors where it is lowered.
                match self.expr(&index.expr).map(|value| value.stage) {
                    Ok(Stage::Buffer(buffer)) if buffer.elem == Elem::Record => {
                        Root::Element(index)
                    }
                    _ => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        Ok(Some((root, path)))
    }

    /// The slice and index of an element of a slice of structs.
    fn record_element(&self, index: &ExprIndex) -> Result<(TokenStream, TokenStream, bool)> {
        let slice = self.expr(&index.expr)?;
        let Stage::Buffer(buffer) = slice.stage else {
            unreachable!("a slice of structs")
        };
        let at = self.expr(&index.index)?;
        let at = self.reg(at, Scalar::Index, &index.index)?;
        Ok((slice.code, at, buffer.write))
    }

    /// The `Record` a struct-valued expression evaluates to, if `e` is one:
    /// a struct binding, an element of a slice of structs, or a field of
    /// either (a scalar field is a `Record` of one register).
    pub(super) fn record(&self, e: &Expr) -> Result<Option<TokenStream>> {
        let Some((root, path)) = self.record_place(e)? else {
            return Ok(None);
        };
        let support = self.record_support();
        Ok(Some(match root {
            Root::Binding(binding, _) => emit!(#binding.field(&[#(#path),*])?),
            Root::Element(index) => {
                let (slice, at, _) = self.record_element(index)?;
                emit!({
                    let (__splice_slice, __splice_index) = (#slice, #at);
                    #support::load(__splice_builder, __splice_slice, __splice_index, &[#(#path),*])?
                })
            }
        }))
    }

    /// Whether `e` is a whole struct value (not one of its fields), which
    /// the macro knows to be a struct.
    pub(super) fn whole_record(&self, e: &Expr) -> Result<bool> {
        Ok(self
            .record_place(e)?
            .is_some_and(|(_, path)| path.is_empty()))
    }

    fn record_support(&self) -> TokenStream {
        let root = self.root;
        emit!(#root::frontend::record::Record)
    }

    /// A struct-valued expression read as a scalar, where the macro needs
    /// a register: rustc checked that the field is one.
    pub(super) fn record_scalar(&self, e: &Expr) -> Result<Option<Value>> {
        Ok(self.record(e)?.map(|record| {
            Value::new(
                Stage::Reg(Scalar::Inferred),
                emit!({ let __splice_record = #record; __splice_record.scalar()? }),
            )
        }))
    }

    /// `e` as a value of the type `ty` (a `DataType` expression) names: a
    /// struct value, a struct literal of struct values and scalars, a
    /// struct literal local, or a scalar.
    fn record_value(&self, e: &Expr, ty: TokenStream) -> Result<TokenStream> {
        let support = self.record_support();
        if let Some(record) = self.record(e)? {
            return Ok(record);
        }
        let root = self.root;
        match ungroup(e) {
            Expr::Struct(literal) => {
                if literal.rest.is_some() {
                    return err(literal, "struct update requires an owned Rust value");
                }
                let fields = literal
                    .fields
                    .iter()
                    .map(|field| {
                        let name = member_name(&field.member);
                        let value = self.record_value(
                            &field.expr,
                            emit!(#support::type_of(__splice_type, &[#name])?),
                        )?;
                        Ok(emit!((#name, #value)))
                    })
                    .collect::<Result<Vec<_>>>()?;
                return Ok(emit!({
                    let __splice_type: #root::hir::DataType = #ty;
                    #support::literal(__splice_type, ::std::vec![#(#fields),*])?
                }));
            }
            // A tuple struct's constructor, `Id(7)` or `Pair(a, b)`: types are
            // named in upper camel case, functions in snake case. A kernel
            // function cannot return a struct.
            Expr::Call(call)
                if matches!(&*call.func, Expr::Path(path) if path.path.segments.last()
                    .is_some_and(|s| s.ident.to_string().starts_with(|c: char| c.is_ascii_uppercase()))) =>
            {
                let fields = call
                    .args
                    .iter()
                    .enumerate()
                    .map(|(index, arg)| {
                        let name = index.to_string();
                        let value = self.record_value(
                            arg,
                            emit!(#support::type_of(__splice_type, &[#name])?),
                        )?;
                        Ok(emit!((#name, #value)))
                    })
                    .collect::<Result<Vec<_>>>()?;
                return Ok(emit!({
                    let __splice_type: #root::hir::DataType = #ty;
                    #support::tuple(__splice_type, ::std::vec![#(#fields),*])?
                }));
            }
            Expr::Path(path) if matches!(self.binding(path), Some(Binding::Aggregate(_))) => {
                let Some(Binding::Aggregate(layout)) = self.binding(path) else {
                    unreachable!()
                };
                let fields = self.aggregate_fields(layout);
                let fields = fields
                    .iter()
                    .enumerate()
                    .map(|(index, (member, _))| {
                        let name = member_name(member);
                        // A known field (an unsuffixed literal the macro read
                        // as `i32` or `f64`) takes the type of the struct's.
                        Ok(emit!((#name, {
                            let __splice_word = #path[#index];
                            match #root::frontend::scalar::RunArg::known(&__splice_word) {
                                ::core::option::Option::Some(__splice_known) => #support::constant(
                                    __splice_builder,
                                    #support::type_of(__splice_type, &[#name])?,
                                    __splice_known,
                                )?,
                                ::core::option::Option::None => {
                                    let __splice_reg = #root::frontend::scalar::RunArg::reg(__splice_word, __splice_builder)?;
                                    #support::of(__splice_builder, __splice_reg)?
                                }
                            }
                        })))
                    })
                    .collect::<Result<Vec<_>>>()?;
                return Ok(emit!({
                    let __splice_type: #root::hir::DataType = #ty;
                    #support::literal(__splice_type, ::std::vec![#(#fields),*])?
                }));
            }
            _ => {}
        }
        let value = self.expr(e)?;
        Ok(match value.stage {
            Stage::Literal(_) => emit!({
                let __splice_type: #root::hir::DataType = #ty;
                #support::constant(
                    __splice_builder,
                    __splice_type,
                    <#root::hir::Constant as ::core::convert::From<_>>::from(#value),
                )?
            }),
            Stage::Host => emit!({
                let __splice_type: #root::hir::DataType = #ty;
                #support::constant(
                    __splice_builder,
                    __splice_type,
                    #root::frontend::scalar::ScalarConstant::constant(#value),
                )?
            }),
            _ => {
                let reg = self.reg(value, Scalar::Inferred, e)?;
                emit!({ let __splice_reg = #reg; #support::of(__splice_builder, __splice_reg)? })
            }
        })
    }

    /// `let p = <struct value>;`: the binding holds registers of its own,
    /// which assignments to a mutable binding change.
    pub(super) fn record_local(
        &mut self,
        local: &syn::Local,
        init: &Expr,
    ) -> Result<Option<TokenStream>> {
        let Some(record) = self.record(init)? else {
            return Ok(None);
        };
        let pat = match &local.pat {
            syn::Pat::Type(t) => &*t.pat,
            p => p,
        };
        let name = super::super::named(pat)?;
        self.declare(name, Binding::Record(name.mutability.is_some()));
        self.debug_names.remove(&name.ident.to_string());
        let id = &name.ident;
        Ok(Some(emit!(
            let #id = { let __splice_record = #record; __splice_record.copy(__splice_builder)? };
        )))
    }

    /// An assignment (or compound assignment) to a struct place: a field of
    /// an element of a slice of structs, the element itself, or a mutable
    /// struct binding or its field.
    pub(super) fn record_assignment(
        &self,
        left: &Expr,
        right: &Expr,
        op: Option<(&str, &BinOp)>,
    ) -> Result<Option<Value>> {
        let Some((root, path)) = self.record_place(left)? else {
            return Ok(None);
        };
        let support = self.record_support();
        let (target, write) = match &root {
            Root::Binding(binding, mutable) => {
                if !mutable {
                    return err(left, "assignment requires a mutable runtime variable");
                }
                (emit!(#binding.path_type(&[#(#path),*])?), emit!())
            }
            Root::Element(index) => {
                let (slice, at, writable) = self.record_element(index)?;
                if !writable {
                    return err(
                        index,
                        "writing a kernel slice requires a &mut slice parameter",
                    );
                }
                (
                    emit!(#support::field_type(__splice_builder, __splice_slice, &[#(#path),*])?),
                    emit!(let (__splice_slice, __splice_index) = (#slice, #at);),
                )
            }
        };
        let value = self.record_value(right, emit!(__splice_target))?;
        let value = match op {
            None => value,
            Some((op, _)) => {
                let current = match &root {
                    Root::Binding(binding, _) => emit!(#binding.field(&[#(#path),*])?.scalar()?),
                    Root::Element(_) => emit!(
                        #support::load(__splice_builder, __splice_slice, __splice_index, &[#(#path),*])?.scalar()?
                    ),
                };
                let combined =
                    self.operator_bin(op, emit!(__splice_current), emit!(__splice_operand));
                emit!({
                    let __splice_operand = #value.scalar()?;
                    let __splice_current = #current;
                    let __splice_combined = #combined;
                    #support::of(__splice_builder, __splice_combined)?
                })
            }
        };
        let finish = match &root {
            Root::Binding(binding, _) => {
                emit!(#binding.assign(__splice_builder, &[#(#path),*], &__splice_value)?;)
            }
            Root::Element(_) => emit!(
                __splice_value.store(__splice_builder, __splice_slice, __splice_index, &[#(#path),*])?;
            ),
        };
        // Rust evaluates the right side first; a slice element's bounds
        // check is part of its load or store.
        let code = emit!({
            #write
            let __splice_target = #target;
            let __splice_value = #value;
            #finish
        });
        Ok(Some(Value::unit(code)))
    }
}
