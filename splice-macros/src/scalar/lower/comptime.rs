//! Generate the mandatory build-time evaluation of an expression. Each
//! operand is checked where Rust would evaluate it, including short circuits.
use super::{Binding, Lower, Scalar};
use proc_macro2::{Ident, TokenStream};
use std::collections::HashSet;
use syn::{BinOp, Expr, FnArg, Pat, Result, Stmt};

impl Lower<'_> {
    pub(in crate::scalar) fn evaluator(
        &self,
        sig: &syn::Signature,
        block: &syn::Block,
        vis: &syn::Visibility,
        method: Option<&Ident>,
    ) -> Result<TokenStream> {
        let name = match method {
            Some(name) => emit!(#name),
            None => emit!(__splice_eval),
        };
        let root = self.root;
        let generics = &sig.generics;
        let where_clause = &generics.where_clause;
        let names = crate::scalar::abi::parameter_names(sig).collect::<Vec<_>>();
        let types = sig
            .inputs
            .iter()
            .map(|a| match a {
                FnArg::Typed(a) => a.ty.as_ref(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        let checks=sig.inputs.iter().zip(&names).filter_map(|(a,name)|{
            let FnArg::Typed(a)=a else{return None};
            a.attrs.iter().any(|a|a.path().is_ident("comptime")).then(||emit!(
                if #name.is_none(){return ::core::result::Result::Err(#root::BuildError::staging("unknown argument passed to a comptime parameter"));}
            ))
        });
        let ty = match &sig.output {
            syn::ReturnType::Type(_, ty) => emit!(#ty),
            _ => emit!(()),
        };
        let inputs = names.iter().map(ToString::to_string).collect();
        let wide = method.is_some()
            && crate::scalar::methods::TraitMethodKind::for_result(&sig.output)
                == crate::scalar::methods::TraitMethodKind::EvalOnly;
        let body = if block.stmts.iter().all(evaluable_statement)
            && (!wide || pure_wide_body(sig, block, &inputs))
            && (method.is_some()
                || types.iter().all(|t| {
                    matches!(t, syn::Type::Reference(reference) if reference.mutability.is_none())
                        || matches!(
                            Scalar::from_type(t, self.types),
                            Some(Scalar::Int | Scalar::Bool | Scalar::Index)
                        )
                })) {
            let mut evaluator = self.clone();
            let label = syn::Lifetime::new("'__splice_evaluated", proc_macro2::Span::mixed_site());
            evaluator.evaluator_return = Some(label.clone());
            let body = evaluator.required_block(
                block,
                Some(self.result_type),
                &HashSet::new(),
                &inputs,
            )?;
            emit!(#[allow(unused_labels)] #label: { #body })
        } else {
            emit!(
                return ::core::result::Result::Err(#root::BuildError::staging(
                    "body cannot be evaluated at specialization"
                ))
            )
        };
        Ok(emit! {
            #[doc(hidden)]
            #[allow(unused_variables,unreachable_code,clippy::type_complexity)]
            #vis fn #name #generics ((#(#names,)*): (#(::core::option::Option<#types>,)*))
                -> #root::Result<#ty> #where_clause {
                let _depth=#root::frontend::scalar::enter_expansion()?;
                #(#checks)*
                ::core::result::Result::Ok(#body)
            }
        })
    }

    pub(super) fn required(&self, expr: &Expr, hint: Option<Scalar>) -> Result<TokenStream> {
        self.required_expr(expr, hint, &HashSet::new(), &HashSet::new())
    }

    fn required_expr(
        &self,
        expr: &Expr,
        hint: Option<Scalar>,
        locals: &HashSet<String>,
        inputs: &HashSet<String>,
    ) -> Result<TokenStream> {
        let support = &self.root;
        let visit = |e: &Expr| self.required_expr(e, None, locals, inputs);
        let primitive = |value: TokenStream| {
            emit!({
                use #support::frontend::evaluation::{CheckPrimitive as _, Operand};
                let value = #value;
                (&Operand::of(&value)).check()?;
                value
            })
        };
        let rejected = || {
            // An unevaluated operand supplies the Rust result type when all
            // names have their original type in the specialization factory.
            if inputs.is_empty() && !super::simd::uses_lane(expr) && self.host_only(expr) {
                emit!(if false { #expr } else { return ::core::result::Result::Err(#support::BuildError::staging("opaque operation in comptime!")) })
            } else {
                let ty = match hint {
                    Some(Scalar::Bool) => emit!(bool),
                    Some(Scalar::Index | Scalar::Length) => emit!(usize),
                    _ => emit!(i32),
                };
                emit!(::core::result::Result::Err::<#ty,#support::BuildError>(#support::BuildError::staging("value unavailable at specialization"))?)
            }
        };
        Ok(match expr {
            Expr::Lit(_) => emit!(#expr),
            Expr::Path(path)
                if path.path.get_ident().is_some_and(|id| {
                    inputs.contains(&id.to_string()) && !locals.contains(&id.to_string())
                }) =>
            {
                emit!((#path).ok_or_else(|| #support::BuildError::staging("runtime argument read in comptime!"))?)
            }
            Expr::Path(path)
                if path
                    .path
                    .get_ident()
                    .is_some_and(|id| locals.contains(&id.to_string())) =>
            {
                emit!(#path)
            }
            Expr::Path(path) => match self.binding(path) {
                Some(Binding::Reg(ty, _)) => {
                    let ty = match ty {
                        Scalar::Bool => emit!(bool),
                        Scalar::Index | Scalar::Length => emit!(usize),
                        _ => emit!(i32),
                    };
                    if self.source_fault.is_some() {
                        emit!(::core::result::Result::Err::<#ty,#support::BuildError>(#support::BuildError::staging("runtime argument read in comptime!"))?)
                    } else {
                        let convert = match ty.to_string().as_str() {
                            "bool" => emit!(word.bits() != 0),
                            "usize" => emit!(word.bits() as usize),
                            _ => emit!(word.bits() as i32),
                        };
                        emit!({
                            let word=__splice_inputs.iter().find(|(reg,_)| *reg==#path)
                                .and_then(|(_,known)| *known)
                                .ok_or_else(|| #support::BuildError::staging("runtime argument read in comptime!"))?;
                            #convert
                        })
                    }
                }
                Some(Binding::Word(ty)) => {
                    let ty = if ty == Scalar::Bool {
                        emit!(bool)
                    } else {
                        emit!(i32)
                    };
                    if self.source_fault.is_some() {
                        emit!(::core::result::Result::Err::<#ty,#support::BuildError>(#support::BuildError::staging("runtime value in comptime!"))?)
                    } else {
                        emit!(<#ty as #support::frontend::scalar::ImportWord>::from_constant(
                            #support::frontend::scalar::RunArg::known(&#path).ok_or_else(|| #support::BuildError::staging("runtime value in comptime!"))?
                        ))
                    }
                }
                Some(
                    Binding::Columns
                    | Binding::Buffer(_)
                    | Binding::Context
                    | Binding::Aggregate(_),
                ) => rejected(),
                Some(Binding::Host(_)) => emit!(#path),
                // A const path constructs a fresh value, including in a
                // match scrutinee. Check its type before evaluating it.
                None => emit!(#support::frontend::evaluation::Dropless::check()?.bind(#path)),
            },
            Expr::Paren(p) => visit(&p.expr)?,
            Expr::Group(p) => visit(&p.expr)?,
            Expr::Tuple(tuple) => {
                let elements = tuple.elems.iter().map(visit).collect::<Result<Vec<_>>>()?;
                emit!((#(#elements,)*))
            }
            Expr::Array(array) => {
                let elements = array.elems.iter().map(visit).collect::<Result<Vec<_>>>()?;
                emit!([#(#elements),*])
            }
            Expr::Struct(literal) => {
                let mut result = literal.clone();
                for field in &mut result.fields {
                    let value = visit(&field.expr)?;
                    field.colon_token = Some(Default::default());
                    field.expr = syn::parse2(value)?;
                }
                if let Some(rest) = &mut result.rest {
                    **rest = syn::parse2(visit(rest)?)?;
                }
                emit!(#support::frontend::evaluation::Dropless::check()?.bind(#result))
            }
            Expr::Index(i) => {
                let read = self.known_index(visit(&i.expr)?, visit(&i.index)?);
                emit!((#read).ok_or(#support::BuildError::Comptime(#support::Trap::Bounds))?)
            }
            Expr::Cast(c) => {
                let value = primitive(visit(&c.expr)?);
                let ty = &c.ty;
                emit!((#value) as #ty)
            }
            Expr::Unary(u) if matches!(u.op, syn::UnOp::Deref(_)) => {
                self.known_deref(visit(&u.expr)?)
            }
            Expr::Unary(u) => {
                let value = primitive(visit(&u.expr)?);
                let op = &u.op;
                let check = matches!(op, syn::UnOp::Neg(_)).then(|| {
                    let checks = super::host::fault_checks(false);
                    emit! {
                        use #support::frontend::scalar::HostNegation as _;
                        if #checks && let ::core::option::Option::Some(fault) = (&(&value,)).negation_fault() {
                            return ::core::result::Result::Err(#support::BuildError::Comptime(fault.trap()));
                        }
                    }
                });
                emit!({ let value = #value; #check #op value })
            }
            Expr::Binary(b) => {
                let op = &b.op;
                let rhs = primitive(visit(&b.right)?);
                if let Some(kind) = super::super::compound_op(op) {
                    if matches!(&*b.left,Expr::Path(p) if p.path.get_ident().is_some_and(|n|locals.contains(&n.to_string())))
                    {
                        let lhs = &b.left;
                        let check = comptime_fault(support, kind, emit!(&(&#lhs, &right)));
                        emit!({
                            use #support::frontend::evaluation::{CheckPrimitive as _, Operand};
                            (&Operand::of(&#lhs)).check()?;
                            let right = #rhs;
                            #check
                            #lhs #op right;
                        })
                    } else {
                        rejected()
                    }
                } else {
                    let lhs = primitive(visit(&b.left)?);
                    match op {
                        BinOp::And(_) | BinOp::Or(_) => emit!((#lhs) #op (#rhs)),
                        _ => {
                            let check = super::host::operator_kind(op).and_then(|kind| {
                                comptime_fault(support, kind, emit!(&(&left, &right)))
                            });
                            emit!({ let left=#lhs; let right=#rhs; #check left #op right })
                        }
                    }
                }
            }
            Expr::MethodCall(m) => {
                let name = m.method.to_string();
                let receiver = visit(&m.receiver)?;
                let method = &m.method;
                if matches!(name.as_str(), "len" | "is_empty") && m.args.is_empty() {
                    let empty = (method == "is_empty").then(|| emit!(== 0));
                    emit!({ use #support::frontend::evaluation::{ReadLength as _, Operand};
                        let receiver= &(#receiver); (&Operand::of(receiver)).length(receiver)? #empty
                    })
                } else if super::memory::primitive_method(&name) {
                    let receiver = primitive(receiver);
                    let args = m
                        .args
                        .iter()
                        .map(|a| visit(a).map(&primitive))
                        .collect::<Result<Vec<_>>>()?;
                    emit!((#receiver).#method(#(#args),*))
                } else {
                    rejected()
                }
            }
            Expr::Block(b) => self.required_block(&b.block, hint, locals, inputs)?,
            Expr::Field(field)
                if matches!(&*field.base, Expr::Path(path)
                if path.path.get_ident().is_some_and(|id| !locals.contains(&id.to_string()))
                && matches!(self.binding(path), Some(Binding::Aggregate(_)))) =>
            {
                let (word, _, constructor) = self
                    .aggregate_field(field)
                    .ok_or_else(|| syn::Error::new_spanned(field, "missing aggregate field"))?;
                let known = if self.source_fault.is_some() {
                    word
                } else {
                    emit!(#support::frontend::scalar::RunArg::known(&#word))
                };
                let member = &field.member;
                emit!(#support::frontend::scalar::known_field(#constructor, |v| &v.#member, #known)?)
            }
            Expr::Field(field)
                if matches!(&*field.base, Expr::Path(path)
                    if path.path.get_ident().is_some_and(|id|
                        inputs.contains(&id.to_string()) && !locals.contains(&id.to_string()))) =>
            {
                let base = &field.base;
                let place = emit!(*#base.as_ref().ok_or_else(|| #support::BuildError::staging("runtime argument read in comptime!"))?);
                self.known_field(place, &field.member)
            }
            Expr::Field(field) if self.host_only(expr) || !inputs.is_empty() => {
                self.known_field(visit(&field.base)?, &field.member)
            }
            Expr::If(i) => {
                let mut bindings = locals.clone();
                let cond = if let Expr::Let(binding) = &*i.cond {
                    let value = visit(&binding.expr)?;
                    let pattern = &binding.pat;
                    bind_locals(&mut bindings, pattern);
                    emit!(let #pattern = #value)
                } else {
                    self.required_expr(&i.cond, Some(Scalar::Bool), locals, inputs)?
                };
                let yes = self.required_block(&i.then_branch, hint, &bindings, inputs)?;
                let no = i
                    .else_branch
                    .as_ref()
                    .map(|(_, e)| self.required_expr(e, hint, locals, inputs))
                    .transpose()?
                    .map(|e| emit!(else {#e}));
                emit!(if #cond {#yes} #no)
            }
            Expr::Match(m) => {
                let value = visit(&m.expr)?;
                let mut arms = Vec::with_capacity(m.arms.len());
                for arm in &m.arms {
                    let mut bindings = locals.clone();
                    let pat = &arm.pat;
                    bind_locals(&mut bindings, pat);
                    let guard = arm
                        .guard
                        .as_ref()
                        .map(|(_, guard)| {
                            self.required_expr(guard, Some(Scalar::Bool), &bindings, inputs)
                                .map(|test| emit!(if #test))
                        })
                        .transpose()?;
                    let body = self.required_expr(&arm.body, hint, &bindings, inputs)?;
                    let attrs = &arm.attrs;
                    arms.push(emit!(#(#attrs)* #pat #guard => { #body },));
                }
                emit!(match #value { #(#arms)* })
            }
            Expr::While(w) => {
                let label = &w.label;
                let cond = self.required_expr(&w.cond, Some(Scalar::Bool), locals, inputs)?;
                let body = self.required_block(&w.body, None, locals, inputs)?;
                emit!(#label while #cond { #support::frontend::scalar::specialization_step()?; #body; })
            }
            Expr::Loop(l) => {
                let label = &l.label;
                let body = self.required_block(&l.body, None, locals, inputs)?;
                emit!(#label loop { #support::frontend::scalar::specialization_step()?; #body; })
            }
            Expr::ForLoop(f) => {
                if let Expr::Range(r) = &*f.expr
                    && let (Some(start), Some(end)) = (&r.start, &r.end)
                {
                    let label = &f.label;
                    let limits = &r.limits;
                    let start = primitive(visit(start)?);
                    let end = primitive(visit(end)?);
                    let pat = &f.pat;
                    let mut locals = locals.clone();
                    crate::scalar::visit_bindings(pat, &mut |binding| {
                        locals.insert(binding.ident.to_string());
                    });
                    let body = self.required_block(&f.body, None, &locals, inputs)?;
                    emit!(#label for #pat in (#start) #limits (#end) {
                        #support::frontend::scalar::specialization_step()?;
                        #body;
                    })
                } else {
                    rejected()
                }
            }
            Expr::Break(b) => {
                let label = &b.label;
                let value = b.expr.as_ref().map(|value| visit(value)).transpose()?;
                emit!(break #label #value)
            }
            Expr::Return(r) if self.evaluator_return.is_some() => {
                let label = self.evaluator_return.as_ref().unwrap();
                let value = r
                    .expr
                    .as_deref()
                    .map(|e| self.required_expr(e, Some(self.result_type), locals, inputs))
                    .transpose()?;
                emit!(break #label #value)
            }
            Expr::Continue(c) => {
                let label = &c.label;
                emit!(continue #label)
            }
            Expr::Call(c) if matches!(c.func.as_ref(), Expr::Path(p) if crate::scalar::methods::method_path(p)) =>
            {
                let Expr::Path(path) = c.func.as_ref() else {
                    unreachable!()
                };
                let trait_owner = crate::scalar::methods::trait_path(path);
                let evaluator = if trait_owner.is_some() {
                    "eval_method"
                } else {
                    "eval_inherent"
                };
                let target = crate::scalar::methods::rename_path(path, evaluator);
                let method = &target.path.segments.last().unwrap().ident;
                let args = c
                    .args
                    .iter()
                    .map(|arg| {
                        if let Expr::Path(path) = arg
                            && let Some(id) = path.path.get_ident()
                            && !locals.contains(&id.to_string())
                        {
                            if inputs.contains(&id.to_string()) {
                                return Ok(emit!(#id));
                            }
                            if matches!(
                                self.binding(path),
                                Some(Binding::Reg(..) | Binding::Buffer(_) | Binding::Columns)
                            ) {
                                return Ok(if trait_owner.is_some() {
                                    emit!(::core::option::Option::None)
                                } else {
                                    emit!(::core::option::Option::None::<i32>)
                                });
                            }
                        }
                        let arg = visit(arg)?;
                        Ok(emit!(::core::option::Option::Some(#arg)))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let output = match hint {
                    Some(Scalar::Bool) => emit!(bool),
                    Some(Scalar::Index | Scalar::Length) => emit!(usize),
                    _ => emit!(i32),
                };
                if let Some(owner) = trait_owner {
                    let method = &path.path.segments.last().unwrap().ident;
                    crate::declaration::trait_method_eval_dispatch(
                        support,
                        &owner,
                        method,
                        emit!(#target((#(#args,)*))?),
                        rejected(),
                    )
                } else {
                    emit!({
                    trait __SpliceOpaqueMethod {
                        fn #method<Args>(_: Args) -> #support::Result<#output> {
                            ::core::result::Result::Err(#support::BuildError::staging("opaque Rust method in comptime!"))
                        }
                    }
                    impl<T: ?Sized> __SpliceOpaqueMethod for T {}
                    #target((#(#args,)*))?
                    })
                }
            }
            Expr::Call(c) if matches!(c.func.as_ref(),Expr::Path(p) if p.qself.is_none()) => {
                let Expr::Path(path) = c.func.as_ref() else {
                    unreachable!()
                };
                let args = c
                    .args
                    .iter()
                    .map(|a| {
                        if let Expr::Path(p) = a
                            && let Some(id) = p.path.get_ident()
                            && !locals.contains(&id.to_string())
                        {
                            if inputs.contains(&id.to_string()) {
                                return Ok(emit!(#id));
                            }
                            if matches!(
                                self.binding(p),
                                Some(Binding::Reg(..) | Binding::Buffer(_) | Binding::Columns)
                            ) {
                                let ty = match self.binding(p) {
                                    Some(Binding::Reg(Scalar::Bool, _)) => emit!(bool),
                                    Some(Binding::Reg(Scalar::Int | Scalar::Inferred, _)) => {
                                        emit!(i32)
                                    }
                                    _ => return Ok(emit!(::core::option::Option::None)),
                                };
                                return Ok(emit!(::core::option::Option::None::<#ty>));
                            }
                        }
                        let a = visit(a)?;
                        Ok(emit!(::core::option::Option::Some(#a)))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut item = path.path.clone();
                let generics = std::mem::take(&mut item.segments.last_mut().unwrap().arguments);
                self.dispatch_call(
                    path,
                    emit!(#item::__SpliceDefinition::__splice_eval #generics ((#(#args,)*))?),
                    rejected(),
                )
            }
            Expr::Macro(m) if m.mac.path.is_ident("comptime") => {
                self.required_expr(&m.mac.parse_body()?, hint, locals, inputs)?
            }
            _ => rejected(),
        })
    }

    fn required_block(
        &self,
        block: &syn::Block,
        hint: Option<Scalar>,
        outer: &HashSet<String>,
        inputs: &HashSet<String>,
    ) -> Result<TokenStream> {
        let root = self.root;
        let mut locals = outer.clone();
        let mut code = TokenStream::new();
        for stmt in &block.stmts {
            match stmt {
                Stmt::Local(local) if local.init.is_some() => {
                    let init = local.init.as_ref().unwrap();
                    let value = self.required_expr(&init.expr, None, &locals, inputs)?;
                    let otherwise = init
                        .diverge
                        .as_ref()
                        .map(|(_, expr)| {
                            self.required_expr(expr, None, &locals, inputs)
                                .map(|expr| emit!(else { #expr }))
                        })
                        .transpose()?;
                    let pat = &local.pat;
                    bind_locals(&mut locals, pat);
                    code.extend(emit!(let #pat = #root::frontend::evaluation::Dropless::check()?.bind(#value) #otherwise;));
                }
                Stmt::Expr(e, semi) => {
                    let value = self.required_expr(
                        e,
                        if semi.is_some() { None } else { hint },
                        &locals,
                        inputs,
                    )?;
                    code.extend(emit!(#value #semi));
                }
                _ => {
                    return Ok(emit!({
                        return ::core::result::Result::Err(#root::BuildError::staging(
                            "unsupported statement in comptime!",
                        ));
                    }));
                }
            }
        }
        Ok(emit!({#code}))
    }
}

fn bind_locals(locals: &mut HashSet<String>, pattern: &Pat) {
    crate::scalar::visit_bindings(pattern, &mut |binding| {
        locals.insert(binding.ident.to_string());
    });
}

fn pure_wide_body(sig: &syn::Signature, block: &syn::Block, inputs: &HashSet<String>) -> bool {
    use crate::scalar::methods::primitive_type;

    fn pure_block(block: &syn::Block, names: &HashSet<String>) -> bool {
        let mut names = names.clone();
        block.stmts.iter().all(|stmt| match stmt {
            Stmt::Expr(expr, _) => pure_expr(expr, &names),
            Stmt::Local(local) => {
                let Some(init) = &local.init else {
                    return false;
                };
                if init.diverge.is_some()
                    || !local.attrs.is_empty()
                    || !pure_expr(&init.expr, &names)
                {
                    return false;
                }
                let pattern = match &local.pat {
                    Pat::Type(pattern) if primitive_type(&pattern.ty) => pattern.pat.as_ref(),
                    pat => pat,
                };
                match pattern {
                    Pat::Ident(pattern) if pattern.by_ref.is_none() && pattern.subpat.is_none() => {
                        names.insert(pattern.ident.to_string());
                        true
                    }
                    Pat::Wild(_) => true,
                    _ => false,
                }
            }
            _ => false,
        })
    }

    fn pure_expr(expr: &Expr, names: &HashSet<String>) -> bool {
        let pure = |expr: &Expr| pure_expr(expr, names);
        match expr {
            Expr::Lit(literal) => matches!(
                literal.lit,
                syn::Lit::Bool(_) | syn::Lit::Char(_) | syn::Lit::Int(_) | syn::Lit::Float(_)
            ),
            Expr::Path(path) => {
                path.qself.is_none()
                    && path
                        .path
                        .get_ident()
                        .is_some_and(|name| names.contains(&name.to_string()))
            }
            Expr::Paren(expr) => pure(&expr.expr),
            Expr::Group(expr) => pure(&expr.expr),
            Expr::Cast(cast) => primitive_type(&cast.ty) && pure(&cast.expr),
            Expr::Unary(expr) => {
                matches!(expr.op, syn::UnOp::Neg(_) | syn::UnOp::Not(_)) && pure(&expr.expr)
            }
            Expr::Binary(expr) => {
                crate::scalar::compound_op(&expr.op).is_none()
                    && pure(&expr.left)
                    && pure(&expr.right)
            }
            Expr::Block(expr) => pure_block(&expr.block, names),
            Expr::If(expr) => {
                pure(&expr.cond)
                    && pure_block(&expr.then_branch, names)
                    && expr
                        .else_branch
                        .as_ref()
                        .is_none_or(|(_, alternate)| pure(alternate))
            }
            Expr::Return(expr) => expr.expr.as_deref().is_some_and(pure),
            Expr::Macro(expr) if expr.mac.path.is_ident("comptime") => {
                expr.mac.parse_body::<Expr>().is_ok_and(|body| pure(&body))
            }
            _ => false,
        }
    }

    // A wide result stays in Rust during mandatory evaluation. Limit this
    // path to owned primitives and expressions with no opaque operands or
    // calls, and use the regular evaluator to check the actual operations.
    sig.inputs.iter().all(|arg| matches!(arg, FnArg::Typed(arg)
        if primitive_type(&arg.ty) && matches!(arg.pat.as_ref(), Pat::Ident(pat) if pat.by_ref.is_none() && pat.subpat.is_none())))
        && pure_block(block, inputs)
}

fn evaluable_statement(statement: &Stmt) -> bool {
    match statement {
        Stmt::Expr(e, _) => evaluable(e),
        Stmt::Local(l) => l.init.as_ref().is_some_and(|i| {
            evaluable(&i.expr) && i.diverge.as_ref().is_none_or(|(_, e)| evaluable(e))
        }),
        _ => false,
    }
}
fn evaluable(expr: &Expr) -> bool {
    match expr {
        Expr::Path(_) | Expr::Lit(_) => true,
        Expr::Tuple(t) => t.elems.iter().all(evaluable),
        Expr::Array(a) => a.elems.iter().all(evaluable),
        Expr::Struct(s) => {
            s.fields.iter().all(|f| evaluable(&f.expr)) && s.rest.as_deref().is_none_or(evaluable)
        }
        Expr::Let(l) => evaluable(&l.expr),
        Expr::Match(m) => {
            evaluable(&m.expr)
                && m.arms.iter().all(|a| {
                    a.guard.as_ref().is_none_or(|(_, guard)| evaluable(guard)) && evaluable(&a.body)
                })
        }
        Expr::Paren(p) => evaluable(&p.expr),
        Expr::Group(p) => evaluable(&p.expr),
        Expr::Call(call) => {
            matches!(&*call.func, Expr::Path(_))
                && !super::simd::uses_lane(&call.func)
                && call.args.iter().all(evaluable)
        }
        Expr::Field(field) => evaluable(&field.base),
        Expr::Cast(cast) => evaluable(&cast.expr),
        Expr::Binary(b) => evaluable(&b.left) && evaluable(&b.right),
        Expr::Unary(u) => evaluable(&u.expr),
        Expr::MethodCall(call) => {
            let name = call.method.to_string();
            (matches!(name.as_str(), "len" | "is_empty") || super::memory::primitive_method(&name))
                && evaluable(&call.receiver)
                && call.args.iter().all(evaluable)
        }
        Expr::Block(b) => b.block.stmts.iter().all(evaluable_statement),
        Expr::If(i) => {
            evaluable(&i.cond)
                && i.then_branch.stmts.iter().all(evaluable_statement)
                && i.else_branch.as_ref().is_none_or(|(_, e)| evaluable(e))
        }
        Expr::While(w) => evaluable(&w.cond) && w.body.stmts.iter().all(evaluable_statement),
        Expr::Loop(l) => l.body.stmts.iter().all(evaluable_statement),
        Expr::ForLoop(f) => {
            matches!(&*f.expr, Expr::Range(r) if r.start.as_ref().is_some_and(|e| evaluable(e)) && r.end.as_ref().is_some_and(|e| evaluable(e)))
                && f.body.stmts.iter().all(evaluable_statement)
        }
        Expr::Return(r) => r.expr.as_deref().is_none_or(evaluable),
        Expr::Break(b) => b.expr.as_ref().is_none_or(|e| evaluable(e)),
        Expr::Continue(_) => true,
        Expr::Macro(m) if m.mac.path.is_ident("comptime") => {
            m.mac.parse_body::<Expr>().is_ok_and(|e| evaluable(&e))
        }
        _ => false,
    }
}

pub(in crate::scalar) fn can_fold(sig: &syn::Signature, block: &syn::Block) -> bool {
    sig.inputs.iter().all(|arg|matches!(arg,FnArg::Typed(a) if matches!(Scalar::from_type(&a.ty,&[]), Some(Scalar::Int|Scalar::Bool)))) &&
    block.stmts.iter().all(evaluable_statement)
}

pub(crate) fn uses_stage_requirements(block: &syn::Block) -> bool {
    use syn::visit::{Visit, visit_expr_call};
    struct StageSensitive(bool);
    impl<'ast> Visit<'ast> for StageSensitive {
        fn visit_macro(&mut self, mac: &'ast syn::Macro) {
            self.0 |= mac.path.is_ident("comptime");
        }
        fn visit_expr_call(&mut self, c: &'ast syn::ExprCall) {
            self.0 = true;
            visit_expr_call(self, c);
        }
    }
    let mut uses = StageSensitive(false);
    uses.visit_block(block);
    uses.0
}

/// Return a build error instead of panicking when host operator `kind` faults
/// on `operands` inside `comptime!`.
fn comptime_fault(support: &syn::Path, kind: &str, operands: TokenStream) -> Option<TokenStream> {
    let (guard, division) = super::host::operator_fault(support, kind, operands)?;
    let checks = super::host::fault_checks(division);
    Some(emit! {
        use #support::frontend::scalar::{HostArithmetic as _, HostShift as _};
        if #checks && let ::core::option::Option::Some(fault) = #guard {
            return ::core::result::Result::Err(#support::BuildError::Comptime(fault.trap()));
        }
    })
}
