//! Generate the mandatory build-time evaluation of an expression. Each
//! operand is checked where Rust would evaluate it, including short circuits.
use super::{Binding, Evaluation, Lower, Scalar};
use proc_macro2::{Ident, TokenStream};
use syn::{BinOp, Expr, FnArg, Pat, Result, Stmt};

impl Lower<'_> {
    /// The mandatory evaluator of a kernel (`__splice_eval`) or method
    /// (`method`), and whether it evaluates the body: a body with a node the
    /// evaluator cannot run on its `Option` inputs returns a staging error.
    pub(in crate::scalar) fn evaluator(
        &self,
        sig: &syn::Signature,
        block: &syn::Block,
        vis: &syn::Visibility,
        method: Option<&Ident>,
    ) -> Result<(TokenStream, bool)> {
        use crate::scalar::methods::primitive_type;
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
        // A wide result (a primitive other than `i32` and `bool`) stays in
        // Rust during mandatory evaluation: only owned primitives and
        // expressions without opaque operands or calls.
        let wide = method.is_some()
            && matches!(&sig.output, syn::ReturnType::Type(_, ty) if primitive_type(ty)
                && !matches!(Scalar::from_type(ty, &[]), Some(Scalar::I32 | Scalar::Bool)));
        let parameters = if wide {
            sig.inputs.iter().all(|arg| matches!(arg, FnArg::Typed(arg)
                if primitive_type(&arg.ty) && matches!(arg.pat.as_ref(), Pat::Ident(pat) if pat.by_ref.is_none() && pat.subpat.is_none())))
        } else {
            method.is_some()
                || types.iter().all(|t| {
                    matches!(t, syn::Type::Reference(reference) if reference.mutability.is_none())
                        || matches!(
                            Scalar::from_type(t, self.types),
                            Some(Scalar::I32 | Scalar::Bool | Scalar::Index)
                        )
                })
        };
        let mut evaluator = self.clone();
        let exit = syn::Lifetime::new("'__splice_evaluated", proc_macro2::Span::mixed_site());
        let evaluation = std::rc::Rc::new(Evaluation {
            exit: exit.clone(),
            wide,
            unevaluable: std::cell::Cell::new(!parameters),
        });
        evaluator.evaluation = Some(evaluation.clone());
        let body = evaluator.required_block(block, Some(self.result_type))?;
        let evaluable = !evaluation.unevaluable.get();
        let body = if evaluable {
            emit!(#[allow(unused_labels)] #exit: { #body })
        } else {
            emit!(
                return ::core::result::Result::Err(#root::BuildError::staging(
                    "body cannot be evaluated at specialization"
                ))
            )
        };
        let overflow = super::overflow_prologue(root);
        Ok((
            emit! {
                #[doc(hidden)]
                #[allow(unused_variables,unreachable_code,clippy::type_complexity)]
                #vis fn #name #generics ((#(#names,)*): (#(::core::option::Option<#types>,)*))
                    -> #root::Result<#ty> #where_clause {
                    let _depth=#root::frontend::scalar::enter_expansion()?;
                    #overflow
                    #(#checks)*
                    ::core::result::Result::Ok(#body)
                }
            },
            evaluable,
        ))
    }

    /// Give up on evaluating the body: a node the evaluator cannot run.
    /// The returned code is never emitted.
    fn unevaluable(&self) -> TokenStream {
        if let Some(evaluation) = &self.evaluation {
            evaluation.unevaluable.set(true);
        }
        TokenStream::new()
    }

    /// Whether the parameters are `Option`s of their values: the mandatory
    /// evaluator of a kernel with parameters.
    fn inputs(&self) -> bool {
        self.optional() && self.body.has_params()
    }

    /// Whether `path` names a parameter that the evaluator reads as an input.
    fn input(&self, path: &syn::ExprPath) -> bool {
        self.inputs()
            && self
                .body
                .decl_of(path)
                .is_some_and(|decl| self.body.is_param(decl))
    }

    /// Whether `path` names a local of the evaluated Rust: one the lowering
    /// has not bound (it is declared inside the evaluated expression).
    fn evaluated_local(&self, path: &syn::ExprPath) -> bool {
        self.body
            .decl_of(path)
            .is_some_and(|decl| !self.bindings.contains_key(&decl))
            && !self.input(path)
    }

    pub(super) fn required(&self, expr: &Expr, hint: Option<Scalar>) -> Result<TokenStream> {
        self.required_expr(expr, hint)
    }

    fn required_expr(&self, expr: &Expr, hint: Option<Scalar>) -> Result<TokenStream> {
        if self.evaluation.as_ref().is_some_and(|e| e.wide) && !wide_node(expr) {
            return Ok(self.unevaluable());
        }
        let support = &self.root;
        let visit = |e: &Expr| self.required_expr(e, None);
        let primitive = |value: TokenStream| {
            let check = self.primitive(emit!(&value));
            emit!({
                let value = #value;
                #check
                value
            })
        };
        // The error of an operation the evaluator does not run, where Rust
        // reaches it: the Rust function a call path selects.
        let opaque = || {
            // An unevaluated operand supplies the Rust result type when all
            // names have their original type in the specialization factory.
            if !self.inputs() && !super::simd::uses_lane(expr) && self.host_only(expr) {
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
        // A node the evaluator does not run. Its inputs are `Option`s, where
        // the Rust expression does not type-check: the body is unevaluable.
        let rejected = || {
            if self.inputs() {
                self.unevaluable()
            } else {
                opaque()
            }
        };
        Ok(match expr {
            Expr::Lit(_) => emit!(#expr),
            Expr::Path(path) if self.input(path) => {
                emit!((#path).ok_or_else(|| #support::BuildError::staging("runtime argument read in comptime!"))?)
            }
            Expr::Path(path) if self.evaluated_local(path) => emit!(#path),
            Expr::Path(path) => match self.binding(path) {
                Some(Binding::Reg(ty, _)) => {
                    let ty = match ty {
                        Scalar::Bool => emit!(bool),
                        Scalar::Index | Scalar::Length => emit!(usize),
                        _ => emit!(i32),
                    };
                    // The printer has no registers: a runtime value is unknown.
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
                    emit!(<#ty as #support::frontend::scalar::ImportWord>::from_constant(
                        #support::frontend::scalar::RunArg::known(&#path).ok_or_else(|| #support::BuildError::staging("runtime value in comptime!"))?
                    ))
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
                None => self.dropless(path),
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
                let fields = literal
                    .fields
                    .iter()
                    .map(|field| {
                        let (member, value) = (&field.member, visit(&field.expr)?);
                        Ok(emit!(#member: #value))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let rest = literal
                    .rest
                    .as_deref()
                    .map(|rest| visit(rest).map(|rest| emit!(.. #rest)))
                    .transpose()?;
                let path = syn::ExprPath {
                    attrs: literal.attrs.clone(),
                    qself: literal.qself.clone(),
                    path: literal.path.clone(),
                };
                self.dropless(emit!(#path { #(#fields,)* #rest }))
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
                    if matches!(&*b.left, Expr::Path(p) if self.evaluated_local(p)) {
                        let lhs = &b.left;
                        let check = comptime_fault(support, kind, emit!(&(&#lhs, &right)));
                        let primitive = self.primitive(emit!(&#lhs));
                        emit!({
                            #primitive
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
            Expr::Block(b) => self.required_block(&b.block, hint)?,
            Expr::Field(field)
                if matches!(&*field.base, Expr::Path(path)
                    if matches!(self.binding(path), Some(Binding::Aggregate(_)))) =>
            {
                let (word, _, constructor) = self
                    .aggregate_field(field)
                    .ok_or_else(|| syn::Error::new_spanned(field, "missing aggregate field"))?;
                // The printer keeps the known constants of an aggregate.
                let known = if self.source_fault.is_some() {
                    word
                } else {
                    emit!(#support::frontend::scalar::RunArg::known(&#word))
                };
                let member = &field.member;
                emit!(#support::frontend::scalar::known_field(#constructor, |v| &v.#member, #known)?)
            }
            Expr::Field(field) if matches!(&*field.base, Expr::Path(path) if self.input(path)) => {
                let base = &field.base;
                let place = emit!(*#base.as_ref().ok_or_else(|| #support::BuildError::staging("runtime argument read in comptime!"))?);
                self.known_field(place, &field.member)
            }
            Expr::Field(field) if self.host_only(expr) || self.inputs() => {
                self.known_field(visit(&field.base)?, &field.member)
            }
            Expr::If(i) => {
                let cond = if let Expr::Let(binding) = &*i.cond {
                    let value = visit(&binding.expr)?;
                    let pattern = &binding.pat;
                    emit!(let #pattern = #value)
                } else {
                    self.required_expr(&i.cond, Some(Scalar::Bool))?
                };
                let yes = self.required_block(&i.then_branch, hint)?;
                let no = i
                    .else_branch
                    .as_ref()
                    .map(|(_, e)| self.required_expr(e, hint))
                    .transpose()?
                    .map(|e| emit!(else {#e}));
                emit!(if #cond {#yes} #no)
            }
            Expr::Match(m) => {
                let value = visit(&m.expr)?;
                let mut arms = Vec::with_capacity(m.arms.len());
                for arm in &m.arms {
                    let pat = &arm.pat;
                    let guard = arm
                        .guard
                        .as_ref()
                        .map(|(_, guard)| {
                            self.required_expr(guard, Some(Scalar::Bool))
                                .map(|test| emit!(if #test))
                        })
                        .transpose()?;
                    let body = self.required_expr(&arm.body, hint)?;
                    let attrs = &arm.attrs;
                    arms.push(emit!(#(#attrs)* #pat #guard => { #body },));
                }
                emit!(match #value { #(#arms)* })
            }
            Expr::While(w) => {
                let label = &w.label;
                let cond = self.required_expr(&w.cond, Some(Scalar::Bool))?;
                let body = self.required_block(&w.body, None)?;
                emit!(#label while #cond { #support::frontend::scalar::specialization_step()?; #body; })
            }
            Expr::Loop(l) => {
                let label = &l.label;
                let body = self.required_block(&l.body, None)?;
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
                    let body = self.required_block(&f.body, None)?;
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
            Expr::Return(r) if self.evaluation.is_some() => {
                let label = &self.evaluation.as_ref().unwrap().exit;
                let value = r
                    .expr
                    .as_deref()
                    .map(|e| self.required_expr(e, Some(self.result_type)))
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
                // A target-dependent operation can only be staged. In a mandatory
                // comptime expression, preserve Rust's result type in the error arm.
                if super::simd::simd_path(path) {
                    return Ok(if self.optional() {
                        self.unevaluable()
                    } else {
                        opaque()
                    });
                }
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
                        if let Expr::Path(path) = arg {
                            if self.input(path) {
                                return Ok(emit!(#path));
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
                let ordinary = if let Some(owner) = trait_owner {
                    let method = &path.path.segments.last().unwrap().ident;
                    crate::declaration::trait_method_eval_dispatch(
                        support,
                        &owner,
                        method,
                        emit!(#target((#(#args,)*))?),
                        opaque(),
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
                };
                if super::memory::math_op(&path.path.segments.last().unwrap().ident.to_string())
                    .is_some()
                {
                    let mut owner = path.path.clone();
                    owner.segments.pop();
                    owner.segments.pop_punct();
                    let args = c.args.iter().map(visit).collect::<Result<Vec<_>>>()?;
                    let unknown = emit!(if false { #path(#(#args),*) } else {
                        return ::core::result::Result::Err(#support::BuildError::staging("intrinsic requires kernel staging"))
                    });
                    if owner.segments.is_empty()
                        || Scalar::from_type(&syn::parse_quote!(#owner), self.types).is_some()
                    {
                        unknown
                    } else {
                        super::calls::select_builtin(support, &owner, "math", unknown, ordinary)
                    }
                } else {
                    ordinary
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
                        if let Expr::Path(p) = a {
                            if self.input(p) {
                                return Ok(emit!(#p));
                            }
                            if matches!(
                                self.binding(p),
                                Some(Binding::Reg(..) | Binding::Buffer(_) | Binding::Columns)
                            ) {
                                let ty = match self.binding(p) {
                                    Some(Binding::Reg(Scalar::Bool, _)) => emit!(bool),
                                    Some(Binding::Reg(Scalar::I32 | Scalar::Inferred, _)) => {
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
                    opaque(),
                )
            }
            Expr::Macro(m) if m.mac.path.is_ident("comptime") => {
                self.required_expr(self.body.comptime(&m.mac), hint)?
            }
            _ => rejected(),
        })
    }

    fn required_block(&self, block: &syn::Block, hint: Option<Scalar>) -> Result<TokenStream> {
        let root = self.root;
        let mut code = TokenStream::new();
        for stmt in &block.stmts {
            match stmt {
                Stmt::Local(local) if local.init.is_some() => {
                    let init = local.init.as_ref().unwrap();
                    if self.evaluation.as_ref().is_some_and(|e| e.wide) && !wide_local(local) {
                        return Ok(self.unevaluable());
                    }
                    let value = match &*init.expr {
                        // A tuple struct's constructor runs as Rust.
                        Expr::Call(call) if self.body.projected_local(local) => {
                            let func = &call.func;
                            let args = call
                                .args
                                .iter()
                                .map(|arg| self.required_expr(arg, None))
                                .collect::<Result<Vec<_>>>()?;
                            emit!(#func(#(#args),*))
                        }
                        init => self.required_expr(init, None)?,
                    };
                    let otherwise = init
                        .diverge
                        .as_ref()
                        .map(|(_, expr)| {
                            self.required_expr(expr, None)
                                .map(|expr| emit!(else { #expr }))
                        })
                        .transpose()?;
                    let pat = &local.pat;
                    let value = self.dropless(value);
                    code.extend(emit!(let #pat = #value #otherwise;));
                }
                Stmt::Expr(e, semi) => {
                    let value = self.required_expr(e, if semi.is_some() { None } else { hint })?;
                    code.extend(emit!(#value #semi));
                }
                _ if self.inputs() => return Ok(self.unevaluable()),
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

/// The nodes the evaluator of a wide result runs in Rust.
fn wide_node(expr: &Expr) -> bool {
    match expr {
        Expr::Lit(literal) => matches!(
            literal.lit,
            syn::Lit::Bool(_) | syn::Lit::Char(_) | syn::Lit::Int(_) | syn::Lit::Float(_)
        ),
        Expr::Path(path) => path.qself.is_none() && path.path.get_ident().is_some(),
        Expr::Cast(cast) => crate::scalar::methods::primitive_type(&cast.ty),
        Expr::Unary(unary) => matches!(unary.op, syn::UnOp::Neg(_) | syn::UnOp::Not(_)),
        Expr::Binary(binary) => crate::scalar::compound_op(&binary.op).is_none(),
        Expr::Return(ret) => ret.expr.is_some(),
        Expr::Macro(m) => m.mac.path.is_ident("comptime"),
        Expr::Paren(_) | Expr::Group(_) | Expr::Block(_) | Expr::If(_) => true,
        _ => false,
    }
}

/// A local the evaluator of a wide result binds: a named or `_` pattern,
/// at most annotated with a primitive type.
fn wide_local(local: &syn::Local) -> bool {
    let pattern = match &local.pat {
        Pat::Type(pattern) if crate::scalar::methods::primitive_type(&pattern.ty) => &*pattern.pat,
        pat => pat,
    };
    local.attrs.is_empty()
        && local
            .init
            .as_ref()
            .is_some_and(|init| init.diverge.is_none())
        && match pattern {
            Pat::Ident(pattern) => pattern.by_ref.is_none() && pattern.subpat.is_none(),
            Pat::Wild(_) => true,
            _ => false,
        }
}

/// Whether a call with known arguments may fold through the evaluator: one
/// that takes and returns only `i32` and `bool` words.
pub(in crate::scalar) fn foldable_signature(sig: &syn::Signature) -> bool {
    let word = |ty: &syn::Type| {
        matches!(Scalar::from_type(ty, &[]), Some(Scalar::I32 | Scalar::Bool))
            || matches!(ty, syn::Type::Tuple(unit) if unit.elems.is_empty())
    };
    let result = match &sig.output {
        syn::ReturnType::Default => true,
        syn::ReturnType::Type(_, ty) => word(ty),
    };
    result
        && sig
            .inputs
            .iter()
            .all(|arg| matches!(arg, FnArg::Typed(a) if word(&a.ty)))
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
