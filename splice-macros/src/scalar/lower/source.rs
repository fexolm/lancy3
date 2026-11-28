//! The printer: the generator code of `print`, which renders the
//! specialized kernel as Rust. It walks the body once: a value the
//! generator knows ([`crate::kast::Body::known`]) is computed as the IR
//! lowering computes it ([`Lower::known_value`]) and printed as a literal;
//! every other node is printed as written, with its operands rendered.
use super::{Binding, LoopKind, Lower, Scalar, Stage, err};
use proc_macro2::{Ident, TokenStream};
use quote::{ToTokens, format_ident};
use syn::visit::{self, Visit};
use syn::visit_mut::{self, VisitMut};
use syn::{Block, Expr, Result, Stmt};

impl Lower<'_> {
    pub(in crate::scalar) fn source_body(&self, block: &Block) -> Result<TokenStream> {
        let mut scope = self.clone();
        let result_type = self.result_type;
        let mut statements = Vec::new();
        let end = self.fresh_label("source");
        scope.source_fault = Some(end.clone());
        for (i, stmt) in block.stmts.iter().enumerate() {
            let rendered = match stmt {
                Stmt::Local(local) => {
                    let Some(init) = &local.init else {
                        return err(local, "initialize kernel locals at their declaration");
                    };
                    let pat = &local.pat;
                    if let Some(literal) = super::aggregate::Literal::of(&scope, local, &init.expr)
                    {
                        let value = scope.source_syntax(&init.expr)?;
                        statements.push(scope.source_aggregate(pat, &literal)?);
                        emit!(::std::format!("let {} = {};", ::core::stringify!(#pat), #value))
                    } else {
                        let hint = match pat {
                            syn::Pat::Type(p) => Scalar::from_type(&p.ty, scope.types),
                            _ => None,
                        };
                        if scope.body.known(&init.expr) && !scope.body.runtime_local(local) {
                            // The generator holds the value; its uses print it.
                            let value = scope.known_operand(&init.expr, hint)?;
                            scope.bind(pat, Binding::Host(scope.region_depth));
                            statements.push(emit!(let #pat = #value;));
                            continue;
                        }
                        let value = scope.source_expr(&init.expr, hint)?;
                        // A value only the listing shows.
                        scope.bind(pat, Binding::Reg(Scalar::Inferred, false));
                        emit!(::std::format!("let {} = {};", ::core::stringify!(#pat), #value))
                    }
                }
                Stmt::Expr(expr, semi) => {
                    let last = i + 1 == block.stmts.len() && semi.is_none();
                    if !last
                        && matches!(
                            expr,
                            Expr::MethodCall(_) | Expr::Assign(_) | Expr::Binary(_)
                        )
                        && scope.body.known(expr)
                    {
                        // The same checked host operation as the IR lowering.
                        let value = scope.known_operand(expr, None)?;
                        statements.push(emit!(#value;));
                        continue;
                    }
                    let rendered = scope.source_expr(expr, last.then_some(result_type))?;
                    if last {
                        rendered
                    } else {
                        emit!({ let text = #rendered; if text.is_empty() { text } else { ::std::format!("{text};") } })
                    }
                }
                Stmt::Item(item @ syn::Item::Use(_)) => {
                    statements.push(emit!(#[allow(unused_imports)] #item));
                    emit!(::std::borrow::ToOwned::to_owned(::core::stringify!(#item)))
                }
                _ => return err(stmt, "unsupported statement in Rust source output"),
            };
            statements.push(emit!({ let text = #rendered; if !text.is_empty() {
                __splice_text.push_str(&text); __splice_text.push('\n');
            } if __splice_source.transfer.is_some() { break #end (::std::string::String::new()); }}));
            if let Stmt::Expr(Expr::Return(_) | Expr::Break(_) | Expr::Continue(_), _) = stmt {
                break;
            }
        }
        Ok(
            emit!({ let mut __splice_text = ::std::string::String::new();
                #[allow(unused_labels)] let fault = #end: { #(#statements)* ::std::string::String::new() };
                __splice_text.push_str(&fault); __splice_text }),
        )
    }

    fn source_expr(&self, e: &Expr, hint: Option<Scalar>) -> Result<TokenStream> {
        let root = self.root;
        let support = emit!(#root::frontend::source);
        let braces = |body: TokenStream| emit!(#support::block(&#body));
        if matches!(e, Expr::ForLoop(_) | Expr::While(_) | Expr::Loop(_))
            && let Some(code) = self.host_loop(e)?
        {
            return Ok(emit!({ #code; ::std::string::String::new() }));
        }
        match e {
            Expr::Block(b) => return Ok(braces(self.source_body(&b.block)?)),
            Expr::If(i) => {
                let (known, scope) = self.source_condition(&i.cond)?;
                let yes = braces(scope.source_body(&i.then_branch)?);
                let no = i
                    .else_branch
                    .as_ref()
                    .map(|(_, e)| {
                        if known.is_some() {
                            self.source_expr(e, hint)
                        } else {
                            let no = scope.isolated_expr(e, hint)?;
                            Ok(if matches!(&**e, Expr::If(_)) {
                                braces(no)
                            } else {
                                no
                            })
                        }
                    })
                    .transpose()?;
                if let Some(cond) = known {
                    let no = no.unwrap_or_else(|| emit!(::std::borrow::ToOwned::to_owned("{}")));
                    return Ok(emit!(if #cond { #yes } else { #no }));
                }
                let cond = self.source_expr(&i.cond, Some(Scalar::Bool))?;
                let yes = isolate(yes);
                return Ok(match no {
                    Some(no) => emit!(::std::format!("if {} {} else {}", #cond, #yes, #no)),
                    None => emit!(::std::format!("if {} {}", #cond, #yes)),
                });
            }
            Expr::Match(m) => {
                if self.body.known(&m.expr) {
                    let value = self.known_operand(&m.expr, None)?;
                    return self.source_known_match(m, value.code, hint);
                }
                let test = self.source_expr(&m.expr, None)?;
                let scope = self.nested_region();
                let mut arms = Vec::new();
                for arm in &m.arms {
                    let pat = &arm.pat;
                    let body = scope.isolated_expr(&arm.body, hint)?;
                    let guard = if let Some((_, g)) = &arm.guard {
                        let guard = scope.isolated_expr(g, Some(Scalar::Bool))?;
                        emit!(::std::format!(" if {}", #guard))
                    } else {
                        emit!(::std::string::String::new())
                    };
                    arms.push(emit!(
                        ::std::format!("{}{} => {},", ::core::stringify!(#pat), #guard, #body)
                    ));
                }
                return Ok(emit!(
                    ::std::format!("match {} {}", #test, #support::block(&[#(#arms),*].join("\n")))
                ));
            }
            Expr::ForLoop(f) => {
                let pat = &f.pat;
                // A known iterator other than a range is unrolled; a
                // runtime one is a slice.
                if !matches!(&*f.expr, Expr::Range(_)) && self.body.known(&f.expr) {
                    let iter = self.plan_iterable(&f.expr)?;
                    let mut scope = self.with_pattern(pat);
                    scope.enter_loop(f.label.as_ref(), LoopKind::Unrolled);
                    let body = scope.source_body(&f.body)?;
                    return Ok(emit!({ let mut text = ::std::string::String::new();
                        let enclosing = __splice_source.enter_unrolled();
                        for #pat in #iter {
                            #root::frontend::scalar::specialization_step()?;
                            let body = #body;
                            text.push_str(__splice_source.iteration_label());
                            text.push_str(&#support::block(&body)); text.push_str(";\n");
                            match __splice_source.transfer {
                                ::core::option::Option::Some(#support::Transfer::Break) => { __splice_source.transfer = ::core::option::Option::None; break; }
                                ::core::option::Option::Some(#support::Transfer::Continue) => __splice_source.transfer = ::core::option::Option::None,
                                ::core::option::Option::Some(#support::Transfer::Return) => break,
                                ::core::option::Option::None => {}
                            }
                        }
                        __splice_source.exit_unrolled(enclosing, &text)
                    }));
                }
                let mut scope = self.nested_region();
                scope.enter_loop(f.label.as_ref(), LoopKind::Runtime);
                let label = label_text(f.label.as_ref());
                scope.bind(pat, Binding::Reg(Scalar::I32, false));
                let range = self.source_syntax(&f.expr)?;
                let body = isolate(braces(scope.source_body(&f.body)?));
                return Ok(emit!(
                    ::std::format!("{}for {} in {} {}", #label, ::core::stringify!(#pat), #range, #body)
                ));
            }
            Expr::While(w) => {
                let mut scope = self.nested_region();
                let cond = scope.isolated_expr(&w.cond, Some(Scalar::Bool))?;
                scope.enter_loop(w.label.as_ref(), LoopKind::Runtime);
                let label = label_text(w.label.as_ref());
                let body = isolate(braces(scope.source_body(&w.body)?));
                return Ok(emit!(::std::format!("{}while {} {}", #label, #cond, #body)));
            }
            Expr::Loop(l) => {
                let mut scope = self.nested_region();
                scope.enter_loop(l.label.as_ref(), LoopKind::Runtime);
                let label = label_text(l.label.as_ref());
                let body = isolate(braces(scope.source_body(&l.body)?));
                return Ok(emit!(::std::format!("{}loop {}", #label, #body)));
            }
            Expr::MethodCall(method) => {
                if let Expr::Index(index) = &*method.receiver
                    && let Expr::Path(table) = &*index.expr
                    && matches!(self.binding(table), Some(Binding::Columns))
                    && let Some(elem) = method
                        .turbofish
                        .as_ref()
                        .and_then(|args| super::simd::element_argument(args, self.types))
                {
                    let slot = self.source_known(&index.index)?;
                    let element = elem.tokens(root, self.types);
                    let ty = elem.ty(root, self.types);
                    return Ok(emit!({
                        let slot = #slot;
                        #root::frontend::scalar::column_kind(__splice_column_layout, slot, #element)?;
                        ::std::format!("{}[{}].as_slice::<{}>()", ::core::stringify!(#table), slot, ::core::stringify!(#ty))
                    }));
                }
            }
            Expr::Call(call) => {
                let Expr::Path(p) = &*call.func else {
                    return self.source_syntax(e);
                };
                if crate::scalar::methods::method_path(p) {
                    let syntax = self.source_syntax(e)?;
                    if super::simd::simd_path(p)
                        && !self.local_owner(p)
                        && p.qself.is_none()
                        && p.path.segments.len() == 2
                    {
                        let head = &p.path.segments.first().unwrap().ident;
                        if head == "Lane" || head == "Mask" {
                            let owner = super::simd::owner_path(p).unwrap();
                            let qualified = emit!(
                                ::std::format!("{}::std::intrinsics::{}", ::core::stringify!(#root), #syntax)
                            );
                            return Ok(super::calls::select_builtin(
                                root, &owner, "simd", qualified, syntax,
                            ));
                        }
                    }
                    return Ok(syntax);
                }
                if !call.args.iter().all(|arg| self.body.known(arg)) {
                    let mut path = p.path.clone();
                    let fish = std::mem::take(&mut path.segments.last_mut().unwrap().arguments);
                    let args = call
                        .args
                        .iter()
                        .map(|arg| {
                            if self.body.known(arg) {
                                let value = self.known_operand(arg, None)?;
                                Ok(emit!(#root::frontend::scalar::Host(#value)))
                            } else {
                                self.source_expr(arg, None)
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let syntax = self.source_syntax(e)?;
                    let kernel = emit!(
                        match #path::__splice_source_call #fish (#(#args,)* __splice_source)? {
                            ::core::option::Option::Some(text) => text,
                            ::core::option::Option::None => #syntax,
                        }
                    );
                    return if p.qself.is_none() && !p.path.is_ident(self.name) {
                        Ok(self.dispatch_call(p, kernel, syntax))
                    } else {
                        Ok(kernel)
                    };
                }
            }
            Expr::Return(_) | Expr::Break(_) | Expr::Continue(_) => {
                let (transfer, unrolled, label) = match e {
                    Expr::Return(_) => (emit!(Return), None, None),
                    Expr::Break(b) => (emit!(Break), Some(emit!(break_unrolled)), b.label.as_ref()),
                    Expr::Continue(c) => (
                        emit!(Continue),
                        Some(emit!(continue_unrolled)),
                        c.label.as_ref(),
                    ),
                    _ => unreachable!(),
                };
                let frames = match unrolled {
                    Some(_) => self.jump_target(e, label)?,
                    None => &[],
                };
                // The listing shows an unrolled loop as its iterations; only
                // the innermost one is left by an unlabeled jump.
                if label.is_some() && frames.iter().any(|frame| frame.kind == LoopKind::Unrolled) {
                    return err(
                        e,
                        "`print` cannot show a labeled break or continue across an unrolled loop",
                    );
                }
                let text = if let Some(unrolled) = unrolled
                    && frames[0].kind == LoopKind::Unrolled
                {
                    emit!(__splice_source.#unrolled())
                } else {
                    self.source_syntax(e)?
                };
                return Ok(
                    emit!({ let text = #text; __splice_source.transfer = ::core::option::Option::Some(#support::Transfer::#transfer); text }),
                );
            }
            Expr::Binary(binary) if matches!(binary.op, syn::BinOp::And(_) | syn::BinOp::Or(_)) => {
                if self.body.known(&binary.left) {
                    let lhs = self.known_operand(&binary.left, Some(Scalar::Bool))?;
                    let rhs = self.source_expr(&binary.right, Some(Scalar::Bool))?;
                    return Ok(if matches!(binary.op, syn::BinOp::And(_)) {
                        emit!(if #lhs { #rhs } else { ::std::borrow::ToOwned::to_owned("false") })
                    } else {
                        emit!(if #lhs { ::std::borrow::ToOwned::to_owned("true") } else { #rhs })
                    });
                }
                let lhs = self.source_expr(&binary.left, Some(Scalar::Bool))?;
                let rhs = self
                    .nested_region()
                    .isolated_expr(&binary.right, Some(Scalar::Bool))?;
                let op = &binary.op;
                return Ok(emit!(
                    ::std::format!("{} {} {}", #lhs, ::core::stringify!(#op), #rhs)
                ));
            }
            Expr::Range(_) => {
                return self.source_syntax(e);
            }
            _ => {}
        }
        // A literal prints as written.
        if self.body.known(e)
            && let Some(value) = self.known_value(e, hint)?
            && value.stage == Stage::Host
        {
            return Ok(emit!({
                use #support::TryLiteral as _;
                let value = &(#value);
                (&#support::Literal(value)).try_literal()?
            }));
        }
        self.source_syntax(e)
    }

    /// The condition of an `if`: its Rust code if the generator knows it,
    /// and the scope of the branch it guards.
    fn source_condition(&self, cond: &Expr) -> Result<(Option<TokenStream>, Self)> {
        if let Expr::Let(binding) = cond {
            let value = self.source_known(&binding.expr)?;
            let pattern = &binding.pat;
            return Ok((
                Some(emit!(let #pattern = #value)),
                self.with_pattern(pattern),
            ));
        }
        Ok(if self.body.known(cond) {
            let value = self.known_operand(cond, Some(Scalar::Bool))?;
            (Some(value.code), self.clone())
        } else {
            (None, self.nested_region())
        })
    }

    /// The value of `e`, which the generator must know.
    fn source_known(&self, e: &Expr) -> Result<TokenStream> {
        if !self.body.known(e) {
            return err(e, "expected a comptime value; found a runtime expression");
        }
        Ok(self.known_operand(e, None)?.code)
    }

    /// A `match` on a known value. With runtime guards, the arms the
    /// generator selects print as an `if` chain of those guards.
    fn source_known_match(
        &self,
        m: &syn::ExprMatch,
        value: TokenStream,
        hint: Option<Scalar>,
    ) -> Result<TokenStream> {
        let root = self.root;
        let mut guards = Vec::new();
        for arm in &m.arms {
            let scope = self.with_pattern(&arm.pat);
            // A known guard is its code; a runtime one is printed.
            let guard = match &arm.guard {
                Some((_, g)) if self.body.known(g) => {
                    Some(Some(scope.known_operand(g, Some(Scalar::Bool))?.code))
                }
                Some(_) => Some(None),
                None => None,
            };
            guards.push((scope, guard));
        }
        let runtime = guards.iter().any(|(_, guard)| matches!(guard, Some(None)));
        if !runtime {
            let mut arms = Vec::new();
            for (arm, (scope, guard)) in m.arms.iter().zip(guards) {
                let (attrs, pat) = (&arm.attrs, &arm.pat);
                let guard = guard.flatten().map(|g| emit!(if #g));
                let body = scope.source_expr(&arm.body, hint)?;
                arms.push(emit!(#(#attrs)* #pat #guard => #body,));
            }
            return Ok(emit!(match #value { #(#arms)* }));
        }
        let selection = self.fresh_label("source_arms");
        let mut chain = Vec::new();
        for (arm, (scope, guard)) in m.arms.iter().zip(guards) {
            let (attrs, pat) = (&arm.attrs, &arm.pat);
            let scope = scope.nested_region();
            let body = scope.isolated_expr(&arm.body, hint)?;
            chain.push(match guard {
                Some(None) => {
                    let guard =
                        scope.isolated_expr(&arm.guard.as_ref().unwrap().1, Some(Scalar::Bool))?;
                    emit!(#[allow(unreachable_patterns)] match __splice_subject {
                        #(#attrs)* #pat => arms.push((::core::option::Option::Some(#guard), #body)),
                        _ => {}
                    })
                }
                guard => {
                    let guard = guard.flatten().map(|g| emit!(if #g));
                    emit!(#[allow(unreachable_patterns)] match __splice_subject {
                        #(#attrs)* #pat #guard => {
                            arms.push((::core::option::Option::None, #body));
                            break #selection;
                        }
                        _ => {}
                    })
                }
            });
        }
        Ok(emit!({
            let __splice_subject = #value;
            let mut arms = ::std::vec::Vec::new();
            #selection: { #(#chain)* }
            #root::frontend::source::guarded_arms(&arms)
        }))
    }

    fn isolated_expr(&self, expr: &Expr, hint: Option<Scalar>) -> Result<TokenStream> {
        Ok(isolate(self.source_operand(expr, hint)?))
    }

    fn source_operand(&self, expr: &Expr, hint: Option<Scalar>) -> Result<TokenStream> {
        if self.region_depth == 0 {
            return self.source_expr(expr, hint);
        }
        // A fault in this operand must preserve earlier operands in the listing.
        let mut scope = self.clone();
        let label = self.fresh_label("source_expr");
        scope.source_fault = Some(label.clone());
        let code = scope.source_expr(expr, hint)?;
        Ok(emit!({ #[allow(unused_labels)] #label: { #code } }))
    }

    fn source_syntax(&self, e: &Expr) -> Result<TokenStream> {
        // Analyze the original child nodes: flow facts are keyed by their addresses.
        #[derive(Default)]
        struct Children<'a>(Vec<&'a Expr>);
        impl<'a> Visit<'a> for Children<'a> {
            fn visit_expr(&mut self, e: &'a Expr) {
                self.0.push(e);
            }
            fn visit_expr_call(&mut self, call: &'a syn::ExprCall) {
                self.0.extend(&call.args);
            }
        }
        let mut children = Children::default();
        visit::visit_expr(&mut children, e);
        let original = e.to_token_stream().to_string();
        if children.0.is_empty() {
            return Ok(emit!(::std::borrow::ToOwned::to_owned(#original)));
        }
        // A marker absent even from literal text cannot collide with a user's path.
        let mut marker = "__splice_source_child_".to_owned();
        while original.contains(&marker) {
            marker.push('_');
        }
        struct Mark(Ident);
        impl VisitMut for Mark {
            fn visit_field_value_mut(&mut self, field: &mut syn::FieldValue) {
                field.colon_token = Some(Default::default());
                self.visit_expr_mut(&mut field.expr);
            }
            fn visit_expr_mut(&mut self, expr: &mut Expr) {
                let id = &self.0;
                *expr = syn::parse_quote!(#id);
            }

            fn visit_expr_call_mut(&mut self, call: &mut syn::ExprCall) {
                for arg in &mut call.args {
                    self.visit_expr_mut(arg);
                }
            }
        }
        let mut template = e.clone();
        visit_mut::visit_expr_mut(&mut Mark(format_ident!("{marker}")), &mut template);
        let format = template
            .to_token_stream()
            .to_string()
            .replace('{', "{{")
            .replace('}', "}}")
            .replace(&marker, "{}");
        let args = children
            .0
            .into_iter()
            .map(|expr| self.source_operand(expr, None))
            .collect::<Result<Vec<_>>>()?;
        if self.region_depth == 0 {
            return Ok(emit!(::std::format!(#format, #(#args),*)));
        }
        let label = self.fresh_label("source_operands");
        let root = self.root;
        let indices = 0..args.len();
        Ok(emit!(#label: {
            let mut operands = ::std::vec::Vec::new();
            #(
                operands.push(#args);
                if __splice_source.transfer.is_some() {
                    break #label (#root::frontend::source::block(&operands.join(";\n")));
                }
            )*
            ::std::format!(#format, #(operands[#indices]),*)
        }))
    }
}

fn isolate(body: TokenStream) -> TokenStream {
    emit!({ let transfer = __splice_source.transfer.take();
        let text = #body;
        __splice_source.transfer = transfer;
        text
    })
}

/// The `'label: ` prefix of a printed loop.
fn label_text(label: Option<&syn::Label>) -> String {
    label.map_or_else(String::new, |label| format!("{}: ", label.name))
}
