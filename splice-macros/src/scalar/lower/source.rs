use super::{Binding, LoopKind, Lower, Scalar, Stage, Value, err};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens, format_ident};
use syn::visit::{self, Visit};
use syn::visit_mut::{self, VisitMut};
use syn::{Block, Expr, Result, Stmt};

impl Lower<'_> {
    pub(in crate::scalar) fn source_body(&self, block: &Block) -> Result<TokenStream> {
        let mut scope = self.clone();
        let result_type = self.result_type;
        let mut statements = Vec::new();
        let end = syn::Lifetime::new(
            &format!("'__splice_source_{}", crate::scalar::flow::key(block)),
            Span::mixed_site(),
        );
        scope.source_fault = Some(end.clone());
        for (i, stmt) in block.stmts.iter().enumerate() {
            let rendered = match stmt {
                Stmt::Local(local) => {
                    let init = local.init.as_ref().expect("validated local");
                    if let Expr::Struct(literal) = &*init.expr {
                        let value = scope.source_syntax(&init.expr)?;
                        let pat = &local.pat;
                        let binding = scope.aggregate_local(pat, literal)?;
                        statements.push(binding);
                        let line = emit!(format!("let {} = {};", stringify!(#pat), #value));
                        statements.push(
                            emit!({ __splice_text.push_str(&#line); __splice_text.push('\n'); }),
                        );
                        continue;
                    }
                    let hint = match &local.pat {
                        syn::Pat::Type(p) => Scalar::from_type(&p.ty, scope.types),
                        _ => None,
                    };
                    let val = scope.expr_with_hint(&init.expr, hint)?;
                    if val.stage.is_host()
                        && !scope.flow.locals.contains(&crate::scalar::flow::key(local))
                    {
                        let binding = scope.local(local)?;
                        statements.push(binding);
                        continue;
                    }
                    let value = scope.source_expr(&init.expr, hint)?;
                    let pat = &local.pat;
                    scope.local(local)?;
                    emit!(format!("let {} = {};", stringify!(#pat), #value))
                }
                Stmt::Expr(expr, semi) => {
                    let last = i + 1 == block.stmts.len() && semi.is_none();
                    if !last
                        && matches!(
                            expr,
                            Expr::MethodCall(_) | Expr::Call(_) | Expr::Assign(_) | Expr::Binary(_)
                        )
                        && scope.host_only(expr)
                    {
                        let value = scope.expr(expr)?;
                        if matches!(value.stage, Stage::Host | Stage::Unit) {
                            // Use the same checked host operation as the executable factory.
                            statements.push(emit!(#value;));
                            continue;
                        }
                    }
                    let rendered = scope.source_expr(expr, last.then_some(result_type))?;
                    if last {
                        rendered
                    } else {
                        emit!({ let text = #rendered; if text.is_empty() { text } else { format!("{text};") } })
                    }
                }
                Stmt::Item(item @ syn::Item::Use(_)) => {
                    statements.push(emit!(#[allow(unused_imports)] #item));
                    emit!(stringify!(#item).to_owned())
                }
                _ => return err(stmt, "unsupported statement in Rust source output"),
            };
            statements.push(emit!({ let text = #rendered; if !text.is_empty() {
                __splice_text.push_str(&text); __splice_text.push('\n');
            } if __splice_source.transfer.is_some() { break #end String::new(); }}));
            if let Stmt::Expr(Expr::Return(_) | Expr::Break(_) | Expr::Continue(_), _) = stmt {
                break;
            }
        }
        Ok(emit!({ let mut __splice_text = String::new();
                #[allow(unused_labels)] let fault = #end: { #(#statements)* String::new() };
                __splice_text.push_str(&fault); __splice_text }))
    }

    fn source_expr(&self, e: &Expr, hint: Option<Scalar>) -> Result<TokenStream> {
        let root = self.root;
        let support = emit!(#root::frontend::source);
        let braces = |body: TokenStream| emit!(#support::block(&#body));
        let loop_body = match e {
            Expr::ForLoop(f) => Some(&f.body),
            Expr::While(w) => Some(&w.body),
            Expr::Loop(l) => Some(&l.body),
            _ => None,
        };
        if let Some(body) = loop_body
            && let Some(code) = self.host_loop(e, body)?
        {
            return Ok(emit!({ #code; String::new() }));
        }
        match e {
            Expr::Block(b) => return Ok(braces(self.source_body(&b.block)?)),
            Expr::If(i) => {
                let (cond, scope) = self.condition(&i.cond)?;
                let yes = braces(scope.source_body(&i.then_branch)?);
                let no = i
                    .else_branch
                    .as_ref()
                    .map(|(_, e)| {
                        if cond.stage.is_host() {
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
                if cond.stage.is_host() {
                    let no = no.unwrap_or_else(|| emit!("{}".to_owned()));
                    return Ok(emit!(if #cond { #yes } else { #no }));
                }
                let cond = self.source_expr(&i.cond, Some(Scalar::Bool))?;
                let yes = isolate(yes);
                return Ok(match no {
                    Some(no) => emit!(format!("if {} {} else {}", #cond, #yes, #no)),
                    None => emit!(format!("if {} {}", #cond, #yes)),
                });
            }
            Expr::Match(m) => {
                let value = self.expr(&m.expr)?;
                if value.stage.is_host() {
                    let mut arms = Vec::new();
                    for arm in &m.arms {
                        let pat = &arm.pat;
                        let attrs = &arm.attrs;
                        let scope = self.with_pattern(pat);
                        let guard = arm
                            .guard
                            .as_ref()
                            .map(|(_, g)| scope.host_raw(g))
                            .transpose()?;
                        let guard = guard.map(|g| emit!(if #g));
                        let body = scope.source_expr(&arm.body, hint)?;
                        arms.push(emit!(#(#attrs)* #pat #guard => #body,));
                    }
                    return Ok(emit!(match #value { #(#arms)* }));
                }
                let test = self.source_expr(&m.expr, None)?;
                let scope = self.nested_region();
                let mut arms = Vec::new();
                for arm in &m.arms {
                    let pat = &arm.pat;
                    let body = scope.isolated_expr(&arm.body, hint)?;
                    let guard = if let Some((_, g)) = &arm.guard {
                        let guard = scope.isolated_expr(g, Some(Scalar::Bool))?;
                        emit!(format!(" if {}", #guard))
                    } else {
                        emit!(String::new())
                    };
                    arms.push(emit!(
                        format!("{}{} => {},", stringify!(#pat), #guard, #body)
                    ));
                }
                return Ok(emit!(
                    format!("match {} {}", #test, #support::block(&[#(#arms),*].join("\n")))
                ));
            }
            Expr::ForLoop(f) => {
                let pat = &f.pat;
                if !matches!(&*f.expr, Expr::Range(_)) && self.iterated_buffer(&f.expr).is_none() {
                    let iter = self.plan_iterable(&f.expr)?;
                    let mut scope = self.with_pattern(pat);
                    scope.loop_kind = Some(LoopKind::Unrolled);
                    let body = scope.source_body(&f.body)?;
                    return Ok(emit!({ let mut text = String::new();
                        for #pat in #iter {
                            #root::frontend::scalar::specialization_step()?;
                            let body = #body;
                            let block = #support::block(&body);
                            if body.contains("break '__splice_iteration") { text.push_str("'__splice_iteration: "); }
                            text.push_str(&block); text.push_str(";\n");
                            match __splice_source.transfer {
                                Some(#support::Transfer::Break) => { __splice_source.transfer = None; break; }
                                Some(#support::Transfer::Continue) => __splice_source.transfer = None,
                                Some(#support::Transfer::Return) => break,
                                None => {}
                            }
                        }
                        if text.contains("break '__splice_unrolled") {
                            format!("'__splice_unrolled: {}", #support::block(&text))
                        } else { #support::block(&text) }
                    }));
                }
                let mut scope = self.nested_region();
                scope.loop_kind = Some(LoopKind::Runtime);
                crate::scalar::bind_pattern(
                    &mut scope.bindings,
                    pat,
                    Binding::Reg(Scalar::Int, false),
                );
                let range = self.source_syntax(&f.expr)?;
                let body = isolate(braces(scope.source_body(&f.body)?));
                return Ok(emit!(
                    format!("for {} in {} {}", stringify!(#pat), #range, #body)
                ));
            }
            Expr::While(w) => {
                let mut scope = self.nested_region();
                let cond = scope.isolated_expr(&w.cond, Some(Scalar::Bool))?;
                scope.loop_kind = Some(LoopKind::Runtime);
                let body = isolate(braces(scope.source_body(&w.body)?));
                return Ok(emit!(format!("while {} {}", #cond, #body)));
            }
            Expr::Loop(l) => {
                let mut scope = self.nested_region();
                scope.loop_kind = Some(LoopKind::Runtime);
                let body = isolate(braces(scope.source_body(&l.body)?));
                return Ok(emit!(format!("loop {}", #body)));
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
                    let slot = self.host(self.expr(&index.index)?, &index.index)?;
                    let element = elem.tokens(root, self.types);
                    let ty = elem.ty(root, self.types);
                    return Ok(emit!({
                        let slot = #slot;
                        #root::frontend::scalar::column((__splice_column_layout, #root::hir::Buffer(0)), slot, #element)?;
                        format!("{}[{}].as_slice::<{}>()", stringify!(#table), slot, stringify!(#ty))
                    }));
                }
            }
            Expr::Call(call) => {
                let Expr::Path(p) = &*call.func else {
                    return self.source_syntax(e);
                };
                if crate::scalar::methods::method_path(p) {
                    return self.source_syntax(e);
                }
                if super::simd::constructor_arity(&p.path).is_some() {
                    return self.source_syntax(e);
                }
                let args = call
                    .args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<Result<Vec<_>>>()?;
                if !args.iter().all(|v| v.stage.is_host()) {
                    let mut path = p.path.clone();
                    let fish = std::mem::take(&mut path.segments.last_mut().unwrap().arguments);
                    let args = call
                        .args
                        .iter()
                        .zip(args)
                        .map(|(arg, value)| {
                            if value.stage.is_host() {
                                Ok(emit!(#root::frontend::scalar::Host(#value)))
                            } else {
                                self.source_expr(arg, None)
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let kernel =
                        emit!(#path::__splice_source_call #fish (#(#args,)* __splice_source)?);
                    return if p.qself.is_none() && !p.path.is_ident(self.name) {
                        Ok(self.dispatch_call(p, kernel, self.source_syntax(e)?))
                    } else {
                        Ok(kernel)
                    };
                }
            }
            Expr::Return(_) | Expr::Break(_) | Expr::Continue(_) => {
                let (transfer, label) = match e {
                    Expr::Return(_) => (emit!(Return), None),
                    Expr::Break(_) => (emit!(Break), Some("break '__splice_unrolled")),
                    _ => (emit!(Continue), Some("break '__splice_iteration")),
                };
                let text = if let Some(label) = label
                    && matches!(self.loop_kind, Some(LoopKind::Unrolled))
                {
                    emit!(#label.to_owned())
                } else {
                    self.source_syntax(e)?
                };
                return Ok(
                    emit!({ let text = #text; __splice_source.transfer = Some(#support::Transfer::#transfer); text }),
                );
            }
            Expr::Binary(binary) if matches!(binary.op, syn::BinOp::And(_) | syn::BinOp::Or(_)) => {
                let lhs = self.expr(&binary.left)?;
                if lhs.stage.is_host() {
                    let rhs = self.source_expr(&binary.right, Some(Scalar::Bool))?;
                    return Ok(if matches!(binary.op, syn::BinOp::And(_)) {
                        emit!(if #lhs { #rhs } else { "false".to_owned() })
                    } else {
                        emit!(if #lhs { "true".to_owned() } else { #rhs })
                    });
                }
                let lhs = self.source_expr(&binary.left, Some(Scalar::Bool))?;
                let rhs = self
                    .nested_region()
                    .isolated_expr(&binary.right, Some(Scalar::Bool))?;
                let op = &binary.op;
                return Ok(emit!(format!("{} {} {}", #lhs, stringify!(#op), #rhs)));
            }
            Expr::Range(_) => {
                return self.source_syntax(e);
            }
            _ => {}
        }
        if let Ok(Value {
            stage: Stage::Host,
            code,
        }) = self.expr_with_hint(e, hint)
        {
            return Ok(emit!({
                use #support::TryLiteral as _;
                let value = &(#code);
                (&#support::Literal(value)).try_literal()?
            }));
        }
        self.source_syntax(e)
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
        let label = syn::Lifetime::new(
            &format!("'__splice_source_expr_{}", crate::scalar::flow::key(expr)),
            Span::mixed_site(),
        );
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
            return Ok(emit!(#original.to_owned()));
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
            return Ok(emit!(format!(#format, #(#args),*)));
        }
        let label = syn::Lifetime::new(
            &format!("'__splice_source_operands_{}", crate::scalar::flow::key(e)),
            Span::mixed_site(),
        );
        let root = self.root;
        let indices = 0..args.len();
        Ok(emit!(#label: {
            let mut operands = Vec::new();
            #(
                operands.push(#args);
                if __splice_source.transfer.is_some() {
                    break #label (#root::frontend::source::block(&operands.join(";\n")));
                }
            )*
            format!(#format, #(operands[#indices]),*)
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
