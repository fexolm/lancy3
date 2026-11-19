//! Lower staged branches, matches, and loops into scalar control flow.
use super::{Binding, LoopKind, Lower, Scalar, Stage, Value, err};
use crate::scalar::Native;
use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{Block, Error, Expr, Lit, Pat, Result};

impl Lower<'_> {
    pub(super) fn staged_loop(
        &self,
        expr: &Expr,
        body: &Block,
        runtime: impl FnOnce() -> Result<Value>,
    ) -> Result<Value> {
        if self.flow.loops.contains(&crate::scalar::flow::key(body)) {
            return runtime();
        }
        let mut node = expr.clone();
        Native.visit_expr_mut(&mut node);
        let code = self.host_raw(&node)?;
        Ok(Value::unit(code))
    }
    // Join two generated Rust arms of a host conditional. A terminating arm
    // can hand back an unreachable register for the surrounding build code.
    fn host_arm(&self, v: Value, to: Stage, at: impl ToTokens) -> Result<TokenStream> {
        if v.stage == Stage::Never {
            let code = v.code;
            return match to {
                Stage::Reg(t) => {
                    let ty = t.value_type(self.root);
                    Ok(emit!({#code; __splice_builder.typed_local(#ty)}))
                }
                Stage::Never | Stage::Unit => Ok(code),
                _ => err(at, "an early kernel return must merge with a scalar result"),
            };
        }
        match to {
            Stage::Reg(t) => self.reg(v, t, at),
            _ => Ok(v.code),
        }
    }
    pub(super) fn if_expr(&self, i: &syn::ExprIf) -> Result<Value> {
        let cond = self.expr(&i.cond)?;
        let scope = if matches!(cond.stage, Stage::Reg(_)) {
            self.nested_region()
        } else {
            self.clone()
        };
        let yes = scope.block(&i.then_branch)?;
        let no = if let Some((_, no)) = &i.else_branch {
            scope.expr(no)?
        } else {
            Value::unit(emit!({}))
        };
        self.choice(cond, yes, no, i)
    }
    fn join<'a>(
        &self,
        arms: impl IntoIterator<Item = &'a Value>,
        runtime: bool,
        at: &impl ToTokens,
    ) -> Result<Stage> {
        let mut stage = Stage::Never;
        let mut needs_register = runtime;
        for arm in arms {
            stage = self.stage_join(stage, arm.stage, at)?;
            needs_register |= arm.stage == Stage::Never;
        }
        if runtime && matches!(stage, Stage::Buffer(_)) {
            return err(
                at,
                "a runtime branch can select a slice element; slice arguments must be fixed when the kernel is built",
            );
        }
        Ok(if needs_register {
            stage.materialized()
        } else {
            stage
        })
    }
    fn write_arm(&self, v: Value, stage: Stage, at: impl ToTokens) -> Result<TokenStream> {
        if v.stage == Stage::Never {
            return Ok(v.code);
        }
        if let Stage::Reg(t) = stage {
            let value = self.reg(v, t, at)?;
            Ok(emit!({
                let __splice_value = #value;
                __splice_builder.assign(__splice_branch_result, __splice_value);
                __splice_builder.jump(__splice_join);
            }))
        } else {
            Ok(emit!({#v; __splice_builder.jump(__splice_join);}))
        }
    }
    pub(super) fn choice(
        &self,
        cond: Value,
        yes: Value,
        no: Value,
        at: impl ToTokens,
    ) -> Result<Value> {
        let host = matches!(cond.stage, Stage::Literal(Scalar::Bool) | Stage::Host);
        let stage = self.join([&yes, &no], !host, &at)?;
        if host {
            let yes = self.host_arm(yes, stage, &at)?;
            let no = self.host_arm(no, stage, &at)?;
            return Ok(Value::new(stage, emit!(if #cond {#yes}else{#no})));
        }

        let test = self.reg(cond, Scalar::Bool, &at)?;
        let inner = self.nested_region();
        let yes_code = inner.delimit(self.write_arm(yes, stage, &at)?);
        let no_code = inner.delimit(self.write_arm(no, stage, &at)?);
        let (init, result) = result_slot(stage, self.root);
        Ok(Value::new(
            stage,
            emit!({
                let __splice_cond = #test;
                #init
                let (__splice_yes, __splice_no, __splice_join) =
                    (__splice_builder.label(), __splice_builder.label(), __splice_builder.label());
                let __splice_known_condition = __splice_builder.immediate_condition(__splice_cond);
                if __splice_known_condition.is_none() { __splice_builder.branch(__splice_cond, __splice_yes, __splice_no); }
                if __splice_known_condition != Some(false) { __splice_builder.mark(__splice_yes); #yes_code; }
                if __splice_known_condition != Some(true) { __splice_builder.mark(__splice_no); #no_code; }
                __splice_builder.mark(__splice_join); #result
            }),
        ))
    }
    pub(super) fn loop_jump(&self, e: &Expr, again: bool) -> Result<Value> {
        let Some(kind) = self.loop_kind else {
            return err(e, "break/continue outside a kernel loop");
        };
        let (target, flag) = if again {
            (emit!(__splice_loop_next), emit!(__splice_reaches_next))
        } else {
            (emit!(__splice_loop_end), emit!(__splice_reaches_end))
        };
        let record = matches!(kind, LoopKind::Unrolled).then(|| emit!(#flag = true;));
        let region = self.region();
        Ok(Value::new(
            Stage::Never,
            emit!({#record __splice_builder.jump(#target); break #region;}),
        ))
    }
    pub(super) fn loop_expr(&self, cond: Option<&Expr>, body: &Block) -> Result<Value> {
        let mut scope = self.nested_region();
        scope.loop_kind = Some(LoopKind::Runtime);
        let test = if let Some(e) = cond {
            let reg = scope.operand(e, Scalar::Bool)?;
            emit!({let __splice_cond = #reg; __splice_builder.branch_if_false(__splice_cond, __splice_loop_end);})
        } else {
            emit!()
        };
        let body_code = scope.block(body)?.code;
        let contents =
            scope.delimit(emit!(#test #body_code; __splice_builder.jump(__splice_loop_next);));
        let code = emit!({
            let (__splice_loop_next, __splice_loop_end) = (__splice_builder.label(), __splice_builder.label());
            __splice_builder.jump(__splice_loop_next);
            __splice_builder.mark(__splice_loop_next);
            #contents
            __splice_builder.mark(__splice_loop_end);
        });
        let stage = if cond.is_none() && !contains_break(body) {
            Stage::Never
        } else {
            Stage::Unit
        };
        Ok(Value::new(stage, code))
    }
    pub(super) fn for_expr(&self, f: &syn::ExprForLoop) -> Result<Value> {
        let Expr::Range(range) = &*f.expr else {
            return self.unroll(f);
        };
        let (Some(start), Some(stop)) = (&range.start, &range.end) else {
            return err(range, "expected start..end");
        };
        if !matches!(range.limits, syn::RangeLimits::HalfOpen(_)) {
            return err(range, "kernel for loops use an exclusive upper bound");
        }
        let pat = crate::scalar::named(&f.pat)?;
        if pat.mutability.is_some() {
            return err(pat, "expected an immutable loop variable");
        }
        let stop_value = self.expr(stop)?;
        let kind = if matches!(stop_value.stage, Stage::Reg(Scalar::Index | Scalar::Length)) {
            // This loop uses a signed i32 induction variable. Both endpoints must fit.
            if stop_value.stage != Stage::Reg(Scalar::Length)
                || !matches!(&**start, Expr::Lit(l) if matches!(&l.lit, Lit::Int(i) if i.base10_parse::<i32>().is_ok()))
            {
                return err(
                    range,
                    "use an i32 range or a literal start with a slice.len() or Lane::lanes() bound",
                );
            }
            Scalar::Length
        } else {
            Scalar::Int
        };
        let initial = self.operand(start, kind)?;
        let bound = self.reg(stop_value, kind, stop)?;
        let mut scope = self.nested_region();
        scope.loop_kind = Some(LoopKind::Runtime);
        let id = &pat.ident;
        scope
            .bindings
            .insert(id.to_string(), Binding::Reg(kind, false));
        let body = scope.block(&f.body)?.code;
        let root = self.root;
        let contents = scope.delimit(emit!(#body; __splice_builder.jump(__splice_loop_next);));
        Ok(Value::unit(emit!({
            let (__splice_start, __splice_limit) = (#initial, #bound);
            let #id = __splice_builder.copy(__splice_start);
            let (__splice_for_head, __splice_for_body, __splice_loop_next, __splice_loop_end) =
                (__splice_builder.label(), __splice_builder.label(), __splice_builder.label(), __splice_builder.label());
            __splice_builder.jump(__splice_for_head);
            __splice_builder.mark(__splice_for_head);
            let __splice_cond = __splice_builder.binary(#root::hir::Binary::Lt, #id, __splice_limit);
            __splice_builder.branch(__splice_cond, __splice_for_body, __splice_loop_end);
            __splice_builder.mark(__splice_for_body);
            #contents
            __splice_builder.mark(__splice_loop_next);
            let __splice_one = __splice_builder.constant(1);
            let __splice_next = __splice_builder.binary(#root::hir::Binary::Add, #id, __splice_one);
            __splice_builder.assign(#id, __splice_next);
            __splice_builder.jump(__splice_for_head);
            __splice_builder.mark(__splice_loop_end);
        })))
    }
    fn unroll(&self, f: &syn::ExprForLoop) -> Result<Value> {
        // The Rust iterator is evaluated once. The pattern (including references
        // and destructuring) exists only while staging this iteration.
        let iterable = self.host_raw(&f.expr)?;
        let pat = &f.pat;

        let bindings = self.with_pattern(pat);
        if !contains_transfer(&f.body) {
            // A straight-line plan needs no iteration labels or induction
            // variable. Only the operations in its selected steps reach IR.
            let body = bindings.block(&f.body)?;
            return Ok(Value::unit(emit!({for #pat in #iterable { #body; }})));
        }
        let mut scope = bindings.nested_region();
        scope.loop_kind = Some(LoopKind::Unrolled);
        let body = scope.block(&f.body)?;
        let fallthrough =
            (body.stage != Stage::Never).then(|| emit!(__splice_reaches_next = true;));
        let iteration = scope.delimit(emit!(#body; #fallthrough));
        let region = self.region();
        Ok(Value::unit(emit!({
            let __splice_loop_end = __splice_builder.label();
            let mut __splice_reaches_end = false;
            let mut __splice_reaches_next = true;
            for #pat in #iterable {
                let __splice_loop_next = __splice_builder.label();
                __splice_reaches_next = false;
                #iteration
                __splice_builder.mark(__splice_loop_next);
                if !__splice_reaches_next { break; }
            }
            __splice_builder.mark(__splice_loop_end);
            if !__splice_reaches_next && !__splice_reaches_end { break #region; }
        })))
    }
    pub(super) fn match_expr(&self, m: &syn::ExprMatch) -> Result<Value> {
        let test = self.expr(&m.expr)?;
        let host = test.stage.is_host();
        let scope = if host {
            self.clone()
        } else {
            self.nested_region()
        };
        let mut arms = Vec::new();
        for arm in &m.arms {
            let pattern_scope = host.then(|| self.with_pattern(&arm.pat));
            let scope = pattern_scope.as_ref().unwrap_or(&scope);
            let value = scope.expr(&arm.body)?;
            let guard = match &arm.guard {
                Some((_, g)) if host => {
                    let test = scope.host_raw(g)?;
                    emit!(if #test)
                }
                Some((_, g)) => {
                    let test = scope.operand(g, Scalar::Bool)?;
                    emit!({let passed = #test; __splice_builder.branch_if_false(passed, __splice_next_arm);})
                }
                None => emit!(),
            };
            arms.push((arm, guard, value));
        }
        let stage = self.join(arms.iter().map(|(_, _, v)| v), !host, &m)?;
        if host {
            let mut cases = TokenStream::new();
            for (arm, guard, value) in arms {
                let attrs = &arm.attrs;
                let pat = &arm.pat;
                let body = self.host_arm(value, stage, &arm.body)?;
                cases.extend(emit!(#(#attrs)* #pat #guard => {#body},));
            }
            return Ok(Value::new(stage, emit!(match #test {#cases})));
        }
        let Stage::Reg(t) = test.stage else {
            return err(&m.expr, "expected a value in match");
        };
        if !m
            .arms
            .last()
            .is_some_and(|a| matches!(a.pat, Pat::Wild(_)) && a.guard.is_none())
        {
            return err(m, "a runtime match requires a final unguarded _ arm");
        }
        let mut body = TokenStream::new();
        for (arm, guard, value) in arms {
            if !arm.attrs.is_empty() {
                return err(arm, "attributes on runtime arms are unsupported");
            }
            let write = self.write_arm(value, stage, &arm.body)?;
            let pattern = if matches!(arm.pat, Pat::Wild(_)) {
                emit!()
            } else {
                let pred = self.pattern(&arm.pat, t)?;
                emit!({let matched = #pred; __splice_builder.branch_if_false(matched, __splice_next_arm);})
            };
            let contents = scope.delimit(emit!(#guard #write;));
            body.extend(emit!({
                let __splice_next_arm = __splice_builder.label();
                #pattern #contents
                __splice_builder.mark(__splice_next_arm);
            }));
        }
        let (init, result) = result_slot(stage, self.root);
        Ok(Value::new(
            stage,
            emit!({
                let __splice_subject = #test;
                #init
                let __splice_join = __splice_builder.label();
                #body
                __splice_builder.mark(__splice_join);
                #result
            }),
        ))
    }
    fn pattern(&self, p: &Pat, t: Scalar) -> Result<TokenStream> {
        match p {
            Pat::Lit(l) => {
                let lit = match (&l.lit, t) {
                    (Lit::Int(_), Scalar::Int | Scalar::Inferred) => {
                        emit!(__splice_builder.constant(#l))
                    }
                    (Lit::Bool(_), Scalar::Bool | Scalar::Inferred) => {
                        emit!(__splice_builder.constant(-i32::from(#l)))
                    }
                    _ => return err(l, "pattern type differs from match subject"),
                };
                Ok(self.bin("Eq", emit!(__splice_subject), lit))
            }
            Pat::Paren(p) => self.pattern(&p.pat, t),
            Pat::Or(or) => {
                let mut parts = or.cases.iter().map(|p| self.pattern(p, t));
                let mut pred = parts.next().expect("or pattern")?;
                for part in parts {
                    pred = self.bin("Or", pred, part?)
                }
                Ok(pred)
            }
            Pat::Range(r) if t != Scalar::Bool => {
                let cmp = if matches!(r.limits, syn::RangeLimits::Closed(_)) {
                    "Le"
                } else {
                    "Lt"
                };
                let mut pred = None;
                if let Some(start) = &r.start {
                    let v = self.host_raw(start)?;
                    pred = Some(self.bin(
                        "Ge",
                        emit!(__splice_subject),
                        emit!(__splice_builder.constant(#v)),
                    ));
                }
                if let Some(stop) = &r.end {
                    let v = self.host_raw(stop)?;
                    let upper = self.bin(
                        cmp,
                        emit!(__splice_subject),
                        emit!(__splice_builder.constant(#v)),
                    );
                    pred = Some(match pred {
                        Some(lower) => self.bin("And", lower, upper),
                        None => upper,
                    })
                }
                pred.ok_or_else(|| Error::new_spanned(r, "expected a range endpoint"))
            }
            _ => err(
                p,
                "use a literal, a range, an or-pattern, or a final _ in a runtime match",
            ),
        }
    }
}

fn result_slot(stage: Stage, root: &syn::Path) -> (TokenStream, TokenStream) {
    if let Stage::Reg(t) = stage {
        let ty = t.value_type(root);
        (
            emit!(let __splice_branch_result = __splice_builder.typed_local(#ty);),
            emit!(__splice_branch_result),
        )
    } else {
        (emit!(), emit!())
    }
}

fn contains_transfer(body: &Block) -> bool {
    struct Transfers(bool);
    impl<'ast> Visit<'ast> for Transfers {
        fn visit_expr(&mut self, expr: &'ast Expr) {
            if matches!(
                expr,
                Expr::Break(_) | Expr::Continue(_) | Expr::Return(_) | Expr::Loop(_)
            ) {
                self.0 = true;
            } else if !self.0 {
                syn::visit::visit_expr(self, expr);
            }
        }
    }
    let mut visitor = Transfers(false);
    visitor.visit_block(body);
    visitor.0
}

fn contains_break(body: &Block) -> bool {
    struct Find(bool);
    impl<'ast> Visit<'ast> for Find {
        fn visit_expr_break(&mut self, _: &'ast syn::ExprBreak) {
            self.0 = true;
        }
        fn visit_expr_while(&mut self, _: &'ast syn::ExprWhile) {}
        fn visit_expr_for_loop(&mut self, _: &'ast syn::ExprForLoop) {}
        fn visit_expr_loop(&mut self, _: &'ast syn::ExprLoop) {}
    }
    let mut find = Find(false);
    find.visit_block(body);
    find.0
}
