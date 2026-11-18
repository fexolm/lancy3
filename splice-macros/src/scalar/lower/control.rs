//! Lower staged branches, matches, and loops into scalar control flow.
use super::{Binding, Lower, PatternNames, Scalar, Stage, Value, err};
use crate::scalar::{Native, private};
use proc_macro2::{Ident, TokenStream};
use quote::{ToTokens, quote};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{Block, Error, Expr, Lit, Pat, Result};

impl Lower<'_> {
    pub(super) fn known_loop(&self, expr: &Expr) -> Result<Value> {
        let mut node = expr.clone();
        Native.visit_expr_mut(&mut node);
        let code = self.host_raw(&node)?;
        Ok(Value::unit(code))
    }
    // Join two generated Rust arms of a host conditional. A terminating arm
    // can hand back an unreachable register for the surrounding build code.
    fn host_arm(&self, v: Value, to: Stage, at: impl ToTokens) -> Result<TokenStream> {
        let b = self.builder;
        if v.stage == Stage::Never {
            let code = v.code;
            return match to {
                Stage::Reg(_) => Ok(quote!({#code; #b.local()})),
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
            Value::unit(quote!({}))
        };
        self.choice(cond, yes, no, i)
    }
    fn runtime_join(stage: Stage, at: impl ToTokens) -> Result<Stage> {
        match stage {
            Stage::Buffer(_) => err(
                at,
                "a runtime branch can select a slice element; slice arguments must be fixed when the kernel is built",
            ),
            Stage::Literal(t) => Ok(Stage::Reg(t)),
            Stage::Host => Ok(Stage::Reg(Scalar::Inferred)),
            _ => Ok(stage),
        }
    }
    fn write_arm(
        &self,
        v: Value,
        stage: Stage,
        dst: &Ident,
        end: &Ident,
        at: impl ToTokens,
    ) -> Result<TokenStream> {
        let b = self.builder;
        if v.stage == Stage::Never {
            return Ok(v.code);
        }
        if let Stage::Reg(t) = stage {
            let value = self.reg(v, t, at)?;
            Ok(quote!({let value=#value; #b.assign(#dst,value); #b.jump(#end);}))
        } else {
            let code = v.code;
            Ok(quote!({#code; #b.jump(#end);}))
        }
    }
    pub(super) fn choice(
        &self,
        cond: Value,
        yes: Value,
        no: Value,
        at: impl ToTokens,
    ) -> Result<Value> {
        let stage = self.stage_join(yes.stage, no.stage, &at)?;
        if matches!(cond.stage, Stage::Literal(Scalar::Bool) | Stage::Host) {
            let stage = if matches!(yes.stage, Stage::Never) || matches!(no.stage, Stage::Never) {
                match stage {
                    Stage::Literal(t) => Stage::Reg(t),
                    Stage::Host => Stage::Reg(Scalar::Inferred),
                    _ => stage,
                }
            } else {
                stage
            };
            let test = cond.code;
            let yes = self.host_arm(yes, stage, &at)?;
            let no = self.host_arm(no, stage, &at)?;
            return Ok(Value::new(stage, quote!(if #test {#yes}else{#no})));
        }
        let b = self.builder;
        let test = self.reg(cond, Scalar::Bool, &at)?;
        let stage = Self::runtime_join(stage, &at)?;
        let dst = private("__lancy_choice");
        let end = private("__lancy_join");
        let cond = private("__lancy_cond");
        let yes_label = private("__lancy_yes");
        let no_label = private("__lancy_no");
        let known = private("__lancy_known_condition");
        let inner = self.nested_region();
        let yes_code = inner.delimit(self.write_arm(yes, stage, &dst, &end, &at)?);
        let no_code = inner.delimit(self.write_arm(no, stage, &dst, &end, &at)?);
        let (init, result) = if matches!(stage, Stage::Reg(_)) {
            (quote!(let #dst=#b.local();), quote!(#dst))
        } else {
            (quote!(), quote!())
        };
        Ok(Value::new(
            stage,
            quote!({
                let #cond=#test; #init
                let (#yes_label,#no_label,#end)=(#b.label(),#b.label(),#b.label());
                let #known = #b.immediate_condition(#cond);
                if #known.is_none() { #b.branch(#cond,#yes_label,#no_label); }
                if #known != Some(false) { #b.mark(#yes_label); #yes_code; }
                if #known != Some(true) { #b.mark(#no_label); #no_code; }
                #b.mark(#end); #result
            }),
        ))
    }
    pub(super) fn loop_jump(&self, e: &Expr, again: bool) -> Result<Value> {
        let Some((next, end)) = &self.loop_labels else {
            return err(e, "break/continue outside a kernel loop");
        };
        let target = if again { next } else { end };
        let record = self.unrolled_exits.as_ref().map(|(n, e)| {
            let reached = if again { n } else { e };
            quote!(#reached = true;)
        });
        let b = self.builder;
        let region = self.region();
        Ok(Value::new(
            Stage::Never,
            quote!({#record #b.jump(#target); break #region;}),
        ))
    }
    pub(super) fn loop_expr(&self, cond: Option<&Expr>, body: &Block) -> Result<Value> {
        let b = self.builder;
        let (header, end) = (private("__lancy_loop_head"), private("__lancy_loop_end"));
        let mut scope = self.nested_region();
        scope.unrolled_exits = None;
        scope.loop_labels = Some((header.clone(), end.clone()));
        let test = if let Some(e) = cond {
            let reg = scope.reg(scope.expr(e)?, Scalar::Bool, e)?;
            quote!({let cond=#reg;let enter=#b.label();#b.branch(cond,enter,#end);#b.mark(enter);})
        } else {
            quote!()
        };
        let body_code = scope.block(body)?.code;
        let contents = scope.delimit(quote!(#test #body_code; #b.jump(#header);));
        let code = quote!({
            let (#header,#end)=(#b.label(),#b.label()); #b.jump(#header); #b.mark(#header);
            #contents #b.mark(#end);
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
        let Pat::Ident(pat) = &*f.pat else {
            return err(&f.pat, "expected a loop variable");
        };
        if pat.by_ref.is_some() || pat.subpat.is_some() || pat.mutability.is_some() {
            return err(pat, "expected an immutable loop variable");
        }
        let stop_value = self.expr(stop)?;
        let kind = if stop_value.stage == Stage::Reg(Scalar::Index) {
            // Lengths are checked at the native API boundary. A general usize
            // produced with `as` may instead exceed the register width.
            if !matches!(&**stop, Expr::MethodCall(m) if m.method == "len")
                || !matches!(&**start, Expr::Lit(l) if matches!(&l.lit, Lit::Int(i) if i.base10_parse::<i32>().is_ok()))
            {
                return err(
                    range,
                    "use an i32 range or a literal start and a runtime slice.len() bound",
                );
            }
            Scalar::Index
        } else {
            Scalar::Int
        };
        let initial = self.reg(self.expr(start)?, kind, start)?;
        let bound = self.reg(stop_value, kind, stop)?;
        let (step, end) = (private("__lancy_loop_step"), private("__lancy_loop_end"));
        let start_reg = private("__lancy_start");
        let limit = private("__lancy_limit");
        let head = private("__lancy_for_head");
        let entry = private("__lancy_for_body");
        let one = private("__lancy_one");
        let next = private("__lancy_next");
        let mut scope = self.nested_region();
        scope.unrolled_exits = None;
        scope.loop_labels = Some((step.clone(), end.clone()));
        let id = &pat.ident;
        scope
            .bindings
            .insert(id.to_string(), Binding::Reg(kind, false));
        let body = scope.block(&f.body)?.code;
        let (root, b) = (self.root, self.builder);
        let contents = scope.delimit(quote!(#body; #b.jump(#step);));
        Ok(Value::unit(quote!({
            let (#start_reg,#limit)=(#initial,#bound);
            let #id=#b.local();#b.assign(#id,#start_reg);
            let (#head,#entry,#step,#end)=(#b.label(),#b.label(),#b.label(),#b.label());
            #b.jump(#head);#b.mark(#head);
            {let cond=#b.binary(#root::hir::Binary::Lt,#id,#limit);
            #b.branch(cond,#entry,#end);}#b.mark(#entry);
            #contents #b.mark(#step);
            let #one=#b.constant(1);let #next=#b.binary(#root::hir::Binary::Add,#id,#one);
            #b.assign(#id,#next);#b.jump(#head);#b.mark(#end);
        })))
    }
    fn unroll(&self, f: &syn::ExprForLoop) -> Result<Value> {
        // The Rust iterator is evaluated once. The pattern (including references
        // and destructuring) exists only while staging this iteration.
        let iterable = self.host_raw(&f.expr)?;
        let pat = &f.pat;
        let b = self.builder;
        let mut bindings = self.clone();
        PatternNames(&mut bindings.bindings).visit_pat(pat);
        if !contains_transfer(&f.body) {
            // A straight-line plan needs no iteration labels or induction
            // variable. Only the operations in its selected steps reach IR.
            let body = bindings.block(&f.body)?.code;
            return Ok(Value::unit(quote!({for #pat in #iterable { #body; }})));
        }
        let next = private("__splice_iteration_end");
        let end = private("__splice_unroll_end");
        let continued = private("__splice_reaches_next");
        let stopped = private("__splice_reaches_end");
        let mut scope = bindings.nested_region();
        scope.loop_labels = Some((next.clone(), end.clone()));
        scope.unrolled_exits = Some((continued.clone(), stopped.clone()));
        let body = scope.block(&f.body)?;
        let fallthrough = (body.stage != Stage::Never).then(|| quote!(#continued = true;));
        let body = body.code;
        let iteration = scope.delimit(quote!(#body; #fallthrough));
        let leave = self.end_region();
        Ok(Value::unit(quote!({
            let #end = #b.label();
            let mut #stopped = false;
            let mut #continued = true;
            for #pat in #iterable {
                let #next = #b.label();
                #continued = false;
                #iteration
                #b.mark(#next);
                if !#continued { break; }
            }
            #b.mark(#end);
            if !#continued && !#stopped { #leave }
        })))
    }
    pub(super) fn match_expr(&self, m: &syn::ExprMatch) -> Result<Value> {
        let test = self.expr(&m.expr)?;
        if matches!(test.stage, Stage::Host | Stage::Literal(_)) {
            let mut lowered = Vec::new();
            let mut stage = Stage::Never;
            let mut has_never = false;
            for arm in &m.arms {
                let mut scope = self.clone();
                PatternNames(&mut scope.bindings).visit_pat(&arm.pat);
                let guard = arm
                    .guard
                    .as_ref()
                    .map(|(_, g)| scope.host_raw(g).map(|g| quote!(if #g)))
                    .transpose()?;
                let v = scope.expr(&arm.body)?;
                has_never |= v.stage == Stage::Never;
                stage = self.stage_join(stage, v.stage, &arm.body)?;
                lowered.push((arm, guard, v));
            }
            if has_never {
                stage = match stage {
                    Stage::Literal(t) => Stage::Reg(t),
                    Stage::Host => Stage::Reg(Scalar::Inferred),
                    other => other,
                };
            }
            let mut arms = Vec::new();
            for (arm, guard, v) in lowered {
                let attrs = &arm.attrs;
                let pat = &arm.pat;
                let body = self.host_arm(v, stage, &arm.body)?;
                arms.push(quote!(#(#attrs)* #pat #guard=>{#body},));
            }
            let code = test.code;
            return Ok(Value::new(stage, quote!(match #code {#(#arms)*})));
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
        let mut stage = Stage::Never;
        let mut arms = Vec::new();
        let scope = self.nested_region();
        for arm in &m.arms {
            let body = scope.expr(&arm.body)?;
            stage = self.stage_join(stage, body.stage, &arm.body)?;
            arms.push((arm, body));
        }
        let stage = Self::runtime_join(stage, m)?;
        let (dst, end, subject) = (
            private("__lancy_match_result"),
            private("__lancy_match_end"),
            private("__lancy_subject"),
        );
        let b = self.builder;
        let mut body = TokenStream::new();
        for (arm, value) in arms {
            if !arm.attrs.is_empty() {
                return err(arm, "attributes on runtime arms are unsupported");
            }
            let after = private("__lancy_next_arm");
            let write = self.write_arm(value, stage, &dst, &end, &arm.body)?;
            let pattern = if matches!(arm.pat, Pat::Wild(_)) {
                quote!()
            } else {
                let pred = self.pattern(&arm.pat, t, &subject)?;
                quote!({let matched=#pred;let hit=#b.label();#b.branch(matched,hit,#after);#b.mark(hit);})
            };
            let guard = if let Some((_, g)) = &arm.guard {
                let v = scope.expr(g)?;
                let test = self.reg(v, Scalar::Bool, g)?;
                quote!({let passed=#test;let hit=#b.label();#b.branch(passed,hit,#after);#b.mark(hit);})
            } else {
                quote!()
            };
            let contents = scope.delimit(quote!(#guard #write;));
            body.extend(quote!({let #after=#b.label();#pattern #contents #b.mark(#after);}));
        }
        let subject_code = test.code;
        let (init, result) = if matches!(stage, Stage::Reg(_)) {
            (quote!(let #dst=#b.local();), quote!(#dst))
        } else {
            (quote!(), quote!())
        };
        Ok(Value::new(
            stage,
            quote!({let #subject=#subject_code;#init let #end=#b.label();#body #b.mark(#end);#result}),
        ))
    }
    fn pattern(&self, p: &Pat, t: Scalar, subject: &Ident) -> Result<TokenStream> {
        let b = self.builder;
        match p {
            Pat::Lit(l) => {
                let lit = match (&l.lit, t) {
                    (Lit::Int(_), Scalar::Int | Scalar::Inferred) => quote!(#b.constant(#l)),
                    (Lit::Bool(_), Scalar::Bool | Scalar::Inferred) => {
                        quote!(#b.constant(-i32::from(#l)))
                    }
                    _ => return err(l, "pattern type differs from match subject"),
                };
                Ok(self.bin("Eq", quote!(#subject), lit))
            }
            Pat::Paren(p) => self.pattern(&p.pat, t, subject),
            Pat::Or(or) => {
                let mut parts = or.cases.iter().map(|p| self.pattern(p, t, subject));
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
                    pred = Some(self.bin("Ge", quote!(#subject), quote!(#b.constant(#v))));
                }
                if let Some(stop) = &r.end {
                    let v = self.host_raw(stop)?;
                    let upper = self.bin(cmp, quote!(#subject), quote!(#b.constant(#v)));
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
