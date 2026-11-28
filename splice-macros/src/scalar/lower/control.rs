//! Lower staged branches, matches, and loops into scalar control flow.
use super::{Binding, LoopFrame, LoopKind, Lower, Scalar, Stage, Value, err};
use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Block, Expr, Lit, Pat, Result};

impl Lower<'_> {
    pub(super) fn staged_loop(
        &self,
        expr: &Expr,
        runtime: impl FnOnce() -> Result<Value>,
    ) -> Result<Value> {
        if let Some(code) = self.host_loop(expr)? {
            // A Rust `loop` has the value of its `break`.
            let stage = if matches!(expr, Expr::Loop(_)) {
                Stage::Host
            } else {
                Stage::Unit
            };
            return Ok(Value::new(stage, code));
        }
        runtime()
    }
    pub(super) fn host_loop(&self, expr: &Expr) -> Result<Option<TokenStream>> {
        if self.body.staged_loop(expr) {
            return Ok(None);
        }
        self.host_code(expr).map(Some)
    }
    // Join two generated Rust arms of a host conditional. A terminating arm
    // can hand back an unreachable register for the surrounding build code.
    fn host_arm(&self, v: Value, to: Stage, at: impl ToTokens) -> Result<TokenStream> {
        if v.stage == Stage::Never {
            let code = v.code;
            return match to {
                Stage::Reg(t) => {
                    let ty = t.value_type(self.root, self.types);
                    Ok(emit!({#code; __splice_builder.local(#ty)?}))
                }
                Stage::Never | Stage::Unit => Ok(code),
                _ => err(at, "an early kernel return must merge with a scalar result"),
            };
        }
        match to {
            Stage::Reg(t) => self.reg(v, t, at),
            Stage::Word(_) => self.word(v, at),
            _ => Ok(v.code),
        }
    }
    pub(super) fn condition(&self, expr: &Expr) -> Result<(Value, Self)> {
        if let Expr::Let(binding) = expr {
            let value = self.host(self.expr(&binding.expr)?, &binding.expr)?;
            let pattern = &binding.pat;
            return Ok((
                Value::new(Stage::Host, emit!(let #pattern = #value)),
                self.with_pattern(pattern),
            ));
        }
        let value = self.expr(expr)?;
        let then_scope = if value.stage.is_host() {
            self.clone()
        } else {
            self.nested_region()
        };
        Ok((value, then_scope))
    }

    pub(super) fn if_expr(&self, i: &syn::ExprIf, hint: Option<Scalar>) -> Result<Value> {
        let (cond, then_scope) = self.condition(&i.cond)?;
        let yes = then_scope.block_with_hint(&i.then_branch, hint)?;
        let no = if let Some((_, no)) = &i.else_branch {
            if cond.stage.is_host() {
                self.expr_with_hint(no, hint)?
            } else {
                then_scope.expr_with_hint(no, hint)?
            }
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
            let root = self.root;
            let assign = if t == Scalar::Inferred {
                emit!(#root::frontend::scalar::assign_inferred(&mut __splice_branch_result, __splice_value, __splice_builder)?;)
            } else {
                emit!(__splice_builder.assign(__splice_branch_result, __splice_value)?;)
            };
            Ok(emit!({
                let __splice_value = #value;
                #assign
                __splice_builder.jump(__splice_join)?;
            }))
        } else {
            Ok(emit!({#v; __splice_builder.jump(__splice_join)?;}))
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
        let mut stage = self.join([&yes, &no], !host, &at)?;
        if host {
            // A known condition selects one arm during specialization, so a
            // scalar result is known exactly when that arm is: a Word.
            if let Stage::Reg(t @ (Scalar::I32 | Scalar::Bool)) = stage
                && yes.stage != Stage::Never
                && no.stage != Stage::Never
            {
                stage = Stage::Word(t);
            }
            let yes = self.host_arm(yes, stage, &at)?;
            let no = self.host_arm(no, stage, &at)?;
            return Ok(Value::new(stage, emit!(if #cond {#yes}else{#no})));
        }

        // A mixed-stage condition is a Word: known when its declaration is
        // resolved during specialization, otherwise a register to branch on.
        let root = self.root;
        let (word, known) = if matches!(cond.stage, Stage::Word(_)) {
            let code = cond.code;
            (
                emit!(let __splice_word = #code;),
                emit!(#root::frontend::scalar::RunArg::known(&__splice_word).map(|value| value.bits() != 0)),
            )
        } else {
            let test = self.reg(cond, Scalar::Bool, &at)?;
            (
                emit!(let __splice_word = #test;),
                emit!(::core::option::Option::<bool>::None),
            )
        };
        let inner = self.nested_region();
        let yes_code = inner.delimit(self.write_arm(yes, stage, &at)?);
        let no_code = inner.delimit(self.write_arm(no, stage, &at)?);
        let (init, result) = result_slot(stage, self);
        Ok(Value::new(
            stage,
            emit!({
                #word
                let __splice_known_condition: ::core::option::Option<bool> = #known;
                #init
                let (__splice_yes, __splice_no, __splice_join) =
                    (__splice_builder.label()?, __splice_builder.label()?, __splice_builder.label()?);
                if __splice_known_condition.is_none() {
                    let __splice_cond = #root::frontend::scalar::RunArg::reg(__splice_word, __splice_builder)?;
                    __splice_builder.branch(__splice_cond, __splice_yes, __splice_no)?;
                }
                if __splice_known_condition != ::core::option::Option::Some(false) { __splice_builder.mark(__splice_yes)?; #yes_code; }
                if __splice_known_condition != ::core::option::Option::Some(true) { __splice_builder.mark(__splice_no)?; #no_code; }
                __splice_builder.mark(__splice_join)?; #result
            }),
        ))
    }
    /// `continue` (`exit` is `None`) or `break` with its value, if any.
    pub(super) fn loop_jump(
        &self,
        e: &Expr,
        label: Option<&syn::Lifetime>,
        exit: Option<Option<&Expr>>,
    ) -> Result<Value> {
        let frame = &self.jump_target(e, label)?[0];
        let (target, flag) = match exit {
            None => (&frame.next, &frame.reaches_next),
            Some(_) => (&frame.end, &frame.reaches_end),
        };
        let record = (frame.kind == LoopKind::Unrolled).then(|| emit!(#flag = true;));
        // rustc allows a value only on a `break` that leaves a `loop`, which
        // stages as a runtime loop.
        let value = match exit.flatten() {
            Some(value) => {
                let lowered = self.expr_with_hint(value, frame.value_type.get())?;
                let t = match (frame.value_type.get(), lowered.stage) {
                    (Some(t), _) => t,
                    (None, Stage::Reg(t) | Stage::Word(t) | Stage::Literal(t)) => t.base(),
                    (None, Stage::Host) => Scalar::Inferred,
                    (None, _) => return err(value, "a loop value must be a scalar"),
                };
                frame.value_type.set(Some(t));
                let register = self.reg(lowered, t, value)?;
                let slot = &frame.value;
                emit!(let __splice_value = #register; __splice_builder.assign(#slot, __splice_value)?;)
            }
            None => TokenStream::new(),
        };
        let region = self.region();
        Ok(Value::new(
            Stage::Never,
            emit!({#value #record __splice_builder.jump(#target)?; break #region;}),
        ))
    }
    pub(super) fn loop_expr(
        &self,
        label: Option<&syn::Label>,
        cond: Option<&Expr>,
        body: &Block,
    ) -> Result<Value> {
        let mut scope = self.nested_region();
        let LoopFrame {
            next,
            end,
            value,
            value_type,
            ..
        } = scope.enter_loop(label, LoopKind::Runtime);
        let test = if let Some(e) = cond {
            let reg = scope.operand(e, Scalar::Bool)?;
            let point = scope.debug_point(e.span());
            emit!({#point let __splice_cond = #reg; __splice_builder.branch_if_false(__splice_cond, #end)?;})
        } else {
            emit!()
        };
        let body_code = scope.block(body)?.code;
        let contents = scope.delimit(emit!(#test #body_code; __splice_builder.jump(#next)?;));
        let (slot, result, stage) = match value_type.get() {
            Some(t) => {
                let ty = t.value_type(self.root, self.types);
                (
                    emit!(let #value = __splice_builder.local(#ty)?;),
                    emit!(#value),
                    Stage::Reg(t),
                )
            }
            None if cond.is_none() && !contains_break(body, label) => {
                (TokenStream::new(), TokenStream::new(), Stage::Never)
            }
            None => (TokenStream::new(), TokenStream::new(), Stage::Unit),
        };
        let code = emit!({
            #slot
            let (#next, #end) = (__splice_builder.label()?, __splice_builder.label()?);
            __splice_builder.jump(#next)?;
            __splice_builder.mark(#next)?;
            #contents
            __splice_builder.mark(#end)?;
            #result
        });
        Ok(Value::new(stage, code))
    }
    pub(super) fn for_expr(&self, f: &syn::ExprForLoop) -> Result<Value> {
        let Expr::Range(range) = &*f.expr else {
            if let Some(chunks) = self.chunks_loop(f)? {
                return Ok(chunks);
            }
            let Some((slice, ty, element, bytes)) = self.iterated_buffer(&f.expr)? else {
                // A method of a runtime slice reports itself, as `chars` does.
                if let Expr::MethodCall(call) = &*f.expr
                    && let Expr::Path(path) = &*call.receiver
                    && matches!(self.binding(path), Some(Binding::Buffer(_)))
                {
                    self.expr(&f.expr)?;
                    return err(&f.expr, "iterate a runtime slice with `for &x in xs`");
                }
                return self.unroll(f);
            };
            // `str::bytes` yields bytes; a slice yields references, which
            // `&x` or a plain name (dereferenced as `*x`) reads by value.
            let binding = match &*f.pat {
                Pat::Reference(pattern) if !bytes => crate::scalar::named(&pattern.pat)?,
                Pat::Ident(_) if !bytes && ty.write => {
                    return err(
                        &f.pat,
                        "a `&mut` element of a runtime slice is not supported; index the slice to write it",
                    );
                }
                pat @ Pat::Ident(_) => crate::scalar::named(pat)?,
                pat if bytes => crate::scalar::named(pat)?,
                _ => {
                    return err(
                        &f.pat,
                        "use a by-value reference pattern to read a scalar slice element",
                    );
                }
            };
            if binding.mutability.is_some() || binding.by_ref.is_some() {
                return err(binding, "expected an immutable loop variable");
            }
            let id = &binding.ident;
            let index = quote::format_ident!(
                "__splice_element_index",
                span = proc_macro2::Span::mixed_site()
            );
            let (start, one) = (self.constant(emit!(0usize)), self.constant(emit!(1usize)));
            let setup = emit!(let buffer = #slice;
                let (__splice_start, __splice_one) = (#start, #one);
                let __splice_limit = buffer.len;);
            let read = emit!(let #id = __splice_builder.load(buffer, #index)?;);
            let binding_kind = Binding::Reg(element, false);
            return self.counted_loop(f, binding, binding_kind, &index, setup, read);
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
        let (kind, one) = match stop_value.stage {
            // A variable below a slice length or a lane count fits a nonnegative i32.
            Stage::Reg(Scalar::Length) if matches!(&**start, Expr::Lit(l) if matches!(&l.lit, Lit::Int(i) if i.base10_parse::<i32>().is_ok())) => {
                (Scalar::Length, emit!(1usize))
            }
            Stage::Reg(Scalar::Index | Scalar::Length) => (Scalar::Index, emit!(1usize)),
            Stage::Reg(t @ Scalar::Number(_)) if !t.is_float() => {
                let ty = t.rust_type();
                (t, emit!(1 as #ty))
            }
            _ => (Scalar::I32, emit!(1i32)),
        };
        let initial = self.operand(start, kind)?;
        let bound = self.reg(stop_value, kind, stop)?;
        let one = self.constant(one);
        let setup =
            emit!(let (__splice_start, __splice_limit, __splice_one) = (#initial, #bound, #one););
        self.counted_loop(
            f,
            pat,
            Binding::Reg(kind, false),
            &pat.ident,
            setup,
            TokenStream::new(),
        )
    }

    /// `for chunk in xs.chunks_exact(n)` (or `chunks_exact_mut`): `chunk`
    /// is the subslice of `n` elements at each multiple of `n` below the
    /// end of the last whole chunk. `n == 0` traps where Rust panics.
    fn chunks_loop(&self, f: &syn::ExprForLoop) -> Result<Option<Value>> {
        let Expr::MethodCall(call) = &*f.expr else {
            return Ok(None);
        };
        let write = call.method == "chunks_exact_mut";
        if !(write || call.method == "chunks_exact") || call.args.len() != 1 {
            return Ok(None);
        }
        let Some((slice, actual, size)) = self.chunks_exact(call, write)? else {
            return Ok(None);
        };
        let Pat::Ident(pat) = &*f.pat else {
            return err(
                &f.pat,
                "bind each chunk to a name: `for chunk in xs.chunks_exact(n)`",
            );
        };
        if pat.mutability.is_some() || pat.by_ref.is_some() || pat.subpat.is_some() {
            return err(pat, "a runtime slice binding must be a fixed reference");
        }
        let id = &pat.ident;
        let offset = quote::format_ident!(
            "__splice_chunk_offset",
            span = proc_macro2::Span::mixed_site()
        );
        let start = self.constant(emit!(0usize));
        let setup = emit!(
            let buffer = #slice;
            let __splice_size = #size;
            let __splice_one = __splice_builder.copy(__splice_size)?;
            let __splice_limit = __splice_builder.chunks_exact_end(buffer, __splice_one)?;
            let __splice_start = #start;
            let __splice_chunk_len = __splice_builder.chunk_len(buffer, __splice_one)?;
        );
        let read = emit!(
            let #id = __splice_builder.chunk(buffer, #offset, __splice_chunk_len)?;
        );
        let ty = super::BufferType { write, ..actual };
        self.counted_loop(f, pat, Binding::Buffer(ty), &offset, setup, read)
            .map(Some)
    }

    /// The slice and the chunk size of `xs.chunks_exact(n)`, or `None`
    /// when `xs` is not a runtime slice.
    pub(super) fn chunks_exact(
        &self,
        call: &syn::ExprMethodCall,
        write: bool,
    ) -> Result<Option<(TokenStream, super::BufferType, TokenStream)>> {
        let slice = self.expr(&call.receiver)?;
        let Stage::Buffer(actual) = slice.stage else {
            return Ok(None);
        };
        if write && !actual.write {
            return err(call, "`chunks_exact_mut` needs a writable slice");
        }
        let size = self.expr(&call.args[0])?;
        let size = self.reg(size, Scalar::Index, &call.args[0])?;
        Ok(Some((slice.code, actual, size)))
    }

    /// A runtime slice a `for` loop reads element by element, and the
    /// element's register.
    pub(super) fn iterated_buffer(
        &self,
        iterable: &Expr,
    ) -> Result<Option<(TokenStream, super::BufferType, Scalar, bool)>> {
        let (operand, bytes) = match iterable {
            Expr::MethodCall(call) if call.method == "iter" && call.args.is_empty() => {
                (&*call.receiver, false)
            }
            Expr::MethodCall(call) if call.method == "bytes" && call.args.is_empty() => {
                (&*call.receiver, true)
            }
            other => (other, false),
        };
        // A path names a slice binding; `&xs[a..b]` reslices one.
        let path = match operand {
            Expr::Path(path) => path,
            Expr::Reference(reference) => match &*reference.expr {
                Expr::Index(index) if matches!(&*index.index, Expr::Range(_)) => {
                    match &*index.expr {
                        Expr::Path(path) => path,
                        _ => return Ok(None),
                    }
                }
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        let Some(Binding::Buffer(ty)) = self.binding(path) else {
            return Ok(None);
        };
        let Some(element) = ty.elem.scalar().filter(|_| ty.utf8 == bytes) else {
            return Ok(None);
        };
        let slice = self.expr(operand)?;
        let Stage::Buffer(ty) = slice.stage else {
            return Ok(None);
        };
        Ok(Some((slice.code, ty, element, bytes)))
    }

    fn counted_loop(
        &self,
        f: &syn::ExprForLoop,
        pat: &syn::PatIdent,
        binding: Binding,
        index: &proc_macro2::Ident,
        setup: TokenStream,
        read: TokenStream,
    ) -> Result<Value> {
        let mut scope = self.nested_region();
        let LoopFrame { next, end, .. } = scope.enter_loop(f.label.as_ref(), LoopKind::Runtime);
        scope.declare(pat, binding);
        let point = scope.debug_point(f.for_token.span);
        let body = scope.block(&f.body)?.code;
        let root = self.root;
        let contents = scope.delimit(emit!(#body; __splice_builder.jump(#next)?;));
        Ok(Value::unit(emit!({
            #setup
            let #index = __splice_builder.copy(__splice_start)?;
            let (__splice_for_head, __splice_for_body, #next, #end) =
                (__splice_builder.label()?, __splice_builder.label()?, __splice_builder.label()?, __splice_builder.label()?);
            __splice_builder.jump(__splice_for_head)?;
            __splice_builder.mark(__splice_for_head)?;
            let __splice_cond = __splice_builder.binary(#root::hir::Binary::Lt, #index, __splice_limit)?;
            __splice_builder.branch(__splice_cond, __splice_for_body, #end)?;
            __splice_builder.mark(__splice_for_body)?;
            #read
            #point
            #contents
            __splice_builder.mark(#next)?;
            let __splice_next = __splice_builder.binary(#root::hir::Binary::Add, #index, __splice_one)?;
            __splice_builder.assign(#index, __splice_next)?;
            __splice_builder.jump(__splice_for_head)?;
            __splice_builder.mark(#end)?;
        })))
    }
    fn unroll(&self, f: &syn::ExprForLoop) -> Result<Value> {
        // The Rust iterator is evaluated once. The pattern (including references
        // and destructuring) exists only while staging this iteration.
        let iterable = self.plan_iterable(&f.expr)?;
        let pat = &f.pat;
        let root = self.root;
        let step = emit!(#root::frontend::scalar::specialization_step()?;);

        let (bindings, captures) = self.with_debug_pattern(pat);
        if !contains_transfer(&f.body) {
            // A straight-line plan needs no iteration labels or induction
            // variable. Only the operations in its selected steps reach IR.
            let body = bindings.block(&f.body)?;
            return Ok(Value::unit(
                emit!({for #pat in #iterable { #step #captures #body; }}),
            ));
        }
        let mut scope = bindings.nested_region();
        let LoopFrame {
            next,
            end,
            reaches_next,
            reaches_end,
            ..
        } = scope.enter_loop(f.label.as_ref(), LoopKind::Unrolled);
        let body = scope.block(&f.body)?;
        let fallthrough = (body.stage != Stage::Never).then(|| emit!(#reaches_next = true;));
        let iteration = scope.delimit(emit!(#body; #fallthrough));
        let region = self.region();
        // An iteration that leaves an outer loop reaches neither exit.
        Ok(Value::unit(emit!({
            let #end = __splice_builder.label()?;
            let mut #reaches_end = false;
            let mut #reaches_next = true;
            for #pat in #iterable {
                #step
                let #next = __splice_builder.label()?;
                #reaches_next = false;
                #captures
                #iteration
                __splice_builder.mark(#next)?;
                if !#reaches_next { break; }
            }
            __splice_builder.mark(#end)?;
            if !#reaches_next && !#reaches_end { break #region; }
        })))
    }
    pub(super) fn match_expr(&self, m: &syn::ExprMatch, hint: Option<Scalar>) -> Result<Value> {
        let test = self.expr(&m.expr)?;
        if test.stage.is_host() {
            return self.known_match(m, test, hint);
        }
        let scope = self.nested_region();
        let mut arms = Vec::new();
        for arm in &m.arms {
            let value = scope.expr_with_hint(&arm.body, hint)?;
            let point = scope.arm_point(arm);
            let guard = match &arm.guard {
                Some((_, g)) => {
                    let test = scope.operand(g, Scalar::Bool)?;
                    let point = scope.debug_point(g.span());
                    emit!({#point let passed = #test; __splice_builder.branch_if_false(passed, __splice_next_arm)?;})
                }
                None => emit!(),
            };
            arms.push((arm, guard, point, value));
        }
        let stage = self.join(arms.iter().map(|(_, _, _, v)| v), true, &m)?;
        let Stage::Reg(t) = test.stage else {
            return err(&m.expr, "expected a value in match");
        };
        // rustc has checked that the arms are exhaustive (`true`/`false`,
        // ranges covering the type): without a final `_`, falling past
        // the last arm cannot happen.
        let exhausted = (!m
            .arms
            .last()
            .is_some_and(|a| matches!(a.pat, Pat::Wild(_)) && a.guard.is_none()))
        .then(|| emit!(__splice_builder.trap_bounds()?;));
        let mut body = TokenStream::new();
        for (arm, guard, point, value) in arms {
            if !arm.attrs.is_empty() {
                return err(arm, "attributes on runtime arms are unsupported");
            }
            let write = self.write_arm(value, stage, &arm.body)?;
            let pattern = if matches!(arm.pat, Pat::Wild(_)) {
                emit!()
            } else {
                let pred = self.pattern(&arm.pat, t)?;
                emit!({let matched = #pred; __splice_builder.branch_if_false(matched, __splice_next_arm)?;})
            };
            let contents = scope.delimit(emit!(#guard #point #write;));
            body.extend(emit!({
                let __splice_next_arm = __splice_builder.label()?;
                #pattern #contents
                __splice_builder.mark(__splice_next_arm)?;
            }));
        }
        let (init, result) = result_slot(stage, self);
        Ok(Value::new(
            stage,
            emit!({
                let __splice_subject = #test;
                #init
                let __splice_join = __splice_builder.label()?;
                #body
                #exhausted
                __splice_builder.mark(__splice_join)?;
                #result
            }),
        ))
    }
    fn arm_point(&self, arm: &syn::Arm) -> TokenStream {
        if matches!(&*arm.body, Expr::Block(_)) {
            emit!()
        } else {
            self.debug_point(arm.body.span())
        }
    }

    /// A `match` on a known value: the generator selects the arm. A guard
    /// with a runtime condition makes the selection a chain: each arm whose
    /// pattern matches tests its guard at run time and falls through to the
    /// next matching arm, up to the first arm that is selected for certain.
    fn known_match(&self, m: &syn::ExprMatch, test: Value, hint: Option<Scalar>) -> Result<Value> {
        let mut guards = Vec::new();
        for arm in &m.arms {
            let (scope, captures) = self.with_debug_pattern(&arm.pat);
            let guard = match &arm.guard {
                Some((_, g)) => Some(scope.expr(g)?),
                None => None,
            };
            guards.push((scope, captures, guard));
        }
        let runtime = guards
            .iter()
            .any(|(_, _, guard)| guard.as_ref().is_some_and(|g| !g.stage.is_host()));
        let mut arms = Vec::new();
        for (arm, (scope, captures, guard)) in m.arms.iter().zip(guards) {
            let scope = if runtime {
                scope.nested_region()
            } else {
                scope
            };
            let value = scope.expr_with_hint(&arm.body, hint)?;
            arms.push((arm, scope.arm_point(arm), captures, guard, value));
        }
        let stage = self.join(arms.iter().map(|(.., v)| v), runtime, &m)?;
        if !runtime {
            let mut cases = TokenStream::new();
            for (arm, point, captures, guard, value) in arms {
                let attrs = &arm.attrs;
                let pat = &arm.pat;
                let guard = guard.map(|g| emit!(if #g));
                let body = self.host_arm(value, stage, &arm.body)?;
                cases.extend(emit!(#(#attrs)* #pat #guard => {#captures #point #body},));
            }
            return Ok(Value::new(stage, emit!(match #test {#cases})));
        }
        let selection = self.fresh_label("arms");
        let mut chain = TokenStream::new();
        for (arm, point, captures, guard, value) in arms {
            let attrs = &arm.attrs;
            let pat = &arm.pat;
            let write = self.write_arm(value, stage, &arm.body)?;
            let (host_guard, runtime_guard, selected) = match guard {
                Some(g) if !g.stage.is_host() => {
                    let test = self.reg(g, Scalar::Bool, &arm.guard.as_ref().unwrap().1)?;
                    (
                        TokenStream::new(),
                        emit!(let passed = #test; __splice_builder.branch_if_false(passed, __splice_next_arm)?;),
                        TokenStream::new(),
                    )
                }
                Some(g) => (emit!(if #g), TokenStream::new(), emit!(break #selection;)),
                None => (
                    TokenStream::new(),
                    TokenStream::new(),
                    emit!(break #selection;),
                ),
            };
            let contents = self
                .nested_region()
                .delimit(emit!(#runtime_guard #point #write;));
            chain.extend(emit! {
                #[allow(unreachable_patterns)]
                match __splice_subject {
                    #(#attrs)* #pat #host_guard => {
                        #captures
                        let __splice_next_arm = __splice_builder.label()?;
                        #contents
                        __splice_builder.mark(__splice_next_arm)?;
                        #selected
                    }
                    _ => {}
                }
            });
        }
        let (init, result) = result_slot(stage, self);
        Ok(Value::new(
            stage,
            emit!({
                let __splice_subject = #test;
                #init
                let __splice_join = __splice_builder.label()?;
                #[allow(unreachable_code)]
                #selection: {
                    #chain
                    // rustc has checked that an arm without a runtime guard
                    // matches when the guarded ones fail.
                    __splice_builder.trap_bounds()?;
                }
                __splice_builder.mark(__splice_join)?;
                #result
            }),
        ))
    }

    fn pattern(&self, p: &Pat, t: Scalar) -> Result<TokenStream> {
        match p {
            Pat::Lit(l) => {
                let lit = match (&l.lit, t) {
                    (Lit::Int(_), t) if t.is_number() || t == Scalar::Inferred => {
                        let ty = t.rust_type().unwrap_or_else(|| quote::format_ident!("i32"));
                        self.constant(emit!({let __splice_value: #ty = #l; __splice_value}))
                    }
                    (Lit::Bool(_), Scalar::Bool | Scalar::Inferred) => self.constant(emit!(#l)),
                    (
                        Lit::Char(_),
                        Scalar::Number(crate::kast::types::Primitive::Char) | Scalar::Inferred,
                    ) => self.constant(emit!(#l)),
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
                let ty = t.rust_type().unwrap_or_else(|| quote::format_ident!("i32"));
                let cmp = if matches!(r.limits, syn::RangeLimits::Closed(_)) {
                    "Le"
                } else {
                    "Lt"
                };
                let mut pred = None;
                if let Some(start) = &r.start {
                    let v = self.host_code(start)?;
                    pred = Some(self.bin(
                        "Ge",
                        emit!(__splice_subject),
                        self.constant(emit!({let __splice_value: #ty = #v; __splice_value})),
                    ));
                }
                if let Some(stop) = &r.end {
                    let v = self.host_code(stop)?;
                    let upper = self.bin(
                        cmp,
                        emit!(__splice_subject),
                        self.constant(emit!({let __splice_value: #ty = #v; __splice_value})),
                    );
                    pred = Some(match pred {
                        Some(lower) => self.bin("And", lower, upper),
                        None => upper,
                    })
                }
                // syn parses an endpoint-free `..` as `Pat::Rest`.
                Ok(pred.expect("a range pattern has an endpoint"))
            }
            _ => err(
                p,
                "use a literal, a range, an or-pattern, or a final _ in a runtime match",
            ),
        }
    }
}

fn result_slot(stage: Stage, lower: &Lower<'_>) -> (TokenStream, TokenStream) {
    if let Stage::Reg(Scalar::Inferred) = stage {
        // Only the builder knows the type: the first arm creates the local.
        let root = lower.root;
        (
            emit!(let mut __splice_branch_result = ::core::option::Option::None;),
            emit!(#root::frontend::scalar::inferred_result(__splice_branch_result, __splice_builder)?),
        )
    } else if let Stage::Reg(t) = stage {
        let ty = t.value_type(lower.root, lower.types);
        (
            emit!(let __splice_branch_result = __splice_builder.local(#ty)?;),
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

/// Whether a `break` in `body` leaves the loop labeled `label`: an
/// unlabeled one outside nested loops, or one naming the label anywhere.
fn contains_break(body: &Block, label: Option<&syn::Label>) -> bool {
    struct Find<'a> {
        label: Option<&'a syn::Lifetime>,
        nested: usize,
        found: bool,
    }
    impl<'ast> Visit<'ast> for Find<'_> {
        fn visit_expr_break(&mut self, exit: &'ast syn::ExprBreak) {
            self.found |= match &exit.label {
                None => self.nested == 0,
                Some(label) => self.label == Some(label),
            };
            syn::visit::visit_expr_break(self, exit);
        }
        fn visit_expr(&mut self, expr: &'ast Expr) {
            let nested = matches!(expr, Expr::While(_) | Expr::ForLoop(_) | Expr::Loop(_));
            self.nested += usize::from(nested);
            syn::visit::visit_expr(self, expr);
            self.nested -= usize::from(nested);
        }
        fn visit_item(&mut self, _: &'ast syn::Item) {}
    }
    let mut find = Find {
        label: label.map(|label| &label.name),
        nested: 0,
        found: false,
    };
    find.visit_block(body);
    find.found
}
