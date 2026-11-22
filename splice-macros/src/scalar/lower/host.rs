//! Prepare Rust expressions used during specialization and preserve the
//! control-flow location of failures in residual code or printed source.
use super::{Binding, Lower, vec_macro};
use proc_macro2::{Ident, TokenStream};
use quote::ToTokens;
use std::collections::HashMap;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{BinOp, Block, Error, Expr, Result, UnOp};

struct HostCheck<'a> {
    names: &'a HashMap<String, Binding>,
    region: usize,
    outer: Vec<Ident>,
    error: Option<Error>,
    rewrite: bool,
    guarded_loop: bool,
}
impl<'ast> Visit<'ast> for HostCheck<'_> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        let loop_expr = matches!(expr, Expr::Loop(_) | Expr::While(_) | Expr::ForLoop(_));
        self.rewrite |= loop_expr
            || matches!(
                expr,
                Expr::Field(_) | Expr::Index(_) | Expr::Call(_) | Expr::MethodCall(_)
            );
        let previous = self.guarded_loop;
        self.guarded_loop |= loop_expr;
        syn::visit::visit_expr(self, expr);
        self.guarded_loop = previous;
    }
    fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
        self.rewrite = true;
        syn::visit::visit_expr_binary(self, binary);
    }
    fn visit_expr_unary(&mut self, unary: &'ast syn::ExprUnary) {
        self.rewrite |= guarded_negation(unary) || !matches!(unary.op, UnOp::Neg(_));
        syn::visit::visit_expr_unary(self, unary);
    }
    fn visit_expr_path(&mut self, p: &'ast syn::ExprPath) {
        if let Some(id) = p.path.get_ident() {
            match self.names.get(&id.to_string()) {
                Some(Binding::Host(depth)) if *depth < self.region && !self.outer.contains(id) => {
                    self.outer.push(id.clone());
                }
                Some(
                    Binding::Aggregate(_)
                    | Binding::Reg(..)
                    | Binding::Word(_)
                    | Binding::Buffer(_)
                    | Binding::Columns
                    | Binding::Context,
                ) => {
                    self.error = Some(Error::new_spanned(
                        id,
                        "runtime value used in a comptime expression",
                    ));
                }
                _ => {}
            }
        }
        syn::visit::visit_expr_path(self, p);
    }
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        if self.guarded_loop && m.path.is_ident("comptime") {
            // The mandatory evaluator checks this expression when reached.
            return;
        }
        if vec_macro(&m.path) {
            let tokens = &m.tokens;
            match syn::parse2::<Expr>(emit!([#tokens])) {
                Ok(elements) => self.visit_expr(&elements),
                Err(error) => self.error = Some(error),
            }
            return;
        }
        self.error = Some(Error::new_spanned(
            m,
            "put nested Rust macros in a host helper outside the kernel",
        ));
    }
}

pub(super) fn guarded_negation(unary: &syn::ExprUnary) -> bool {
    // Rust permits a negative literal with the minimum signed value.
    let mut operand = &*unary.expr;
    loop {
        operand = match operand {
            Expr::Paren(p) => &p.expr,
            Expr::Group(g) => &g.expr,
            _ => break,
        };
    }
    matches!(unary.op, UnOp::Neg(_)) && !matches!(operand, Expr::Lit(_))
}

struct PrepareHost<'a, 'ast> {
    lower: &'a Lower<'ast>,
    guarded_loop: bool,
    error: Option<Error>,
}

impl PrepareHost<'_, '_> {
    fn index_store(
        &mut self,
        left: &mut Expr,
        right: &mut Expr,
        op: Option<&BinOp>,
    ) -> Option<TokenStream> {
        let Expr::Index(index) = left else {
            return None;
        };
        let Expr::Path(base) = &*index.expr else {
            return Some(emit!(Err::<(), String>(
                "specialization cannot write through an opaque place".into()
            )?));
        };
        self.visit_expr_mut(right);
        self.visit_expr_mut(&mut index.index);
        let offset = &index.index;
        let root = self.lower.root;
        let place = emit!(#base[__splice_index]);
        let store = match op.and_then(|op| super::super::compound_op(op).map(|name| (op, name))) {
            Some((op, name)) => self
                .lower
                .host_compound(place, emit!(__splice_value), op, name),
            None => emit!(#place = __splice_value),
        };
        Some(emit!({
            use #root::frontend::evaluation::{Operand, CheckOwnedIndex as _};
            let __splice_value = #right;
            let __splice_index = #offset;
            (&Operand::of(&#base)).check_owned_index(&__splice_index)?;
            #store;
        }))
    }

    fn loop_body(&mut self, body: &mut Block) {
        let guarded_loop = std::mem::replace(&mut self.guarded_loop, true);
        self.visit_block_mut(body);
        self.guarded_loop = guarded_loop;
        let root = self.lower.root;
        body.stmts.insert(
            0,
            syn::parse_quote!(#root::frontend::scalar::specialization_step()?;),
        );
    }
}

impl VisitMut for PrepareHost<'_, '_> {
    fn visit_macro_mut(&mut self, m: &mut syn::Macro) {
        if vec_macro(&m.path) {
            let tokens = &m.tokens;
            let mut elements: Expr = syn::parse_quote!([#tokens]);
            self.visit_expr_mut(&mut elements);
            m.tokens = match elements {
                Expr::Array(a) => a.elems.into_token_stream(),
                Expr::Repeat(a) => {
                    let (value, len) = (a.expr, a.len);
                    emit!(#value; #len)
                }
                _ => unreachable!("validated vec! arguments"),
            };
        }
    }
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // These Rust bodies have their own execution context. A branch in the
        // surrounding JIT function cannot terminate inside one of them.
        if matches!(expr, Expr::Async(_) | Expr::Const(_) | Expr::Closure(_)) {
            return;
        }
        let store = match expr {
            Expr::Assign(assign) => self.index_store(&mut assign.left, &mut assign.right, None),
            Expr::Binary(binary) if super::super::compound_op(&binary.op).is_some() => {
                self.index_store(&mut binary.left, &mut binary.right, Some(&binary.op))
            }
            _ => None,
        };
        if let Some(store) = store {
            *expr = syn::parse_quote!(#store);
            return;
        }
        if let Expr::Reference(reference) = expr
            && reference.mutability.is_none()
            && let Expr::Index(index) = &mut *reference.expr
        {
            self.visit_expr_mut(&mut index.expr);
            self.visit_expr_mut(&mut index.index);
            let read = self
                .lower
                .known_index_reference(index.expr.to_token_stream(), index.index.to_token_stream());
            let fail = self.lower.bounds_failure();
            *expr = syn::parse_quote!(match (#read)? { Some(value) => value, None => { #fail } });
            return;
        }
        if matches!(expr, Expr::Call(_) | Expr::MethodCall(_)) {
            match self.lower.required(expr, None).and_then(syn::parse2) {
                Ok(code) => *expr = code,
                Err(error) => self.error = Some(error),
            }
            return;
        }
        if self.guarded_loop
            && let Expr::Macro(value) = expr
            && value.mac.path.is_ident("comptime")
        {
            match value
                .mac
                .parse_body::<Expr>()
                .and_then(|value| self.lower.required(&value, None))
                .and_then(syn::parse2)
            {
                Ok(code) => *expr = code,
                Err(error) => self.error = Some(error),
            }
            return;
        }
        match expr {
            Expr::ForLoop(value) => {
                match self.lower.plan_iterable(&value.expr).and_then(syn::parse2) {
                    Ok(code) => *value.expr = code,
                    Err(error) => self.error = Some(error),
                }
                self.loop_body(&mut value.body);
                return;
            }
            Expr::While(value) => {
                let guarded_loop = std::mem::replace(&mut self.guarded_loop, true);
                self.visit_expr_mut(&mut value.cond);
                self.guarded_loop = guarded_loop;
                self.loop_body(&mut value.body);
                return;
            }
            Expr::Loop(value) => {
                self.loop_body(&mut value.body);
                return;
            }
            _ => {}
        }
        syn::visit_mut::visit_expr_mut(self, expr);
        let code = match expr {
            Expr::Field(field) => self
                .lower
                .known_field(field.base.to_token_stream(), &field.member),
            Expr::Binary(binary) => {
                if let Some(kind) = super::super::compound_op(&binary.op) {
                    self.lower
                        .host_compound(&binary.left, &binary.right, &binary.op, kind)
                } else {
                    self.lower
                        .host_binary(&binary.left, &binary.op, &binary.right)
                }
            }
            Expr::Unary(unary) if matches!(unary.op, UnOp::Deref(_)) => {
                self.lower.known_deref(&unary.expr)
            }
            Expr::Unary(unary) if matches!(unary.op, UnOp::Not(_)) => {
                self.lower.host_not(&unary.expr)
            }
            Expr::Unary(unary) if guarded_negation(unary) => self.lower.host_negation(&unary.expr),
            _ => return,
        };
        *expr = syn::parse_quote!(#code);
    }
    fn visit_item_mut(&mut self, _: &mut syn::Item) {}
}
impl Lower<'_> {
    /// Validate a plan source before executing its constructor or next step.
    pub(super) fn plan_iterable(&self, expr: &Expr) -> Result<TokenStream> {
        let root = self.root;
        let support = emit!(#root::frontend::evaluation);
        self.host_check(expr, false)?;
        let reject = || {
            emit!(if false { #expr } else {
                return Err("opaque iterator constructor during specialization".into());
            })
        };
        match expr {
            Expr::MethodCall(call) if call.method == "enumerate" && call.args.is_empty() => {
                let source = self.plan_iterable(&call.receiver)?;
                Ok(emit!((#source).enumerate()))
            }
            Expr::MethodCall(call) if call.method == "iter" && call.args.is_empty() => {
                let Some(receiver) = self.plan_place(&call.receiver) else {
                    return Ok(reject());
                };
                Ok(emit!({
                    use #support::{Operand, CheckPlanSource as _};
                    let source = &(#receiver);
                    (&Operand::of(source)).check_source()?;
                    source.iter()
                }))
            }
            Expr::Range(range) => {
                let lo = range
                    .start
                    .as_ref()
                    .map(|e| self.required(e, None))
                    .transpose()?;
                let hi = range
                    .end
                    .as_ref()
                    .map(|e| self.required(e, None))
                    .transpose()?;
                let op = &range.limits;
                Ok(emit!({
                    use #support::{Operand, CheckPlanSource as _};
                    let source = #lo #op #hi;
                    (&Operand::of(&source)).check_source()?;
                    source
                }))
            }
            value => {
                let Some(place) = self.plan_place(value) else {
                    return Ok(reject());
                };
                Ok(emit!({
                use #support::{Operand, CheckPlanSource as _};
                let source = #place;
                (&Operand::of(&source)).check_source()?;
                source
                }))
            }
        }
    }

    /// Build a place without invoking a user-defined Deref while projecting fields.
    fn plan_place(&self, expr: &Expr) -> Option<TokenStream> {
        Some(match expr {
            Expr::Path(_) => emit!(#expr),
            Expr::Reference(r) if r.mutability.is_none() => {
                let place = self.plan_place(&r.expr)?;
                emit!(&(#place))
            }
            Expr::Field(f) => self.known_field(self.plan_place(&f.base)?, &f.member),
            Expr::Paren(p) => self.plan_place(&p.expr)?,
            Expr::Group(g) => self.plan_place(&g.expr)?,
            _ => return None,
        })
    }

    pub(super) fn known_deref(&self, base: impl ToTokens) -> TokenStream {
        let root = self.root;
        emit!(*{
            use #root::frontend::evaluation::{CheckReference as _, Operand};
            let receiver = &(#base);
            (&Operand::of(receiver)).check_reference()?;
            &**receiver
        })
    }

    pub(super) fn host_not(&self, operand: impl ToTokens) -> TokenStream {
        let root = self.root;
        emit!({
            use #root::frontend::evaluation::{CheckPrimitive as _, Operand};
            let operand = #operand;
            (&Operand::of(&operand)).check()?;
            !operand
        })
    }

    pub(super) fn known_field(&self, base: TokenStream, member: &syn::Member) -> TokenStream {
        let root = self.root;
        let name = match member {
            syn::Member::Named(id) => id.to_string().trim_start_matches("r#").to_owned(),
            syn::Member::Unnamed(index) => index.index.to_string(),
        };
        emit!(*{
            use #root::frontend::evaluation::{OwnField as _, ReferentField as _, Operand};
            let receiver = &(#base);
            let operand = Operand::of(receiver);
            if !(&operand).own_field(#name) && !(&operand).referent_field(#name) {
                return Err("field has no Splice layout description".into());
            }
            &receiver.#member
        })
    }

    pub(super) fn host_raw(&self, e: &Expr) -> Result<TokenStream> {
        self.host_code(e, false)
    }

    pub(super) fn host_only(&self, e: &Expr) -> bool {
        self.host_check(e, false).is_ok()
    }

    fn host_check(&self, e: &Expr, guarded_loop: bool) -> Result<HostCheck<'_>> {
        let mut check = HostCheck {
            names: &self.bindings,
            region: self.region_depth,
            outer: Vec::new(),
            error: None,
            rewrite: false,
            guarded_loop,
        };
        check.visit_expr(e);
        if let Some(error) = check.error.take() {
            return Err(error);
        }
        Ok(check)
    }

    pub(super) fn host_code(&self, e: &Expr, guarded_loop: bool) -> Result<TokenStream> {
        let check = self.host_check(e, guarded_loop)?;
        let code = if check.rewrite || guarded_loop {
            let mut expr = e.clone();
            let mut prepare = PrepareHost {
                lower: self,
                guarded_loop,
                error: None,
            };
            prepare.visit_expr_mut(&mut expr);
            if let Some(error) = prepare.error {
                return Err(error);
            }
            emit!(#expr)
        } else {
            emit!(#e)
        };
        let outer = check.outer;
        if outer.is_empty() {
            return Ok(code);
        }
        // The generator visits both runtime arms. A method call in one arm
        // must not mutate a captured host collection that the other arm sees.
        Ok(emit!({
            let __splice_borrowed = (#(&#outer,)*);
            let __splice_value = #code;
            let _ = &__splice_borrowed;
            __splice_value
        }))
    }
    pub(super) fn host_compound(
        &self,
        left: impl ToTokens,
        right: impl ToTokens,
        op: &BinOp,
        kind: &str,
    ) -> TokenStream {
        let root = self.root;
        let arithmetic = match kind {
            "Div" | "Rem" => Some(kind),
            _ => super::checked_operator(kind),
        }
        .map(|checked| self.host_arithmetic_check(checked, emit!(&(&*place, &operand))));
        emit!({
            use #root::frontend::evaluation::{CheckPrimitive as _, Operand};
            let operand = #right;
            let place = &mut #left;
            (&Operand::of(place)).check()?;
            (&Operand::of(&operand)).check()?;
            #arithmetic
            *place #op operand;
        })
    }

    pub(super) fn host_binary(
        &self,
        lhs: impl ToTokens,
        op: &BinOp,
        rhs: impl ToTokens,
    ) -> TokenStream {
        let root = self.root;
        let operation = match op {
            BinOp::Eq(_)
            | BinOp::Ne(_)
            | BinOp::Lt(_)
            | BinOp::Le(_)
            | BinOp::Gt(_)
            | BinOp::Ge(_) => {
                // Comparison operands are implicitly borrowed by Rust.
                return emit!({
                    let operands = (&(#lhs), &(#rhs));
                    use #root::frontend::evaluation::{CheckPrimitive as _, Operand};
                    (&Operand::of(operands.0)).check()?;
                    (&Operand::of(operands.1)).check()?;
                    *operands.0 #op *operands.1
                });
            }
            BinOp::And(_) | BinOp::Or(_) => return emit!((#lhs) #op (#rhs)),
            BinOp::Add(_) => "CheckedAdd",
            BinOp::Sub(_) => "CheckedSub",
            BinOp::Mul(_) => "CheckedMul",
            BinOp::Shl(_) => "CheckedShl",
            BinOp::Shr(_) => "CheckedShr",
            BinOp::Div(_) => "Div",
            BinOp::Rem(_) => "Rem",
            _ => "",
        };
        let check = (!operation.is_empty())
            .then(|| self.host_arithmetic_check(operation, emit!(&operands)));
        emit!({
            let operands = (#lhs, #rhs);
            use #root::frontend::evaluation::{CheckPrimitive as _, Operand};
            (&Operand::of(&operands.0)).check()?;
            (&Operand::of(&operands.1)).check()?;
            #check
            operands.0 #op operands.1
        })
    }

    pub(super) fn host_negation(&self, value: impl ToTokens) -> TokenStream {
        let root = self.root;
        let check = self.host_arithmetic_check("CheckedSub", emit!(&(0_i32, &operand)));
        emit!({
            let operand = #value;
            use #root::frontend::evaluation::{CheckPrimitive as _, Operand};
            (&Operand::of(&operand)).check()?;
            #check
            -operand
        })
    }

    pub(super) fn host_arithmetic_check(&self, op: &str, operands: TokenStream) -> TokenStream {
        let (root, region) = (self.root, self.region());
        let operation = Ident::new(op, proc_macro2::Span::call_site());
        let division = matches!(op, "Div" | "Rem");
        let checks = if division {
            emit!(true)
        } else {
            crate::overflow::enabled()
        };
        let fault = if let Some(label) = &self.source_fault {
            let panic = if division {
                emit!(#root::frontend::source::division_fault(z))
            } else {
                emit!(r#"panic!("kernel integer arithmetic overflow")"#.to_owned())
            };
            emit! {
                __splice_source.transfer = Some(#root::frontend::source::Transfer::Return);
                break #label (#panic);
            }
        } else {
            emit! {
                let (a, z) = (__splice_builder.constant(a), __splice_builder.constant(z));
                let fault = __splice_builder.binary(#root::hir::Binary::#operation, a, z);
                __splice_builder.ret(fault);
                break #region;
            }
        };
        emit! {
            use #root::frontend::scalar::ArithmeticGuard as _;
            if #checks && let Some((a, z)) = (#operands).faulting_i32_arithmetic(#root::hir::Binary::#operation) {
                let _ = (a, z);
                #fault
            }
        }
    }
}
