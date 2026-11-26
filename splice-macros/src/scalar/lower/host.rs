//! Prepare Rust expressions used during specialization and preserve the
//! control-flow location of failures in residual code or printed source.
use super::{Binding, Lower};
use crate::kast::vec_macro;
use proc_macro2::{Ident, TokenStream};
use quote::ToTokens;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{BinOp, Block, Error, Expr, Path, Result, UnOp};

struct HostCheck<'a> {
    lower: &'a Lower<'a>,
    region: usize,
    outer: Vec<Ident>,
    error: Option<Error>,
    rewrite: bool,
}
impl<'ast, 'a: 'ast> Visit<'ast> for HostCheck<'a> {
    fn visit_expr(&mut self, expr: &'ast Expr) {
        self.rewrite |= matches!(
            expr,
            Expr::Loop(_)
                | Expr::While(_)
                | Expr::ForLoop(_)
                | Expr::Field(_)
                | Expr::Index(_)
                | Expr::Call(_)
                | Expr::MethodCall(_)
        );
        syn::visit::visit_expr(self, expr);
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
            match self.lower.binding(p) {
                Some(Binding::Host(depth)) if depth < self.region && !self.outer.contains(id) => {
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
        if m.path.is_ident("comptime") {
            // The mandatory evaluator checks this expression when reached.
            self.rewrite = true;
            return;
        }
        if vec_macro(&m.path) {
            self.visit_expr(self.lower.body.vec_elements(m));
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

/// Emit the Rust code of a host expression that the generator runs during
/// specialization. It reads the nodes of the kernel body and never lowers a
/// clone: every expression it hands to the lowering is a node of the body.
struct PrepareHost<'a, 'ast> {
    lower: &'a Lower<'ast>,
    error: Option<Error>,
}

/// The nearest expressions below a node, through its statements and
/// patterns, and its `vec!` statements: the parts [`PrepareHost`] rewrites.
#[derive(Default)]
struct Children<'n>(Vec<Child<'n>>);

enum Child<'n> {
    Expr(&'n Expr),
    Vec(&'n syn::Macro),
}

impl<'n> Visit<'n> for Children<'n> {
    fn visit_expr(&mut self, expr: &'n Expr) {
        self.0.push(Child::Expr(expr));
    }
    fn visit_macro(&mut self, m: &'n syn::Macro) {
        if vec_macro(&m.path) {
            self.0.push(Child::Vec(m));
        }
    }
    fn visit_item(&mut self, _: &'n syn::Item) {}
}

/// Put the rewritten [`Children`] of a node into the printed copy of that
/// node, in the same walk order.
struct Replace(std::vec::IntoIter<TokenStream>);

impl Replace {
    fn next(&mut self) -> TokenStream {
        self.0
            .next()
            .expect("internal error: a rewritten child for every child")
    }
}

impl VisitMut for Replace {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // A parsed child lets the printer add the parentheses that keep the
        // precedence of the original node (`(*field)[i]`).
        let code = self.next();
        *expr = syn::parse2(code.clone()).unwrap_or(Expr::Verbatim(code));
    }
    fn visit_macro_mut(&mut self, m: &mut syn::Macro) {
        if vec_macro(&m.path) {
            m.tokens = self.next();
        }
    }
    fn visit_item_mut(&mut self, _: &mut syn::Item) {}
}

impl PrepareHost<'_, '_> {
    fn attempt(&mut self, code: Result<TokenStream>) -> TokenStream {
        code.unwrap_or_else(|error| {
            self.error = Some(error);
            TokenStream::new()
        })
    }

    /// `node` with each of its children rewritten. The copy is only printed.
    fn rebuild<'n, T: Clone + ToTokens>(
        &mut self,
        node: &'n T,
        children: impl FnOnce(&mut Children<'n>, &'n T),
        replace: impl FnOnce(&mut Replace, &mut T),
    ) -> TokenStream {
        let mut found = Children::default();
        children(&mut found, node);
        let rewritten = found
            .0
            .into_iter()
            .map(|child| match child {
                Child::Expr(expr) => self.expr(expr),
                Child::Vec(m) => self.vec_elements(m),
            })
            .collect::<Vec<_>>();
        let mut copy = node.clone();
        replace(&mut Replace(rewritten.into_iter()), &mut copy);
        copy.into_token_stream()
    }

    fn children(&mut self, expr: &Expr) -> TokenStream {
        self.rebuild(expr, syn::visit::visit_expr, syn::visit_mut::visit_expr_mut)
    }

    fn stmt(&mut self, stmt: &syn::Stmt) -> TokenStream {
        self.rebuild(stmt, syn::visit::visit_stmt, syn::visit_mut::visit_stmt_mut)
    }

    /// The rewritten arguments of a `vec!` invocation.
    fn vec_elements(&mut self, m: &syn::Macro) -> TokenStream {
        match self.lower.body.vec_elements(m) {
            Expr::Array(array) => {
                let values = array.elems.iter().map(|value| self.expr(value));
                emit!(#(#values),*)
            }
            Expr::Repeat(repeat) => {
                let (value, len) = (self.expr(&repeat.expr), self.expr(&repeat.len));
                emit!(#value; #len)
            }
            _ => unreachable!("validated vec! arguments"),
        }
    }

    fn index_store(
        &mut self,
        left: &Expr,
        right: &Expr,
        op: Option<&BinOp>,
    ) -> Option<TokenStream> {
        let Expr::Index(index) = left else {
            return None;
        };
        let root = self.lower.root;
        let Expr::Path(base) = &*index.expr else {
            return Some(emit!(::core::result::Result::Err::<
                (),
                #root::BuildError,
            >(#root::BuildError::staging(
                "specialization cannot write through an opaque place"
            ))?));
        };
        let right = self.expr(right);
        let offset = self.expr(&index.index);
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

    fn loop_body(&mut self, body: &Block) -> TokenStream {
        let stmts = body
            .stmts
            .iter()
            .map(|stmt| self.stmt(stmt))
            .collect::<Vec<_>>();
        let root = self.lower.root;
        emit!({
            #root::frontend::scalar::specialization_step()?;
            #(#stmts)*
        })
    }

    fn expr(&mut self, expr: &Expr) -> TokenStream {
        // These Rust bodies have their own execution context. A branch in the
        // surrounding JIT function cannot terminate inside one of them.
        if matches!(expr, Expr::Async(_) | Expr::Const(_) | Expr::Closure(_)) {
            return expr.to_token_stream();
        }
        let store = match expr {
            Expr::Assign(assign) => self.index_store(&assign.left, &assign.right, None),
            Expr::Binary(binary) if super::super::compound_op(&binary.op).is_some() => {
                self.index_store(&binary.left, &binary.right, Some(&binary.op))
            }
            _ => None,
        };
        if let Some(store) = store {
            return store;
        }
        if let Expr::Reference(reference) = expr
            && reference.mutability.is_none()
            && let Expr::Index(index) = &*reference.expr
        {
            let (receiver, offset) = (self.expr(&index.expr), self.expr(&index.index));
            let read = self.lower.known_index_reference(receiver, offset);
            let fail = self.lower.bounds_failure();
            return emit!(match (#read)? { Some(value) => value, None => { #fail } });
        }
        match expr {
            Expr::Call(_) | Expr::MethodCall(_) => {
                let code = self.lower.required(expr, None);
                self.attempt(code)
            }
            Expr::Macro(value) if value.mac.path.is_ident("comptime") => {
                let code = self
                    .lower
                    .required(self.lower.body.comptime(&value.mac), None);
                self.attempt(code)
            }
            Expr::Macro(value) if vec_macro(&value.mac.path) => {
                let mut mac = value.mac.clone();
                mac.tokens = self.vec_elements(&value.mac);
                let attrs = &value.attrs;
                emit!(#(#attrs)* #mac)
            }
            Expr::ForLoop(value) => {
                let iterable = self.lower.plan_iterable(&value.expr);
                let iterable = self.attempt(iterable);
                let body = self.loop_body(&value.body);
                let (attrs, label, pat) = (&value.attrs, &value.label, &value.pat);
                emit!(#(#attrs)* #label for #pat in #iterable #body)
            }
            Expr::While(value) => {
                let cond = self.expr(&value.cond);
                let body = self.loop_body(&value.body);
                let (attrs, label) = (&value.attrs, &value.label);
                emit!(#(#attrs)* #label while #cond #body)
            }
            Expr::Loop(value) => {
                let body = self.loop_body(&value.body);
                let (attrs, label) = (&value.attrs, &value.label);
                emit!(#(#attrs)* #label loop #body)
            }
            Expr::Field(field) => {
                let base = self.expr(&field.base);
                self.lower.known_field(base, &field.member)
            }
            Expr::Binary(binary) => {
                let (left, right) = (self.expr(&binary.left), self.expr(&binary.right));
                match super::super::compound_op(&binary.op) {
                    Some(kind) => self.lower.host_compound(left, right, &binary.op, kind),
                    None => self.lower.host_binary(left, &binary.op, right),
                }
            }
            Expr::Unary(unary) if matches!(unary.op, UnOp::Deref(_)) => {
                let operand = self.expr(&unary.expr);
                self.lower.known_deref(operand)
            }
            Expr::Unary(unary) if matches!(unary.op, UnOp::Not(_)) => {
                let operand = self.expr(&unary.expr);
                self.lower.host_not(operand)
            }
            Expr::Unary(unary) if guarded_negation(unary) => {
                let operand = self.expr(&unary.expr);
                self.lower.host_negation(operand)
            }
            _ => self.children(expr),
        }
    }
}

impl Lower<'_> {
    /// Validate a plan source before executing its constructor or next step.
    pub(super) fn plan_iterable(&self, expr: &Expr) -> Result<TokenStream> {
        let root = self.root;
        let support = emit!(#root::frontend::evaluation);
        self.host_check(expr)?;
        let reject = || {
            emit!(if false { #expr } else {
                return ::core::result::Result::Err(#root::BuildError::staging("opaque iterator constructor during specialization"));
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

    /// The optional evaluator of a call with known arguments gives up on an
    /// unsupported type and the call stages at run time; staging itself
    /// rejects the kernel when it compiles.
    pub(super) fn optional(&self) -> bool {
        self.evaluation.is_some()
    }

    /// Require `operand` (a `&T`) to be a Rust primitive.
    pub(super) fn primitive(&self, operand: impl ToTokens) -> TokenStream {
        let root = self.root;
        if self.optional() {
            emit!({
                use #root::frontend::evaluation::{CheckPrimitive as _, Operand};
                (&Operand::of(#operand)).check()?;
            })
        } else {
            emit!(#root::frontend::evaluation::primitive(#operand);)
        }
    }

    /// Require a value constructed during specialization to have no drop glue.
    pub(super) fn dropless(&self, value: impl ToTokens) -> TokenStream {
        let root = self.root;
        if self.optional() {
            emit!(#root::frontend::evaluation::Dropless::check()?.bind(#value))
        } else {
            emit!(#root::frontend::evaluation::dropless(#value))
        }
    }

    pub(super) fn known_deref(&self, base: impl ToTokens) -> TokenStream {
        let root = self.root;
        if self.optional() {
            return emit!(*{
                use #root::frontend::evaluation::{CheckReference as _, Operand};
                let receiver = &(#base);
                (&Operand::of(receiver)).check_reference()?;
                &**receiver
            });
        }
        emit!(*#root::frontend::evaluation::deref(&(#base)))
    }

    pub(super) fn host_not(&self, operand: impl ToTokens) -> TokenStream {
        let check = self.primitive(emit!(&operand));
        emit!({
            let operand = #operand;
            #check
            !operand
        })
    }

    pub(super) fn known_field(&self, base: TokenStream, member: &syn::Member) -> TokenStream {
        let root = self.root;
        let name = match member {
            syn::Member::Named(id) => id.to_string().trim_start_matches("r#").to_owned(),
            syn::Member::Unnamed(index) => index.index.to_string(),
        };
        let check = if self.optional() {
            emit!({
                use #root::frontend::evaluation::Layout as _;
                let owner = (&Operand::of(receiver)).owner();
                let layout = (&owner).layout();
                owner.declares(layout, #name)?;
            })
        } else {
            emit!(#root::frontend::evaluation::field((&Operand::of(receiver)).owner(), #name)?;)
        };
        // The layout requirement is reported at the field the kernel reads.
        let at = proc_macro2::Span::mixed_site().located_at(syn::spanned::Spanned::span(member));
        quote::quote_spanned!(at=> *{
            use #root::frontend::evaluation::{Operand, Projection as _};
            let receiver = &(#base);
            #check
            &receiver.#member
        })
    }

    pub(super) fn host_raw(&self, e: &Expr) -> Result<TokenStream> {
        self.host_code(e)
    }

    pub(super) fn host_only(&self, e: &Expr) -> bool {
        self.host_check(e).is_ok()
    }

    fn host_check(&self, e: &Expr) -> Result<HostCheck<'_>> {
        let mut check = HostCheck {
            lower: self,
            region: self.region_depth,
            outer: Vec::new(),
            error: None,
            rewrite: false,
        };
        check.visit_expr(e);
        if let Some(error) = check.error.take() {
            return Err(error);
        }
        Ok(check)
    }

    pub(super) fn host_code(&self, e: &Expr) -> Result<TokenStream> {
        let check = self.host_check(e)?;
        let code = if check.rewrite {
            let mut prepare = PrepareHost {
                lower: self,
                error: None,
            };
            let code = prepare.expr(e);
            if let Some(error) = prepare.error {
                return Err(error);
            }
            code
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
        let check = self.host_operator_check(kind, emit!(&(&*place, &operand)));
        let (place_check, operand_check) = (
            self.primitive(emit!(place)),
            self.primitive(emit!(&operand)),
        );
        emit!({
            let operand = #right;
            let place = &mut #left;
            #place_check
            #operand_check
            #check
            *place #op operand;
        })
    }

    pub(super) fn host_binary(
        &self,
        lhs: impl ToTokens,
        op: &BinOp,
        rhs: impl ToTokens,
    ) -> TokenStream {
        let kind = match op {
            BinOp::Eq(_)
            | BinOp::Ne(_)
            | BinOp::Lt(_)
            | BinOp::Le(_)
            | BinOp::Gt(_)
            | BinOp::Ge(_) => {
                // Comparison operands are implicitly borrowed by Rust.
                let (left, right) = (
                    self.primitive(emit!(operands.0)),
                    self.primitive(emit!(operands.1)),
                );
                return emit!({
                    let operands = (&(#lhs), &(#rhs));
                    #left
                    #right
                    *operands.0 #op *operands.1
                });
            }
            BinOp::And(_) | BinOp::Or(_) => return emit!((#lhs) #op (#rhs)),
            _ => operator_kind(op),
        };
        let check = kind.and_then(|kind| self.host_operator_check(kind, emit!(&operands)));
        let (left, right) = (
            self.primitive(emit!(&operands.0)),
            self.primitive(emit!(&operands.1)),
        );
        emit!({
            let operands = (#lhs, #rhs);
            #left
            #right
            #check
            operands.0 #op operands.1
        })
    }

    pub(super) fn host_negation(&self, value: impl ToTokens) -> TokenStream {
        let check = self.host_fault_check(emit!((&(&operand,)).negation_fault()), false);
        let primitive = self.primitive(emit!(&operand));
        emit!({
            let operand = #value;
            #primitive
            #check
            -operand
        })
    }

    /// Defer the fault of host operator `kind` (`Add`, `Div`, `Shl`, ...) on
    /// `operands`, a reference to the operand pair.
    fn host_operator_check(&self, kind: &str, operands: TokenStream) -> Option<TokenStream> {
        let (guard, division) = operator_fault(self.root, kind, operands)?;
        Some(self.host_fault_check(guard, division))
    }

    /// Evaluate `guard`, an `Option<HostFault>`, and on a fault leave the
    /// region so that the kernel traps when it reaches this point. Overflow
    /// faults follow the crate's overflow-check mode; division always faults.
    pub(super) fn host_fault_check(&self, guard: TokenStream, division: bool) -> TokenStream {
        let (root, region) = (self.root, self.region());
        let checks = fault_checks(division);
        let fault = if let Some(label) = &self.source_fault {
            emit! {
                __splice_source.transfer = ::core::option::Option::Some(#root::frontend::source::Transfer::Return);
                break #label (#root::frontend::source::fault_panic(fault));
            }
        } else {
            emit! {
                let (operation, a, z) = fault.kernel_operation();
                let (a, z) = (__splice_builder.constant(a)?, __splice_builder.constant(z)?);
                // The operation traps; the explicit trap after it is unreachable.
                __splice_builder.binary(operation, a, z)?;
                __splice_builder.trap_bounds()?;
                break #region;
            }
        };
        emit! {
            use #root::frontend::scalar::{HostArithmetic as _, HostNegation as _, HostShift as _};
            if #checks && let ::core::option::Option::Some(fault) = #guard {
                #fault
            }
        }
    }
}

/// The name of a binary operator that can fault on host integers.
pub(super) fn operator_kind(op: &BinOp) -> Option<&'static str> {
    Some(match op {
        BinOp::Add(_) => "Add",
        BinOp::Sub(_) => "Sub",
        BinOp::Mul(_) => "Mul",
        BinOp::Div(_) => "Div",
        BinOp::Rem(_) => "Rem",
        BinOp::Shl(_) => "Shl",
        BinOp::Shr(_) => "Shr",
        _ => return None,
    })
}

/// An `Option<HostFault>` expression for host operator `kind` (`Add`, `Div`,
/// `Shl`, ...) on `operands`, a reference to the operand pair, and whether the
/// operator divides. `None` when the operator cannot fault.
pub(super) fn operator_fault(
    root: &Path,
    kind: &str,
    operands: TokenStream,
) -> Option<(TokenStream, bool)> {
    let division = matches!(kind, "Div" | "Rem");
    let op = if division {
        kind
    } else {
        super::checked_operator(kind)?
    };
    let op = Ident::new(op, proc_macro2::Span::call_site());
    let guard = if matches!(kind, "Shl" | "Shr") {
        emit!((#operands).shift_fault(#root::hir::Binary::#op))
    } else {
        emit!((#operands).arithmetic_fault(#root::hir::Binary::#op))
    };
    Some((guard, division))
}

/// Whether to evaluate a fault guard: overflow follows the overflow-check
/// mode of the kernel's crate (host code is native Rust there), while
/// division faults in every mode.
pub(super) fn fault_checks(division: bool) -> TokenStream {
    if division {
        emit!(true)
    } else {
        emit!(__splice_overflow)
    }
}

/// Bind `__splice_overflow`, whether the kernel's crate compiles Rust
/// arithmetic with overflow checks, at the start of a generated function.
/// The probe is compiled in that crate, so it sees its real flags.
pub(crate) fn overflow_prologue(root: &syn::Path) -> TokenStream {
    emit! {
        #[allow(unused_variables)]
        let __splice_overflow: bool = {
            fn probe(value: i32) -> i32 {
                value + 1
            }
            #root::frontend::scalar::overflow_checks(
                ::core::module_path!(),
                probe,
                ::core::cfg!(panic = "unwind"),
                ::core::cfg!(debug_assertions),
            )
        };
    }
}

/// Whether staged kernel arithmetic checks overflow: the build options'
/// mode, or else the kernel crate's.
pub(super) fn kernel_overflow_checks() -> TokenStream {
    emit!(
        __splice_builder
            .overflow_checks()
            .unwrap_or(__splice_overflow)
    )
}
