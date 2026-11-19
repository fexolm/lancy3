//! Lower expressions, calls and bindings while tracking their stage and scalar type.
mod control;
mod memory;
pub(super) mod simd;
pub(super) use simd::{BufferType, Elem};

use proc_macro2::{Ident, TokenStream};
use quote::ToTokens;
use std::collections::HashMap;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{BinOp, Block, Error, Expr, Lit, Pat, Path, Result, Stmt, UnOp};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Scalar {
    Int,
    Bool,
    /// A usize used for indexing. Native slice lengths fit in i32.
    Index,
    /// A usize known to fit a nonnegative i32 (slice length or SIMD lane count).
    Length,
    Inferred,
    Vector(Elem),
    Mask(Elem),
}
impl Scalar {
    fn base(self) -> Self {
        if self == Self::Length {
            Self::Index
        } else {
            self
        }
    }
    pub(super) fn from_type(ty: &syn::Type) -> Option<Self> {
        if let Some(t) = simd::vector_type(ty) {
            return Some(t);
        }
        let name = match ty {
            syn::Type::Path(p) if p.qself.is_none() => p.path.get_ident()?,
            _ => return None,
        };
        [
            ("i32", Self::Int),
            ("bool", Self::Bool),
            ("usize", Self::Index),
        ]
        .into_iter()
        .find_map(|(ty, scalar)| (name == ty).then_some(scalar))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Host,
    Literal(Scalar),
    Reg(Scalar),
    Buffer(BufferType),
    Unit,
    Never,
}
impl Stage {
    fn is_host(self) -> bool {
        matches!(self, Self::Host | Self::Literal(_))
    }

    fn materialized(self) -> Self {
        match self {
            Self::Literal(t) => Self::Reg(t),
            Self::Host => Self::Reg(Scalar::Inferred),
            stage => stage,
        }
    }
}
#[derive(Clone, Copy)]
pub(super) enum Binding {
    Host,
    Reg(Scalar, bool),
    Buffer(BufferType),
}
struct Value {
    stage: Stage,
    code: TokenStream,
}
impl ToTokens for Value {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.code.to_tokens(tokens);
    }
}

impl Value {
    fn new(stage: Stage, code: impl ToTokens) -> Self {
        Self {
            stage,
            code: code.into_token_stream(),
        }
    }
    fn unit(code: TokenStream) -> Self {
        Self::new(Stage::Unit, code)
    }
}
#[derive(Clone, Copy)]
pub(super) enum LoopKind {
    Runtime,
    Unrolled,
}

#[derive(Clone)]
pub(super) struct Lower<'a> {
    pub(super) root: &'a Path,
    pub(super) name: &'a Ident,
    pub(super) formals: &'a [Binding],
    pub(super) bindings: HashMap<String, Binding>,
    pub(super) fish: TokenStream,
    pub(super) loop_kind: Option<LoopKind>,
    pub(super) region_depth: usize,
    pub(super) flow: &'a super::flow::Flow,
}

fn err<T>(at: impl ToTokens, msg: &str) -> Result<T> {
    Err(Error::new_spanned(at, msg))
}

struct HostCheck<'a> {
    names: &'a HashMap<String, Binding>,
    error: Option<Error>,
    division: bool,
}
impl<'ast> Visit<'ast> for HostCheck<'_> {
    fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
        self.division |= matches!(binary.op, BinOp::Div(_) | BinOp::Rem(_));
        syn::visit::visit_expr_binary(self, binary);
    }
    fn visit_expr_path(&mut self, p: &'ast syn::ExprPath) {
        if let Some(id) = p.path.get_ident()
            && matches!(
                self.names.get(&id.to_string()),
                Some(Binding::Reg(..) | Binding::Buffer(_))
            )
        {
            self.error = Some(Error::new_spanned(
                id,
                "runtime value used in a comptime expression",
            ));
        }
        syn::visit::visit_expr_path(self, p);
    }
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        self.error = Some(Error::new_spanned(
            m,
            "put nested Rust macros in a host helper outside the kernel",
        ));
    }
}

struct GuardHostDivision<'a, 'ast>(&'a Lower<'ast>);
impl VisitMut for GuardHostDivision<'_, '_> {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // These Rust bodies have their own execution context. A branch in the
        // surrounding JIT function cannot terminate inside one of them.
        if matches!(expr, Expr::Async(_) | Expr::Const(_) | Expr::Closure(_)) {
            return;
        }
        syn::visit_mut::visit_expr_mut(self, expr);
        if let Expr::Binary(binary) = expr
            && matches!(binary.op, BinOp::Div(_) | BinOp::Rem(_))
        {
            let code = self.0.host_binary(&binary.left, &binary.op, &binary.right);
            *expr = syn::parse_quote!(#code);
        }
    }
    fn visit_item_mut(&mut self, _: &mut syn::Item) {}
}
impl Lower<'_> {
    // Each runtime arm builds independently. A host-selected return, break
    // or continue can skip the rest of that arm without skipping its siblings.
    fn region(&self) -> syn::Lifetime {
        syn::Lifetime::new(
            &format!("'__splice_arm_{}", self.region_depth),
            proc_macro2::Span::mixed_site(),
        )
    }
    fn nested_region(&self) -> Self {
        let mut inner = self.clone();
        inner.region_depth += 1;
        inner
    }
    fn delimit(&self, code: TokenStream) -> TokenStream {
        let label = self.region();
        emit!(#[allow(unused_labels)] #label: { #code; })
    }
    fn host_raw(&self, e: &Expr) -> Result<TokenStream> {
        let mut check = HostCheck {
            names: &self.bindings,
            error: None,
            division: false,
        };
        check.visit_expr(e);
        if let Some(error) = check.error {
            return Err(error);
        }
        if self.region_depth > 0 && check.division {
            let mut expr = e.clone();
            GuardHostDivision(self).visit_expr_mut(&mut expr);
            return Ok(emit!(#expr));
        }
        Ok(emit!(#e))
    }
    fn host(&self, v: Value, at: impl ToTokens) -> Result<TokenStream> {
        match v.stage {
            Stage::Host | Stage::Literal(_) => Ok(v.code),
            _ => err(at, "expected a comptime value; found a runtime expression"),
        }
    }
    fn reg(&self, v: Value, want: Scalar, at: impl ToTokens) -> Result<TokenStream> {
        let root = self.root;
        let want = want.base();
        let bits = match (v.stage, want) {
            (Stage::Reg(t), _)
                if t.base() == want
                    || (!matches!(
                        t.base(),
                        Scalar::Index | Scalar::Vector(_) | Scalar::Mask(_)
                    ) && !matches!(
                        want,
                        Scalar::Index | Scalar::Vector(_) | Scalar::Mask(_)
                    ) && (t == Scalar::Inferred || want == Scalar::Inferred)) =>
            {
                return Ok(v.code);
            }
            (Stage::Host | Stage::Literal(Scalar::Int), Scalar::Index) => emit!({
                let __splice_index: usize = #v;
                let __splice_word = __splice_index as i32;
                if __splice_word as usize != __splice_index {
                    return Err("runtime usize value exceeds the scalar register width".into());
                }
                __splice_word
            }),
            (Stage::Literal(Scalar::Int), Scalar::Int | Scalar::Inferred) => v.code,
            (Stage::Literal(Scalar::Bool), Scalar::Bool | Scalar::Inferred) => {
                emit!(-i32::from(#v))
            }
            (Stage::Host, Scalar::Int) => emit!(*::core::borrow::Borrow::<i32>::borrow(&(#v))),
            (Stage::Host, Scalar::Bool) => {
                emit!(-i32::from(*::core::borrow::Borrow::<bool>::borrow(&(#v))))
            }
            (Stage::Host, Scalar::Inferred) => {
                emit!(#root::frontend::scalar::ScalarConstant::bits(#v))
            }
            _ => {
                return err(
                    at,
                    match want {
                        Scalar::Int => "expected a runtime i32",
                        Scalar::Index | Scalar::Length => "expected a usize slice index",
                        Scalar::Bool => "expected a runtime bool condition",
                        Scalar::Inferred => "expected a runtime scalar",
                        Scalar::Vector(_) | Scalar::Mask(_) => "SIMD operand type mismatch",
                    },
                );
            }
        };
        Ok(emit!({let __splice_word = #bits; __splice_builder.constant(__splice_word)}))
    }
    fn operand(&self, expr: &Expr, kind: Scalar) -> Result<TokenStream> {
        let value = self.expr_with_hint(expr, Some(kind))?;
        self.reg(value, kind, expr)
    }

    /// Either data operand can supply the type of an unqualified Lane constructor.
    /// The generated operands still execute in source order.
    fn paired_operands(
        &self,
        lhs: &Expr,
        rhs: &Expr,
        hint: Option<Scalar>,
        fallback: Option<Scalar>,
    ) -> Result<(Value, Value)> {
        match self.expr_with_hint(lhs, hint) {
            Ok(left) => {
                let ty = if let Stage::Reg(t) = left.stage {
                    Some(t)
                } else {
                    hint
                };
                Ok((left, self.expr_with_hint(rhs, ty)?))
            }
            Err(error) if simd::uses_lane(lhs) => {
                let right = self.expr_with_hint(rhs, hint.or(fallback))?;
                let Stage::Reg(t @ (Scalar::Vector(_) | Scalar::Mask(_))) = right.stage else {
                    return Err(error);
                };
                Ok((self.expr_with_hint(lhs, Some(t))?, right))
            }
            Err(error) => Err(error),
        }
    }
    fn bin(&self, op: &str, x: TokenStream, y: TokenStream) -> TokenStream {
        let root = self.root;
        let op = Ident::new(op, proc_macro2::Span::call_site());
        emit!({let (a,z)=(#x,#y); __splice_builder.binary(#root::hir::Binary::#op,a,z)})
    }
    fn stage_join(&self, a: Stage, z: Stage, at: impl ToTokens) -> Result<Stage> {
        use Stage::*;
        match (a, z) {
            (Never, other) | (other, Never) => Ok(other),
            (a, z) if a == z => Ok(a),
            (Reg(a), Reg(b)) if a.base() == b.base() => Ok(Reg(a.base())),
            (Buffer(a), Buffer(b)) if a.elem == b.elem => Ok(Buffer(BufferType {
                write: a.write && b.write,
                ..a
            })),
            (Reg(t), Literal(u)) | (Literal(u), Reg(t)) if t == u => Ok(Reg(t)),
            (Reg(Scalar::Inferred), Reg(t) | Literal(t))
            | (Reg(t) | Literal(t), Reg(Scalar::Inferred)) => Ok(Reg(t.base())),
            (Host, Literal(_)) | (Literal(_), Host) => Ok(Host),
            (Host, Reg(t)) | (Reg(t), Host) => Ok(Reg(t.base())),
            _ => err(at, "branch arms have different result types"),
        }
    }
    fn with_pattern(&self, pat: &Pat) -> Self {
        let mut scope = self.clone();
        super::bind_pattern(&mut scope.bindings, pat, Binding::Host);
        scope
    }

    fn binding(&self, p: &syn::ExprPath) -> Option<Binding> {
        self.bindings.get(&p.path.get_ident()?.to_string()).copied()
    }
    fn path(&self, p: &syn::ExprPath) -> Value {
        match self.binding(p) {
            Some(Binding::Reg(t, true)) => {
                // A later operand can assign this variable. Snapshot its value
                // at the Rust evaluation point before constructing the RHS.
                Value::new(Stage::Reg(t), emit!(__splice_builder.copy(#p)))
            }
            Some(Binding::Reg(t, false)) => Value::new(Stage::Reg(t), p),
            Some(Binding::Buffer(writable)) => Value::new(Stage::Buffer(writable), p),
            _ => Value::new(Stage::Host, p),
        }
    }
    pub(super) fn kernel_body(&self, block: &Block) -> Result<TokenStream> {
        let body = self.block(block)?;
        let code = if body.stage == Stage::Never {
            body.code
        } else {
            self.return_value(body, block)?
        };
        Ok(self.delimit(code))
    }

    fn return_value(&self, value: Value, at: impl ToTokens) -> Result<TokenStream> {
        let value = self.reg(value, Scalar::Int, at)?;
        Ok(emit!({
            let __splice_value = #value;
            __splice_builder.assign(__splice_result, __splice_value);
            __splice_builder.jump(__splice_return_label);
        }))
    }

    fn block(&self, block: &Block) -> Result<Value> {
        let mut scope = self.clone();
        let mut prefix = TokenStream::new();
        let mut tail = Value::unit(emit!({}));
        // Rust imports apply to the whole block, including expressions before a return.
        for s in &block.stmts {
            if let Stmt::Item(syn::Item::Use(item)) = s {
                prefix.extend(emit!(#[allow(unused_imports)] #item));
            }
        }
        for (i, s) in block.stmts.iter().enumerate() {
            match s {
                Stmt::Item(syn::Item::Use(_)) => {}
                Stmt::Local(local) => prefix.extend(scope.local(local)?),
                Stmt::Expr(e, semi) => {
                    let v = scope.expr(e)?;
                    if v.stage == Stage::Never || (i + 1 == block.stmts.len() && semi.is_none()) {
                        tail = v;
                        break;
                    }
                    prefix.extend(emit!(#v;));
                }
                _ => return err(s, "expected a let binding or scalar kernel expression"),
            }
        }
        Ok(Value::new(tail.stage, emit!({#prefix #tail})))
    }
    fn local(&mut self, local: &syn::Local) -> Result<TokenStream> {
        let Some(init) = &local.init else {
            return err(local, "initialize kernel locals at their declaration");
        };
        if init.diverge.is_some() || !local.attrs.is_empty() {
            return err(local, "expected a kernel let binding");
        }
        let hint = if let Pat::Type(t) = &local.pat {
            Scalar::from_type(&t.ty)
        } else {
            None
        };
        let value = self.expr_with_hint(&init.expr, hint)?;
        let pat = &local.pat;
        if value.stage.is_host() && !self.flow.locals.contains(&super::flow::key(local)) {
            super::bind_pattern(&mut self.bindings, pat, Binding::Host);
            return Ok(emit!(let #pat = #value;));
        }

        let (pat, ty) = match pat {
            Pat::Type(t) => (&*t.pat, Some(&*t.ty)),
            p => (p, None),
        };
        let name = super::named(pat)?;
        let id = &name.ident;
        let mutable = name.mutability.is_some();
        let (binding, rhs) = if let Stage::Buffer(writable) = value.stage {
            if mutable {
                return err(pat, "a runtime slice binding must be a fixed reference");
            }
            (
                Binding::Buffer(ty.and_then(super::slice_type).unwrap_or(writable)),
                value.code,
            )
        } else {
            let annotation = ty.and_then(Scalar::from_type);
            let scalar = match value.stage {
                Stage::Reg(t) | Stage::Literal(t) => t,
                Stage::Host => match annotation {
                    Some(t @ (Scalar::Int | Scalar::Bool)) => t,
                    _ => Scalar::Inferred,
                },
                _ => return err(pat, "initializer must produce a scalar value"),
            };
            if let Some(ty) = ty
                && scalar != Scalar::Inferred
                && annotation != Some(scalar.base())
            {
                return err(ty, "type annotation differs from the runtime initializer");
            }
            let mut rhs = self.reg(value, scalar, &init.expr)?;
            if mutable {
                rhs = emit!({let value=#rhs;__splice_builder.copy(value)});
            }
            (
                Binding::Reg(if mutable { scalar.base() } else { scalar }, mutable),
                rhs,
            )
        };
        self.bindings.insert(id.to_string(), binding);
        Ok(emit!(let #id = #rhs;))
    }

    fn expr(&self, e: &Expr) -> Result<Value> {
        self.expr_with_hint(e, None)
    }

    fn expr_with_hint(&self, e: &Expr, hint: Option<Scalar>) -> Result<Value> {
        if let Some(v) = self.simd_constructor(e, hint)? {
            return Ok(v);
        }
        if matches!(
            e,
            Expr::Index(_) | Expr::Cast(_) | Expr::MethodCall(_) | Expr::Reference(_)
        ) && !simd::uses_lane(e)
            && let Ok(code) = self.host_raw(e)
        {
            return Ok(Value::new(Stage::Host, code));
        }
        match e {
            Expr::Lit(l) => Ok(Value::new(
                match l.lit {
                    Lit::Int(_) => Stage::Literal(Scalar::Int),
                    Lit::Bool(_) => Stage::Literal(Scalar::Bool),
                    _ => Stage::Host,
                },
                e,
            )),
            Expr::Path(p) => Ok(self.path(p)),
            Expr::Paren(x) => self.expr_with_hint(&x.expr, hint),
            Expr::Group(x) => self.expr_with_hint(&x.expr, hint),
            Expr::Block(x) if x.label.is_none() => self.block(&x.block),
            Expr::If(x) => self.if_expr(x),
            Expr::Match(x) => self.match_expr(x),
            Expr::While(x) if x.label.is_none() => {
                self.staged_loop(e, &x.body, || self.loop_expr(Some(&x.cond), &x.body))
            }
            Expr::Loop(x) if x.label.is_none() => {
                self.staged_loop(e, &x.body, || self.loop_expr(None, &x.body))
            }
            Expr::ForLoop(x) if x.label.is_none() => {
                self.staged_loop(e, &x.body, || self.for_expr(x))
            }
            Expr::Continue(x) if x.label.is_none() => self.loop_jump(e, true),
            Expr::Break(x) if x.label.is_none() && x.expr.is_none() => self.loop_jump(e, false),
            Expr::Return(x) => {
                let Some(arg) = &x.expr else {
                    return err(e, "a kernel must return an i32");
                };
                let ret = self.return_value(self.expr(arg)?, arg)?;
                // A host-known return transfers control in the generator too.
                // The common exit remains the target of earlier runtime returns.
                let region = self.region();
                Ok(Value::new(Stage::Never, emit!({#ret; break #region;})))
            }
            Expr::Binary(x) => self.binary(x, hint),
            Expr::Assign(x) => self.assignment(&x.left, &x.right, None),
            Expr::Unary(x) => {
                let v = self.expr(&x.expr)?;
                let op = &x.op;
                let t = match v.stage {
                    Stage::Reg(t) => t,
                    Stage::Host | Stage::Literal(_) => {
                        let stage = if matches!(op, UnOp::Deref(_)) {
                            Stage::Host
                        } else {
                            v.stage
                        };
                        return Ok(Value::new(stage, emit!(#op (#v))));
                    }
                    _ => return err(x, "expected an operand"),
                };
                if matches!(t, Scalar::Vector(_) | Scalar::Mask(_)) {
                    return self.simd_unary(t, op, v, x);
                }
                let code = match op {
                    UnOp::Neg(_) if t != Scalar::Bool => {
                        self.bin("Sub", emit!(__splice_builder.constant(0)), v.code)
                    }
                    UnOp::Not(_) => self.bin("Xor", v.code, emit!(__splice_builder.constant(-1))),
                    _ => return err(x, "unsupported runtime unary operator"),
                };
                Ok(Value::new(Stage::Reg(t.base()), code))
            }
            Expr::Call(x) => self.call(x),
            Expr::Index(x) => self.index(x),
            Expr::Cast(x) => self.cast(x),
            Expr::MethodCall(x) => self.method(x, hint),
            Expr::Reference(x) => self.reference(x),
            Expr::Macro(m) if m.mac.path.is_ident("comptime") => {
                let expr = m.mac.parse_body::<Expr>()?;
                Ok(Value::new(Stage::Host, self.host_raw(&expr)?))
            }
            Expr::Field(_) | Expr::Tuple(_) => Ok(Value::new(Stage::Host, self.host_raw(e)?)),
            _ => err(
                e,
                "unsupported scalar kernel expression; use comptime!(...) for host calculations",
            ),
        }
    }
    fn assignment(&self, left: &Expr, right: &Expr, op: Option<(&str, &BinOp)>) -> Result<Value> {
        if let Expr::Index(index) = left {
            return self.store(index, right, op);
        }
        let Expr::Path(p) = left else {
            return err(left, "assign to a mutable runtime variable");
        };
        let binding = self.binding(p);
        if matches!(binding, Some(Binding::Host)) {
            let rhs = self.host(self.expr(right)?, right)?;
            let code = if let Some((_, op)) = op {
                emit!(#left #op #rhs)
            } else {
                emit!(#left = #rhs)
            };
            return Ok(Value::unit(code));
        }
        let Some(Binding::Reg(t, true)) = binding else {
            return err(left, "assignment requires a mutable runtime variable");
        };
        let rhs = self.operand(right, t)?;

        let code = if let Some((op, _)) = op {
            if matches!(t, Scalar::Vector(_) | Scalar::Mask(_)) {
                self.simd_operator(t, op, vec![emit!(#left), rhs], left)?
                    .code
            } else {
                self.bin(op, emit!(#left), rhs)
            }
        } else {
            rhs
        };
        Ok(Value::unit(
            emit!({let __splice_assigned_value=#code; __splice_builder.assign(#left,__splice_assigned_value);}),
        ))
    }
    fn binary(&self, x: &syn::ExprBinary, hint: Option<Scalar>) -> Result<Value> {
        use BinOp::*;
        let op = &x.op;
        if let Some(operation) = super::compound_op(op) {
            return self.assignment(&x.left, &x.right, Some((operation, op)));
        }
        let (left, right) = if matches!(op, And(_) | Or(_)) {
            let left = self.expr(&x.left)?;
            let right = if matches!(left.stage, Stage::Reg(_)) {
                self.nested_region().expr(&x.right)?
            } else {
                self.expr(&x.right)?
            };
            (left, right)
        } else {
            self.paired_operands(&x.left, &x.right, hint, None)?
        };
        let t = match (left.stage, right.stage) {
            (Stage::Reg(t), _) | (_, Stage::Reg(t)) => t,
            (Stage::Host | Stage::Literal(_), Stage::Literal(_) | Stage::Host) => {
                return Ok(Value::new(Stage::Host, self.host_binary(left, op, right)));
            }
            _ => return err(x, "expected scalar operands"),
        };
        if matches!(t, Scalar::Vector(_) | Scalar::Mask(_)) {
            return self.simd_binary(t, op, left, right, x);
        }
        let index = [left.stage, right.stage]
            .iter()
            .any(|s| matches!(s, Stage::Reg(Scalar::Index | Scalar::Length)));
        if index && !matches!(op, Eq(_) | Ne(_) | Lt(_) | Le(_) | Gt(_) | Ge(_)) {
            return err(x, "cast the runtime index to i32 before arithmetic");
        }
        let (operation, input, output) = match op {
            Add(_) => ("Add", Scalar::Int, Scalar::Int),
            Sub(_) => ("Sub", Scalar::Int, Scalar::Int),
            Mul(_) => ("Mul", Scalar::Int, Scalar::Int),
            Div(_) => ("Div", Scalar::Int, Scalar::Int),
            Rem(_) => ("Rem", Scalar::Int, Scalar::Int),
            Shl(_) => ("Shl", Scalar::Int, Scalar::Int),
            Shr(_) => ("Shr", Scalar::Int, Scalar::Int),
            BitAnd(_) => ("And", t, t),
            BitOr(_) => ("Or", t, t),
            BitXor(_) => ("Xor", t, t),
            Eq(_) => ("Eq", t, Scalar::Bool),
            Ne(_) => ("Ne", t, Scalar::Bool),
            Lt(_) => ("Lt", Scalar::Int, Scalar::Bool),
            Gt(_) => ("Gt", Scalar::Int, Scalar::Bool),
            Le(_) => ("Le", Scalar::Int, Scalar::Bool),
            Ge(_) => ("Ge", Scalar::Int, Scalar::Bool),
            And(_) | Or(_) => {
                let literal = Value::new(Stage::Literal(Scalar::Bool), emit!(false));
                let true_ = Value::new(Stage::Literal(Scalar::Bool), emit!(true));
                return if matches!(op, And(_)) {
                    self.choice(left, right, literal, x)
                } else {
                    self.choice(left, true_, right, x)
                };
            }
            _ => return err(op, "unsupported runtime operator"),
        };
        let input = if index { Scalar::Index } else { input };
        let mut lhs = self.reg(left, input, &x.left)?;
        let mut rhs = self.reg(right, input, &x.right)?;
        if index && !matches!(op, Eq(_) | Ne(_)) {
            // Flip the sign bit to compare register-width indices as unsigned.
            lhs = self.bin("Xor", lhs, emit!(__splice_builder.constant(i32::MIN)));
            rhs = self.bin("Xor", rhs, emit!(__splice_builder.constant(i32::MIN)));
        }
        Ok(Value::new(
            Stage::Reg(output),
            self.bin(operation, lhs, rhs),
        ))
    }
    fn host_binary(&self, lhs: impl ToTokens, op: &BinOp, rhs: impl ToTokens) -> TokenStream {
        let operation = match op {
            BinOp::Div(_) if self.region_depth > 0 => emit!(Div),
            BinOp::Rem(_) if self.region_depth > 0 => emit!(Rem),
            _ => return emit!((#lhs) #op (#rhs)),
        };
        let (root, region) = (self.root, self.region());
        emit!({
            use #root::frontend::scalar::DivisionGuard as _;
            let operands = (#lhs, #rhs);
            if let Some((a, z)) = (&operands).faulting_i32_division() {
                let (a, z) = (__splice_builder.constant(a), __splice_builder.constant(z));
                let fault = __splice_builder.binary(#root::hir::Binary::#operation, a, z);
                // This arm always faults. Terminate its CFG and stop staging
                // subsequent host code without inventing a result value.
                __splice_builder.ret(fault);
                break #region;
            }
            operands.0 #op operands.1
        })
    }
    fn call(&self, c: &syn::ExprCall) -> Result<Value> {
        let Expr::Path(p) = &*c.func else {
            return err(&c.func, "expected a kernel function path");
        };
        let root = self.root;
        let support = emit!(#root::frontend::scalar);
        let values = c
            .args
            .iter()
            .map(|a| self.expr(a))
            .collect::<Result<Vec<_>>>()?;
        if values.iter().all(|v| v.stage.is_host()) {
            return Ok(Value::new(Stage::Host, emit!(#p(#(#values),*))));
        }
        let recursive = p.path.is_ident(self.name);
        let target = if recursive {
            if !self.formals.iter().any(|p| matches!(p, Binding::Host)) {
                return err(
                    c,
                    "runtime recursion is unsupported; recursive expansion needs a terminating #[comptime] plan",
                );
            }
            if values.len() != self.formals.len() {
                return err(c, "wrong number of recursive arguments");
            }
            let fish = &self.fish;
            emit!(Self::__splice_inline #fish)
        } else {
            let mut path = p.path.clone();
            let fish = std::mem::take(&mut path.segments.last_mut().unwrap().arguments);
            emit!(#path::__splice_call #fish)
        };
        let mut args = Vec::new();
        for (i, (expr, value)) in c.args.iter().zip(values).enumerate() {
            let formal = recursive.then(|| self.formals[i]);
            args.push(match (formal, value.stage) {
                (Some(Binding::Reg(t, _)), _) => self.reg(value, t, expr)?,
                (Some(Binding::Buffer(write)), Stage::Buffer(actual))
                    if write.elem == actual.elem && (!write.write || actual.write) =>
                {
                    value.code
                }
                (Some(Binding::Buffer(_)), _) => {
                    return err(expr, "expected a runtime slice of the required mutability");
                }
                (_, stage) if stage.is_host() => emit!(#support::Host(#value)),
                (Some(Binding::Host), _) => {
                    return err(
                        expr,
                        "expected a comptime value; found a runtime expression",
                    );
                }
                (None, Stage::Reg(Scalar::Int | Scalar::Inferred) | Stage::Buffer(_)) => value.code,
                _ => {
                    return err(
                        expr,
                        "expected a comptime value, a runtime i32, or a runtime slice",
                    );
                }
            });
        }
        Ok(Value::new(
            Stage::Reg(Scalar::Int),
            emit!(#target(#(#args,)* __splice_builder)?),
        ))
    }
}
