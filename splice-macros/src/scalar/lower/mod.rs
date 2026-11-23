//! Lower expressions, calls and bindings while tracking their stage and scalar type.
mod aggregate;
mod calls;
mod comptime;
mod context;
mod control;
pub(crate) use calls::select_free;
pub(super) use comptime::can_fold;
pub(crate) use comptime::uses_stage_requirements;
pub(super) use context::Context;
mod host;
use host::guarded_negation;
mod memory;
mod owned;
pub(super) mod simd;
mod source;
mod word;
pub(super) use simd::BufferType;
pub(crate) use simd::Elem;

use proc_macro2::{Ident, TokenStream};
use quote::ToTokens;
use std::collections::HashMap;
use syn::spanned::Spanned;
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
    pub(super) fn from_type(ty: &syn::Type, types: &[&Ident]) -> Option<Self> {
        if let Some(t) = simd::vector_type(ty, types) {
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
    Word(Scalar),
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
            Self::Literal(t) | Self::Word(t) => Self::Reg(t),
            Self::Host => Self::Reg(Scalar::Inferred),
            stage => stage,
        }
    }
}
#[derive(Clone, Copy)]
pub(super) enum Binding {
    Aggregate(usize),
    Context,
    Word(Scalar),
    Host(usize), // Region in which the Rust binding was introduced.
    Columns,
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
    pub(super) types: &'a [&'a Ident],
    pub(super) name: &'a Ident,
    pub(super) generics: &'a syn::Generics,
    pub(super) result_type: Scalar,
    pub(super) unit_result: bool,
    pub(super) formals: &'a [Binding],
    pub(super) bindings: HashMap<String, Binding>,
    pub(super) aggregates: &'a std::cell::RefCell<Vec<aggregate::Layout>>,
    pub(super) debug_names: std::collections::BTreeMap<String, Ident>,
    pub(super) fish: TokenStream,
    pub(super) loop_kind: Option<LoopKind>,
    pub(super) region_depth: usize,
    pub(super) source_fault: Option<syn::Lifetime>,
    pub(super) evaluator_return: Option<syn::Lifetime>,
    pub(super) flow: &'a super::flow::Flow,
    pub(super) context: Option<&'a Context>,
}

fn err<T>(at: impl ToTokens, msg: &str) -> Result<T> {
    Err(Error::new_spanned(at, msg))
}

fn checked_operator(op: &str) -> Option<&'static str> {
    Some(match op {
        "Add" => "CheckedAdd",
        "Sub" => "CheckedSub",
        "Mul" => "CheckedMul",
        "Shl" => "CheckedShl",
        "Shr" => "CheckedShr",
        _ => return None,
    })
}

fn vec_macro(path: &Path) -> bool {
    path.is_ident("vec")
        || (path.segments.len() == 2
            && (path.segments[0].ident == "std" || path.segments[0].ident == "alloc")
            && path.segments[1].ident == "vec")
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
            (Stage::Word(_), _) => {
                return Ok(emit!(#root::frontend::scalar::RunArg::reg(#v,__splice_builder)));
            }
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
                    return ::core::result::Result::Err(#root::BuildError::staging("runtime usize value exceeds the scalar register width"));
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
    fn operator_bin(&self, op: &str, x: TokenStream, y: TokenStream) -> TokenStream {
        let Some(checked) = checked_operator(op) else {
            return self.bin(op, x, y);
        };
        match crate::overflow::explicit_mode() {
            Some(mode) => self.bin(if mode { checked } else { op }, x, y),
            None => {
                let root = self.root;
                let op = Ident::new(op, proc_macro2::Span::call_site());
                let checked = Ident::new(checked, proc_macro2::Span::call_site());
                emit!({
                    let (a, b) = (#x, #y);
                    let op = if ::core::cfg!(debug_assertions) {
                        #root::hir::Binary::#checked
                    } else {
                        #root::hir::Binary::#op
                    };
                    __splice_builder.binary(op, a, b)
                })
            }
        }
    }
    fn stage_join(&self, a: Stage, z: Stage, at: impl ToTokens) -> Result<Stage> {
        use Stage::*;
        match (a, z) {
            (Never, other) | (other, Never) => Ok(other),
            (a, z) if a == z => Ok(a),
            (Word(t), Host | Literal(_)) | (Host | Literal(_), Word(t)) => Ok(Word(t)),
            (Word(t), Reg(_)) | (Reg(_), Word(t)) => Ok(Reg(t)),
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
        super::bind_pattern(&mut scope.bindings, pat, Binding::Host(self.region_depth));
        super::visit_bindings(pat, &mut |pat| {
            scope.debug_names.remove(&pat.ident.to_string());
        });
        scope
    }

    fn with_debug_pattern(&self, pat: &Pat) -> (Self, TokenStream) {
        let mut scope = self.with_pattern(pat);
        let captures = scope.capture_pattern(pat);
        (scope, captures)
    }

    fn capture_pattern(&mut self, pat: &Pat) -> TokenStream {
        let mut captures = TokenStream::new();
        if self.source_fault.is_none() {
            super::visit_bindings(pat, &mut |pat| {
                let name = pat.ident.to_string();
                // A bare identifier can name a constant or a variant.
                // A ref binding or an @ binding is explicit.
                if pat.mutability.is_some() || (pat.by_ref.is_none() && pat.subpat.is_none()) {
                    self.debug_names.remove(&name);
                } else {
                    captures.extend(self.capture_constant(&pat.ident));
                    self.debug_names.insert(name, pat.ident.clone());
                }
            });
        }
        captures
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
            Some(Binding::Word(t)) => Value::new(Stage::Word(t), p),
            Some(Binding::Buffer(writable)) => Value::new(Stage::Buffer(writable), p),
            Some(Binding::Context | Binding::Aggregate(_)) => {
                let root = self.root;
                Value::new(
                    Stage::Reg(Scalar::Inferred),
                    emit! {
                        ::core::result::Result::Err::<#root::hir::Reg, #root::BuildError>(#root::BuildError::staging("a runtime Rust object cannot be used as a scalar operand"))?
                    },
                )
            }
            _ => Value::new(Stage::Host, p),
        }
    }
    pub(super) fn kernel_body(&self, block: &Block) -> Result<TokenStream> {
        let body = self.block_with_hint(block, Some(self.result_type))?;
        let code = if body.stage == Stage::Never {
            body.code
        } else {
            self.return_value(body, block)?
        };
        Ok(self.delimit(code))
    }

    fn return_value(&self, value: Value, at: impl ToTokens) -> Result<TokenStream> {
        let value = if self.unit_result && value.stage == Stage::Unit {
            emit!({#value; __splice_builder.constant(0)})
        } else {
            self.reg(value, self.result_type, at)?
        };
        Ok(emit!({
            let __splice_value = #value;
            __splice_builder.assign(__splice_result, __splice_value);
            __splice_builder.jump(__splice_return_label);
        }))
    }

    fn block(&self, block: &Block) -> Result<Value> {
        self.block_with_hint(block, None)
    }

    fn constant_name(ident: &Ident) -> Ident {
        quote::format_ident!(
            "__splice_debug_{}",
            ident.to_string().trim_start_matches("r#"),
            span = proc_macro2::Span::mixed_site()
        )
    }

    pub(super) fn capture_constant(&self, ident: &Ident) -> TokenStream {
        self.capture_value(ident, emit!(#ident))
    }

    fn capture_value(&self, ident: &Ident, value: TokenStream) -> TokenStream {
        let variable = Self::constant_name(ident);
        let root = self.root;
        emit! {
            let #variable = if __splice_wants_debug {
                #[allow(unused_imports)]
                use #root::debug::SnapshotConstant as _;
                (&#root::debug::ConstantArgument(&#value)).snapshot()
            } else { ::core::option::Option::None };
        }
    }
    fn debug_point(&self, span: proc_macro2::Span) -> TokenStream {
        if self.source_fault.is_some() {
            return TokenStream::new();
        }
        let root = self.root;
        let (file, line, column) = if proc_macro::is_available() {
            let source = span.unwrap();
            (source.file(), source.line() as u32, source.column() as u32)
        } else {
            // The macro's token-level unit tests run outside rustc's expansion context.
            (String::new(), 0, 0)
        };
        let location = emit!(
            #root::debug::Location {
                file: #file,
                directory: ::core::env!("CARGO_MANIFEST_DIR"),
                line: #line,
                column: #column,
            }
        );
        let variables = self.debug_names.iter().filter_map(|(name, ident)| {
            let name = name.trim_start_matches("r#");
            let binding = self.bindings.get(&ident.to_string())?;
            let mut ident = ident.clone();
            ident.set_span(ident.span().located_at(proc_macro2::Span::mixed_site()));
            let value = match binding {
                Binding::Word(_) => return None,
                Binding::Reg(kind, _) => {
                    let ty = match kind {
                        Scalar::Bool => emit!(Bool),
                        Scalar::Index | Scalar::Length => emit!(Index),
                        _ => emit!(I32),
                    };
                    emit!(#root::debug::Value::Register(#ident, #root::debug::Scalar::#ty))
                }
                Binding::Buffer(_) => emit!(#root::debug::Value::Slice(#ident)),
                Binding::Host(_) => {
                    let constant = Self::constant_name(&ident);
                    emit!(#root::debug::Value::Constant(#constant))
                }
                Binding::Columns | Binding::Context | Binding::Aggregate(_) => return None,
            };
            Some(emit!(#root::debug::Variable { name: #name, value: #value }))
        });
        emit! {
            if __splice_wants_debug {
                __splice_builder.debug_point(#location, &[#(#variables),*]);
            }
        }
    }
    fn block_with_hint(&self, block: &Block, hint: Option<Scalar>) -> Result<Value> {
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
                Stmt::Local(local) => {
                    prefix.extend(scope.debug_point(local.span()));
                    prefix.extend(scope.local(local)?);
                }
                Stmt::Expr(e, semi) => {
                    prefix.extend(scope.debug_point(e.span()));
                    let last = i + 1 == block.stmts.len() && semi.is_none();
                    let v = scope.expr_with_hint(e, if last { hint } else { None })?;
                    if v.stage == Stage::Never || last {
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
        if let Expr::Struct(literal) = &*init.expr {
            return self.aggregate_local(&local.pat, literal);
        }
        let hint = if let Pat::Type(t) = &local.pat {
            Scalar::from_type(&t.ty, self.types)
        } else {
            None
        };
        let value = self.expr_with_hint(&init.expr, hint)?;
        let pat = &local.pat;
        if matches!(pat, Pat::Wild(_)) {
            return Ok(emit!(let _ = #value;));
        }
        if value.stage.is_host() && !self.flow.locals.contains(&super::flow::key(local)) {
            super::bind_pattern(&mut self.bindings, pat, Binding::Host(self.region_depth));
            let (binding, ty) = match pat {
                Pat::Type(t) => (&*t.pat, Some(&*t.ty)),
                p => (p, None),
            };
            if self.source_fault.is_none()
                && let Pat::Ident(name) = binding
                && name.mutability.is_none()
                && name.by_ref.is_none()
                && name.subpat.is_none()
            {
                // Capture the initializer. An irrefutable pattern can also
                // name a unit variant, so evaluating its identifier as an
                // expression could lose the original type context.
                let id = &name.ident;
                self.debug_names.insert(id.to_string(), id.clone());
                let ty = ty.map(|ty| emit!(: #ty));
                let capture = self.capture_value(id, emit!(__splice_host_value));
                return Ok(emit! {
                    let __splice_host_value #ty = #value;
                    #capture
                    let #pat = __splice_host_value;
                });
            }
            let constants = self.capture_pattern(pat);
            return Ok(emit!(let #pat = #value; #constants));
        }

        let (pat, ty) = match pat {
            Pat::Type(t) => (&*t.pat, Some(&*t.ty)),
            p => (p, None),
        };
        let name = super::named(pat)?;
        let id = &name.ident;
        let mutable = name.mutability.is_some();
        let (binding, rhs) = if let Stage::Word(t) = value.stage
            && !mutable
        {
            (Binding::Word(t), value.code)
        } else if let Stage::Buffer(writable) = value.stage {
            if mutable {
                return err(pat, "a runtime slice binding must be a fixed reference");
            }
            (
                Binding::Buffer(
                    ty.and_then(|ty| super::slice_type(ty, self.types))
                        .unwrap_or(writable),
                ),
                value.code,
            )
        } else {
            let annotation = ty.and_then(|ty| Scalar::from_type(ty, self.types));
            let scalar = match value.stage {
                Stage::Reg(t) | Stage::Word(t) | Stage::Literal(t) => t,
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
        self.debug_names.insert(id.to_string(), id.clone());
        Ok(emit!(let #id = #rhs;))
    }

    fn expr(&self, e: &Expr) -> Result<Value> {
        self.expr_with_hint(e, None)
    }

    fn expr_with_hint(&self, e: &Expr, hint: Option<Scalar>) -> Result<Value> {
        if self.source_fault.is_none()
            && let Some(context) = self.context
            && let Some(value) = context.lower(self, e, hint)?
        {
            return Ok(value);
        }
        if let Some(v) = self.simd_constructor(e, hint)? {
            return Ok(v);
        }
        if matches!(e, Expr::Reference(_))
            && !simd::uses_lane(e)
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
            Expr::Path(p) if matches!(self.binding(p), Some(Binding::Columns)) => err(
                p,
                "use columns[comptime_index].as_slice::<T>() to bind a typed column",
            ),
            Expr::Path(p) => Ok(self.path(p)),
            Expr::Paren(x) => self.expr_with_hint(&x.expr, hint),
            Expr::Group(x) => self.expr_with_hint(&x.expr, hint),
            Expr::Block(x) if x.label.is_none() => self.block_with_hint(&x.block, hint),
            Expr::If(x) => self.if_expr(x, hint),
            Expr::Match(x) => self.match_expr(x, hint),
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
                let value = match &x.expr {
                    Some(arg) => self.expr_with_hint(arg, Some(self.result_type))?,
                    None if self.unit_result => Value::unit(emit!(())),
                    None => return err(e, "a kernel return requires a value"),
                };
                let ret = self.return_value(value, e)?;
                // A host-known return transfers control in the generator too.
                // The common exit remains the target of earlier runtime returns.
                let region = self.region();
                Ok(Value::new(Stage::Never, emit!({#ret; break #region;})))
            }
            Expr::Binary(x) => match super::compound_op(&x.op) {
                Some(op) => self.assignment(e, &x.left, &x.right, Some((op, &x.op))),
                None => self.binary(x, hint),
            },
            Expr::Assign(x) => self.assignment(e, &x.left, &x.right, None),
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
                        let code = if matches!(op, UnOp::Deref(_)) {
                            self.known_deref(v)
                        } else if matches!(op, UnOp::Not(_)) {
                            self.host_not(v)
                        } else if guarded_negation(x) {
                            self.host_negation(v)
                        } else {
                            emit!(#op (#v))
                        };
                        return Ok(Value::new(stage, code));
                    }
                    _ => return err(x, "expected an operand"),
                };
                if matches!(t, Scalar::Vector(_) | Scalar::Mask(_)) {
                    return self.simd_unary(t, op, v, x);
                }
                let code = match op {
                    UnOp::Neg(_) if t != Scalar::Bool => {
                        self.operator_bin("Sub", emit!(__splice_builder.constant(0)), v.code)
                    }
                    UnOp::Not(_) => self.bin("Xor", v.code, emit!(__splice_builder.constant(-1))),
                    _ => return err(x, "unsupported runtime unary operator"),
                };
                Ok(Value::new(Stage::Reg(t.base()), code))
            }
            Expr::Call(x) => self.call(x, hint, None),
            Expr::Index(x) => self.index(x),
            Expr::Cast(x) => self.cast(x),
            Expr::MethodCall(x) => self.method(x, hint),
            Expr::Reference(x) => self.reference(x),
            Expr::Macro(m) if vec_macro(&m.mac.path) => {
                Ok(Value::new(Stage::Host, self.host_raw(e)?))
            }
            Expr::Macro(m) if m.mac.path.is_ident("comptime") => {
                let expr = m.mac.parse_body::<Expr>()?;
                Ok(Value::new(Stage::Host, self.required(&expr, hint)?))
            }
            Expr::Field(field) => match self.aggregate_field(field) {
                Some((value, kind, _)) => Ok(Value::new(Stage::Word(kind), value)),
                None => Ok(Value::new(Stage::Host, self.host_raw(e)?)),
            },
            Expr::Tuple(_) => Ok(Value::new(Stage::Host, self.host_raw(e)?)),
            _ => err(
                e,
                "unsupported scalar kernel expression; use comptime!(...) for host calculations",
            ),
        }
    }
    fn assignment(
        &self,
        expr: &Expr,
        left: &Expr,
        right: &Expr,
        op: Option<(&str, &BinOp)>,
    ) -> Result<Value> {
        if let Expr::Index(index) = left {
            if !simd::uses_lane(expr)
                && let Ok(code) = self.host_raw(expr)
            {
                return Ok(Value::new(Stage::Host, code));
            }
            return self.store(index, right, op);
        }
        let Expr::Path(p) = left else {
            return err(left, "assign to a mutable runtime variable");
        };
        let binding = self.binding(p);
        if matches!(binding, Some(Binding::Host(_))) {
            let rhs = self.host(self.expr(right)?, right)?;
            let code = if let Some((kind, op)) = op {
                self.host_compound(left, rhs, op, kind)
            } else {
                emit!(#left = #rhs)
            };
            return Ok(Value::unit(code));
        }
        let Some(Binding::Reg(t, true)) = binding else {
            return err(left, "assignment requires a mutable runtime variable");
        };
        if op.is_some() && matches!(t, Scalar::Index | Scalar::Length) {
            // Registers hold usize in 32 bits with i32 checks; see `binary`.
            return err(expr, "cast the runtime index to i32 before arithmetic");
        }
        let rhs = self.operand(right, t)?;

        let code = if let Some((op, _)) = op {
            if matches!(t, Scalar::Vector(_) | Scalar::Mask(_)) {
                self.simd_operator(t, op, vec![emit!(#left), rhs], left)?
                    .code
            } else {
                self.operator_bin(op, emit!(#left), rhs)
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
        if matches!(left.stage, Stage::Word(_)) || matches!(right.stage, Stage::Word(_)) {
            return self.word_binary(x, left, right);
        }
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
            self.operator_bin(operation, lhs, rhs),
        ))
    }
}
