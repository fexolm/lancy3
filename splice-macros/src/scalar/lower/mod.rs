//! Lower expressions, calls and bindings while tracking their stage and scalar type.
mod aggregate;
mod calls;
mod comptime;
mod context;
mod control;
pub(super) use comptime::foldable_signature;
pub(crate) use comptime::uses_stage_requirements;
pub(super) use context::Context;
mod host;
pub(crate) use host::overflow_prologue;
mod memory;
pub(crate) mod simd;
mod source;
mod word;
pub(super) use simd::BufferType;
pub(crate) use simd::Elem;

use crate::kast::DeclId;
use proc_macro2::{Ident, TokenStream};
use quote::ToTokens;
use std::collections::HashMap;
use syn::spanned::Spanned;
use syn::{BinOp, Block, Error, Expr, Pat, Path, Result, Stmt, UnOp};

use crate::kast::types::Primitive;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Scalar {
    /// A runtime number other than `usize`, or a `char`: a primitive
    /// [`Primitive::is_kernel_scalar`] accepts. rustc checked the
    /// operations on it; the IR type is `<T as KernelScalar>::TYPE`.
    Number(Primitive),
    Bool,
    /// A `usize`.
    Index,
    /// A usize known to fit a nonnegative i32 (slice length or SIMD lane count).
    Length,
    Inferred,
    Vector(Elem),
    Mask(Elem),
}
impl Scalar {
    pub(in crate::scalar) const I32: Self = Self::Number(Primitive::I32);
    pub(in crate::scalar) fn base(self) -> Self {
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
        Self::of(crate::kast::types::Primitive::of(ty)?)
    }
    /// The runtime scalar of a primitive Rust type.
    pub(in crate::scalar) fn of(primitive: Primitive) -> Option<Self> {
        Some(match primitive {
            Primitive::Bool => Self::Bool,
            Primitive::Usize => Self::Index,
            p if p.is_kernel_scalar() => Self::Number(p),
            _ => return None,
        })
    }
    /// The primitive Rust type of a scalar register.
    pub(in crate::scalar) fn primitive(self) -> Option<Primitive> {
        Some(match self.base() {
            Self::Number(p) => p,
            Self::Bool => Primitive::Bool,
            Self::Index => Primitive::Usize,
            _ => return None,
        })
    }
    /// The Rust type of a scalar register, unless only the builder knows it.
    pub(in crate::scalar) fn rust_type(self) -> Option<Ident> {
        Some(Ident::new(
            self.primitive()?.name(),
            proc_macro2::Span::call_site(),
        ))
    }
    /// A scalar a kernel signature can carry.
    pub(in crate::scalar) fn is_scalar(self) -> bool {
        matches!(
            self,
            Self::Number(_) | Self::Bool | Self::Index | Self::Length
        )
    }
    fn is_float(self) -> bool {
        matches!(self, Self::Number(n) if n.is_float())
    }
    fn is_number(self) -> bool {
        matches!(self, Self::Number(p) if p.is_number())
            || matches!(self, Self::Index | Self::Length)
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
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LoopKind {
    Runtime,
    Unrolled,
}

/// An enclosing loop: its Rust label and the names of its generated IR
/// labels and, when unrolled, of the flags that record which of its exits
/// an iteration reached.
#[derive(Clone)]
pub(super) struct LoopFrame {
    label: Option<String>,
    kind: LoopKind,
    next: Ident,
    end: Ident,
    reaches_next: Ident,
    reaches_end: Ident,
    /// The register of a `loop`'s value and its type, set by the first
    /// `break value` that leaves it.
    value: Ident,
    value_type: std::rc::Rc<std::cell::Cell<Option<Scalar>>>,
}

/// The mandatory evaluator of a body: its exit label, whether it evaluates
/// a wide result, and whether it met a node it cannot run.
pub(super) struct Evaluation {
    exit: syn::Lifetime,
    wide: bool,
    unevaluable: std::cell::Cell<bool>,
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
    /// The lowering of each declaration in scope.
    pub(super) bindings: HashMap<DeclId, Binding>,
    pub(super) aggregates: &'a std::cell::RefCell<Vec<aggregate::Layout>>,
    /// Numbers generated labels in expansion order, so output is deterministic.
    pub(super) labels: &'a std::cell::Cell<u32>,
    /// The declarations a debugger sees, by name.
    pub(super) debug_names: std::collections::BTreeMap<String, DeclId>,
    pub(super) fish: TokenStream,
    /// Enclosing loops, innermost last.
    pub(super) loops: Vec<LoopFrame>,
    pub(super) region_depth: usize,
    /// The label a fault in printed code breaks to: set only by the printer.
    pub(super) source_fault: Option<syn::Lifetime>,
    /// The mandatory evaluator being generated, if any.
    pub(super) evaluation: Option<std::rc::Rc<Evaluation>>,
    pub(super) body: &'a crate::kast::Body<'a>,
    pub(super) context: Option<&'a Context>,
    pub(super) imports: Vec<Vec<syn::ItemUse>>,
    pub(super) local_types: &'a [Ident],
}

/// The IR type of a runtime scalar, or `None` for a vector, a mask, or a
/// scalar whose type only the builder knows.
pub(super) fn scalar_type(root: &Path, t: Scalar) -> Option<TokenStream> {
    let ty = t.rust_type()?;
    Some(emit!(<#ty as #root::KernelScalar>::TYPE))
}

/// The stage of a number literal: its suffix, else the number type its use
/// expects, else Rust's default. A literal of a type registers cannot hold
/// is a host value.
fn literal_stage(suffix: &str, float: bool, hint: Option<Scalar>) -> Stage {
    if !suffix.is_empty() {
        return Primitive::named(suffix)
            .and_then(Scalar::of)
            .map_or(Stage::Host, Stage::Literal);
    }
    Stage::Literal(match hint.map(Scalar::base) {
        Some(t) if t.is_number() && t.is_float() == float => t,
        _ if float => Scalar::Number(Primitive::F64),
        _ => Scalar::I32,
    })
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

impl Lower<'_> {
    /// A label `'__splice_{kind}_{n}` distinct from every other label of
    /// this expansion.
    fn fresh_label(&self, kind: &str) -> syn::Lifetime {
        let n = self.labels.get();
        self.labels.set(n + 1);
        syn::Lifetime::new(
            &format!("'__splice_{kind}_{n}"),
            proc_macro2::Span::mixed_site(),
        )
    }
    /// Enter a loop with Rust `label`; the frame names its generated labels.
    fn enter_loop(&mut self, label: Option<&syn::Label>, kind: LoopKind) -> LoopFrame {
        let n = self.labels.get();
        self.labels.set(n + 1);
        let name = |part: &str| {
            quote::format_ident!(
                "__splice_{part}_{n}",
                span = proc_macro2::Span::mixed_site()
            )
        };
        let frame = LoopFrame {
            label: label.map(|label| label.name.ident.to_string()),
            kind,
            next: name("loop_next"),
            end: name("loop_end"),
            reaches_next: name("reaches_next"),
            reaches_end: name("reaches_end"),
            value: name("loop_value"),
            value_type: std::rc::Rc::default(),
        };
        self.loops.push(frame.clone());
        frame
    }

    /// The loop a `break`/`continue` with `label` leaves, and the frames
    /// it crosses, innermost last.
    fn jump_target(&self, at: &Expr, label: Option<&syn::Lifetime>) -> Result<&[LoopFrame]> {
        let index = match label {
            None => self.loops.len().checked_sub(1),
            Some(label) => self
                .loops
                .iter()
                .rposition(|frame| frame.label.as_deref() == Some(&label.ident.to_string())),
        };
        match index {
            Some(index) => Ok(&self.loops[index..]),
            None => err(at, "break/continue outside a kernel loop"),
        }
    }

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
    /// The IR type of a runtime scalar, or `None` if only the builder knows it.
    fn scalar_type(&self, t: Scalar) -> Option<TokenStream> {
        scalar_type(self.root, t)
    }
    /// A register holding the constant `value`.
    fn constant(&self, value: TokenStream) -> TokenStream {
        let root = self.root;
        emit!(__splice_builder.constant(<#root::hir::Constant as ::core::convert::From<_>>::from(#value))?)
    }
    fn reg(&self, v: Value, want: Scalar, at: impl ToTokens) -> Result<TokenStream> {
        let root = self.root;
        let want = want.base();
        let bits = match (v.stage, want) {
            (Stage::Word(_), _) => {
                return Ok(emit!(#root::frontend::scalar::RunArg::reg(#v, __splice_builder)?));
            }
            (Stage::Reg(t), _)
                if t.base() == want
                    || (!matches!(t, Scalar::Vector(_) | Scalar::Mask(_))
                        && !matches!(want, Scalar::Vector(_) | Scalar::Mask(_))
                        && (t == Scalar::Inferred || want == Scalar::Inferred)) =>
            {
                // An inferred register has the type the builder knows.
                return Ok(v.code);
            }
            (Stage::Literal(t), want)
                if t.base() == want
                    || want == Scalar::Inferred
                    || (t.is_number() && want.is_number() && t.is_float() == want.is_float()) =>
            {
                // rustc gave an unsuffixed literal the type its use expects.
                let ty = if want == Scalar::Inferred { t } else { want };
                let ty = ty.rust_type();
                emit!({let __splice_value: #ty = #v; <#root::hir::Constant as ::core::convert::From<_>>::from(__splice_value)})
            }
            (Stage::Host, Scalar::Inferred) => {
                emit!(#root::frontend::scalar::ScalarConstant::constant(#v))
            }
            (Stage::Host, want) if want.is_scalar() => {
                let ty = want.rust_type();
                emit!(<#root::hir::Constant as ::core::convert::From<_>>::from(*::core::borrow::Borrow::<#ty>::borrow(&(#v))))
            }
            _ => {
                let message = match want {
                    Scalar::Number(n) => format!("expected a runtime {}", n.name()),
                    Scalar::Index | Scalar::Length => "expected a usize slice index".into(),
                    Scalar::Bool => "expected a runtime bool condition".into(),
                    Scalar::Inferred => "expected a runtime scalar".into(),
                    Scalar::Vector(_) | Scalar::Mask(_) => "SIMD operand type mismatch".into(),
                };
                return err(at, &message);
            }
        };
        Ok(emit!({let __splice_constant = #bits; __splice_builder.constant(__splice_constant)?}))
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
                if left.stage == Stage::Reg(Scalar::Inferred) && simd::uses_lane(lhs) {
                    let right = self.expr_with_hint(rhs, hint.or(fallback))?;
                    if let Stage::Reg(t @ (Scalar::Vector(_) | Scalar::Mask(_))) = right.stage {
                        return Ok((self.expr_with_hint(lhs, Some(t))?, right));
                    }
                }
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
        emit!({let (a,z)=(#x,#y); __splice_builder.binary(#root::hir::Binary::#op,a,z)?})
    }
    /// The Rust operator `op`: integer arithmetic checks overflow when the
    /// kernel does.
    fn operator_bin(&self, op: &str, x: TokenStream, y: TokenStream) -> TokenStream {
        self.operator_pair(op, emit!((#x, #y)))
    }
    /// `op` on the operand pair `operands` evaluates to.
    fn operator_pair(&self, op: &str, operands: TokenStream) -> TokenStream {
        let root = self.root;
        let op = Ident::new(op, proc_macro2::Span::call_site());
        let checks = host::kernel_overflow_checks();
        emit!({
            let (a, b) = #operands;
            let checks = #checks;
            #root::frontend::scalar::arithmetic(__splice_builder, #root::hir::Binary::#op, checks, a, b)?
        })
    }
    /// The operand pair of a binary operator whose register operand has a
    /// type only the builder knows and whose other operand is a number
    /// literal, which rustc gave that type.
    fn beside_inferred(&self, left: &Value, right: &Value) -> Option<TokenStream> {
        let root = self.root;
        let like = |literal: &Value| match literal.stage {
            Stage::Literal(t) if t.is_float() => Some(emit!(#root::frontend::scalar::float_like)),
            Stage::Literal(t) if t.is_number() => Some(emit!(#root::frontend::scalar::int_like)),
            _ => None,
        };
        match (left.stage, right.stage) {
            (Stage::Reg(Scalar::Inferred), Stage::Literal(_)) => {
                let like = like(right)?;
                Some(emit!({let a = #left; let z = #like(__splice_builder, a, #right)?; (a, z)}))
            }
            (Stage::Literal(_), Stage::Reg(Scalar::Inferred)) => {
                let like = like(left)?;
                Some(emit!({let z = #right; let a = #like(__splice_builder, z, #left)?; (a, z)}))
            }
            _ => None,
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
                utf8: a.utf8 && b.utf8,
                ..a
            })),
            (Reg(t), Literal(u)) | (Literal(u), Reg(t)) if t == u => Ok(Reg(t)),
            // rustc typed both arms alike: a number literal beside a
            // register has the register's type.
            (Reg(t), Literal(u)) | (Literal(u), Reg(t))
                if t.is_number() && u.is_number() && t.is_float() == u.is_float() =>
            {
                Ok(Reg(t.base()))
            }
            (Reg(Scalar::Inferred), Reg(t) | Literal(t))
            | (Reg(t) | Literal(t), Reg(Scalar::Inferred)) => Ok(Reg(t.base())),
            (Host, Literal(_)) | (Literal(_), Host) => Ok(Host),
            (Host, Reg(t)) | (Reg(t), Host) => Ok(Reg(t.base())),
            _ => err(at, "branch arms have different result types"),
        }
    }
    /// Bind every name `pat` introduces to `binding`.
    pub(super) fn bind(&mut self, pat: &Pat, binding: Binding) {
        super::visit_bindings(pat, &mut |pat| {
            self.bindings.insert(self.body.binding(pat), binding);
        });
    }

    /// Bind the named declaration `pat` and show it to a debugger.
    pub(super) fn declare(&mut self, pat: &syn::PatIdent, binding: Binding) {
        let decl = self.body.binding(pat);
        self.bindings.insert(decl, binding);
        self.debug_names.insert(pat.ident.to_string(), decl);
    }

    fn with_pattern(&self, pat: &Pat) -> Self {
        let mut scope = self.clone();
        scope.bind(pat, Binding::Host(self.region_depth));
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
        super::visit_bindings(pat, &mut |pat| {
            let name = pat.ident.to_string();
            // A bare identifier can name a constant or a variant.
            // A ref binding or an @ binding is explicit.
            if pat.mutability.is_some() || (pat.by_ref.is_none() && pat.subpat.is_none()) {
                self.debug_names.remove(&name);
            } else {
                captures.extend(self.capture_constant(&pat.ident));
                self.debug_names.insert(name, self.body.binding(pat));
            }
        });
        captures
    }

    fn binding(&self, p: &syn::ExprPath) -> Option<Binding> {
        self.bindings.get(&self.body.decl_of(p)?).copied()
    }

    fn path(&self, p: &syn::ExprPath) -> Value {
        self.bound_path(p, self.binding(p))
    }

    /// The value of path `p` bound to `binding`.
    fn bound_path(&self, p: &syn::ExprPath, binding: Option<Binding>) -> Value {
        match binding {
            Some(Binding::Reg(t, true)) => {
                // A later operand can assign this variable. Snapshot its value
                // at the Rust evaluation point before constructing the RHS.
                Value::new(Stage::Reg(t), emit!(__splice_builder.copy(#p)?))
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
            let zero = self.constant(emit!(0i32));
            emit!({#value; #zero})
        } else {
            self.reg(value, self.result_type, at)?
        };
        Ok(emit!({
            let __splice_value = #value;
            __splice_builder.assign(__splice_result, __splice_value)?;
            __splice_builder.jump(__splice_return_label)?;
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
        let root = self.root;
        let (file, directory, line, column) = if proc_macro::is_available() {
            let source = span.unwrap();
            // rustc names a file relative to its working directory, which is
            // also the directory of the compilation unit.
            let directory = std::env::current_dir().unwrap_or_default();
            let file = source.local_file().map_or_else(
                || directory.join(source.file()),
                |file| directory.join(file),
            );
            (
                file.to_string_lossy().into_owned(),
                directory.to_string_lossy().into_owned(),
                source.line() as u32,
                source.column() as u32,
            )
        } else {
            // The macro's token-level unit tests run outside rustc's expansion context.
            (String::new(), String::new(), 0, 0)
        };
        let location = emit!(
            #root::debug::Location {
                file: #file,
                directory: #directory,
                line: #line,
                column: #column,
            }
        );
        let variables = self.debug_names.iter().filter_map(|(name, decl)| {
            let name = name.trim_start_matches("r#");
            let binding = self.bindings.get(decl)?;
            let mut ident = self.body.name(*decl).clone();
            ident.set_span(ident.span().located_at(proc_macro2::Span::mixed_site()));
            let value = match binding {
                Binding::Word(_) => return None,
                Binding::Reg(..) => emit!(#root::debug::Value::Register(#ident)),
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
                __splice_builder.debug_point(#location, &[#(#variables),*])?;
            }
        }
    }
    fn block_with_hint(&self, block: &Block, hint: Option<Scalar>) -> Result<Value> {
        let mut scope = self.clone();
        let mut prefix = TokenStream::new();
        let mut tail = Value::unit(emit!({}));
        if scope.context.is_some() || !scope.local_types.is_empty() {
            scope.imports.push(
                block
                    .stmts
                    .iter()
                    .filter_map(|s| match s {
                        Stmt::Item(syn::Item::Use(item)) => Some(item.clone()),
                        _ => None,
                    })
                    .collect(),
            );
        }
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
        if let Some(literal) = aggregate::Literal::of(self, local, &init.expr) {
            return self.aggregate_local(&local.pat, &literal);
        }
        if let Some(split) = self.split_local(&local.pat, &init.expr)? {
            return Ok(split);
        }
        let hint = match &local.pat {
            Pat::Type(t) => Scalar::from_type(&t.ty, self.types),
            Pat::Ident(_) => self.body.literal_type(local).and_then(Scalar::of),
            _ => None,
        };
        let value = self.expr_with_hint(&init.expr, hint)?;
        let pat = &local.pat;
        if matches!(pat, Pat::Wild(_)) {
            return Ok(emit!(let _ = #value;));
        }
        if value.stage.is_host() && !self.body.runtime_local(local) {
            self.bind(pat, Binding::Host(self.region_depth));
            let (binding, ty) = match pat {
                Pat::Type(t) => (&*t.pat, Some(&*t.ty)),
                p => (p, None),
            };
            if let Pat::Ident(name) = binding
                && name.mutability.is_none()
                && name.by_ref.is_none()
                && name.subpat.is_none()
            {
                // Capture the initializer. An irrefutable pattern can also
                // name a unit variant, so evaluating its identifier as an
                // expression could lose the original type context.
                let id = &name.ident;
                self.debug_names
                    .insert(id.to_string(), self.body.binding(name));
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
                    Some(t) if t.is_scalar() => t.base(),
                    _ => Scalar::Inferred,
                },
                _ => return err(pat, "initializer must produce a scalar value"),
            };
            let mut rhs = self.reg(value, scalar, &init.expr)?;
            if let Some(ty) = ty {
                if let Some(witness) = scalar.vector_witness(ty, self.root, self.types) {
                    // The initializer determines the IR type. Let Rust resolve
                    // the annotation, including aliases and public re-exports.
                    rhs = emit!({ #witness #rhs });
                } else if scalar != Scalar::Inferred && annotation != Some(scalar.base()) {
                    return err(ty, "type annotation differs from the runtime initializer");
                }
            }
            if mutable {
                rhs = emit!({let value=#rhs;__splice_builder.copy(value)?});
            }
            (
                Binding::Reg(if mutable { scalar.base() } else { scalar }, mutable),
                rhs,
            )
        };
        self.declare(name, binding);
        Ok(emit!(let #id = #rhs;))
    }

    /// `let (head, tail) = xs.split_at(mid);`, or `split_at_mut`: two
    /// subslices, after a check that traps unless `mid` is within the length.
    fn split_local(&mut self, pat: &Pat, init: &Expr) -> Result<Option<TokenStream>> {
        let (Pat::Tuple(tuple), Expr::MethodCall(m)) = (pat, init) else {
            return Ok(None);
        };
        let write = m.method == "split_at_mut";
        if !(write || m.method == "split_at") || m.args.len() != 1 {
            return Ok(None);
        }
        let slice = self.expr(&m.receiver)?;
        let Stage::Buffer(actual) = slice.stage else {
            return Ok(None);
        };
        let names: Vec<_> = tuple.elems.iter().collect();
        let [Pat::Ident(head), Pat::Ident(tail)] = names[..] else {
            return err(
                pat,
                "bind the two halves to names: `let (head, tail) = ...`",
            );
        };
        if [head, tail]
            .iter()
            .any(|p| p.mutability.is_some() || p.by_ref.is_some() || p.subpat.is_some())
        {
            return err(pat, "a runtime slice binding must be a fixed reference");
        }
        if write && !actual.write {
            return err(m, "`split_at_mut` needs a writable slice");
        }
        let mid = self.expr(&m.args[0])?;
        let mid = self.reg(mid, Scalar::Index, &m.args[0])?;
        let ty = BufferType { write, ..actual };
        self.declare(head, Binding::Buffer(ty));
        self.declare(tail, Binding::Buffer(ty));
        let (head, tail) = (&head.ident, &tail.ident);
        Ok(Some(emit! {
            let (#head, #tail) = {
                let __splice_slice = #slice;
                let __splice_mid = #mid;
                __splice_builder.split_at(__splice_slice, __splice_mid)?
            };
        }))
    }

    fn expr(&self, e: &Expr) -> Result<Value> {
        self.expr_with_hint(e, None)
    }

    fn expr_with_hint(&self, e: &Expr, hint: Option<Scalar>) -> Result<Value> {
        if let Some(context) = self.context
            && let Some(value) = context.lower(self, e, hint)?
        {
            return Ok(value);
        }
        if let Some(v) = self.qualified_simd(e, hint)? {
            return Ok(v);
        }
        if let Expr::Call(call) = e
            && let Expr::Path(path) = call.func.as_ref()
            && let Some(value) = self.qualified_math(call, path)?
        {
            return Ok(value);
        }
        if self.body.known(e)
            && let Some(value) = self.known_value(e, hint)?
        {
            return Ok(value);
        }
        match e {
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
            Expr::While(x) => self.staged_loop(e, || {
                self.loop_expr(x.label.as_ref(), Some(&x.cond), &x.body)
            }),
            Expr::Loop(x) => {
                self.staged_loop(e, || self.loop_expr(x.label.as_ref(), None, &x.body))
            }
            Expr::ForLoop(x) => self.staged_loop(e, || self.for_expr(x)),
            Expr::Continue(x) => self.loop_jump(e, x.label.as_ref(), None),
            Expr::Break(x) => self.loop_jump(e, x.label.as_ref(), Some(x.expr.as_deref())),
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
                let Stage::Reg(t) = v.stage else {
                    return err(x, "expected an operand");
                };
                if matches!(t, Scalar::Vector(_) | Scalar::Mask(_)) {
                    return self.simd_unary(t, op, v, x);
                }
                let root = self.root;
                let support = emit!(#root::frontend::scalar);
                let code = match op {
                    UnOp::Neg(_) if t != Scalar::Bool => {
                        let checks = host::kernel_overflow_checks();
                        emit!({let (x, checks) = (#v, #checks); #support::negate(__splice_builder, x, checks)?})
                    }
                    UnOp::Not(_) => emit!({let x = #v; #support::not(__splice_builder, x)?}),
                    _ => return err(x, "unsupported runtime unary operator"),
                };
                Ok(Value::new(Stage::Reg(t.base()), code))
            }
            Expr::Call(x) => self.call(x, hint),
            Expr::Index(x) => self.index(x),
            Expr::Cast(x) => self.cast(x),
            Expr::MethodCall(_) => self.method(e, hint),
            Expr::Reference(x) => self.reference(x),
            Expr::Field(field) => match self.aggregate_field(field) {
                Some((value, kind, _)) => Ok(Value::new(Stage::Word(kind), value)),
                // Reports the runtime value the field reads.
                None => Ok(Value::new(Stage::Host, self.host_code(e)?)),
            },
            // Reports the runtime value the tuple holds.
            Expr::Tuple(_) => Ok(Value::new(Stage::Host, self.host_code(e)?)),
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
            if self.body.known(expr) {
                return Ok(Value::new(Stage::Host, self.host_code(expr)?));
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
        let rhs = self.operand(right, t)?;

        let code = if let Some((op, _)) = op {
            if matches!(t, Scalar::Vector(_) | Scalar::Mask(_)) {
                // `x op= y` stages the declared `op_assign` intrinsic.
                let method = match op {
                    "And" => "bitand_assign".to_string(),
                    "Or" => "bitor_assign".to_string(),
                    "Xor" => "bitxor_assign".to_string(),
                    op => format!("{}_assign", op.to_lowercase()),
                };
                self.simd_operator(t, &method, vec![emit!(#left), rhs], left)?
                    .code
            } else {
                self.operator_bin(op, emit!(#left), rhs)
            }
        } else {
            rhs
        };
        Ok(Value::unit(
            emit!({let __splice_assigned_value=#code; __splice_builder.assign(#left,__splice_assigned_value)?;}),
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
            (Stage::Reg(t), Stage::Reg(Scalar::Inferred))
            | (_, Stage::Reg(t))
            | (Stage::Reg(t), _) => t,
            _ => return err(x, "expected scalar operands"),
        };
        if matches!(t, Scalar::Vector(_) | Scalar::Mask(_)) {
            return self.simd_binary(t, op, left, right, x);
        }
        let index = [left.stage, right.stage]
            .iter()
            .any(|s| matches!(s, Stage::Reg(Scalar::Index | Scalar::Length)));
        let number = if index {
            Scalar::Index
        } else if let Scalar::Number(_) | Scalar::Inferred = t {
            t
        } else {
            Scalar::I32
        };
        let t = if index { Scalar::Index } else { t };
        let (operation, input, output) = match op {
            Add(_) => ("Add", number, number),
            Sub(_) => ("Sub", number, number),
            Mul(_) => ("Mul", number, number),
            Div(_) => ("Div", number, number),
            Rem(_) => ("Rem", number, number),
            Shl(_) => ("Shl", number, number),
            Shr(_) => ("Shr", number, number),
            BitAnd(_) => ("And", t, t),
            BitOr(_) => ("Or", t, t),
            BitXor(_) => ("Xor", t, t),
            Eq(_) => ("Eq", t, Scalar::Bool),
            Ne(_) => ("Ne", t, Scalar::Bool),
            Lt(_) => ("Lt", number, Scalar::Bool),
            Gt(_) => ("Gt", number, Scalar::Bool),
            Le(_) => ("Le", number, Scalar::Bool),
            Ge(_) => ("Ge", number, Scalar::Bool),
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
        if input == Scalar::Inferred
            && let Some(operands) = self.beside_inferred(&left, &right)
        {
            return Ok(Value::new(
                Stage::Reg(output),
                self.operator_pair(operation, operands),
            ));
        }
        let lhs = self.reg(left, input, &x.left)?;
        let rhs = self.reg(right, input, &x.right)?;
        Ok(Value::new(
            Stage::Reg(output),
            self.operator_bin(operation, lhs, rhs),
        ))
    }
}
