//! Lower expressions, calls and bindings while tracking their stage and scalar type.
mod control;
mod memory;

use proc_macro2::{Ident, TokenStream};
use quote::{ToTokens, quote};
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
    Inferred,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Stage {
    Host,
    Literal(Scalar),
    Reg(Scalar),
    Buffer(bool),
    Unit,
    Never,
}
#[derive(Clone, Copy)]
pub(super) enum Binding {
    Host,
    Reg(Scalar, bool),
    Buffer(bool),
}
pub(super) struct Value {
    pub(super) stage: Stage,
    pub(super) code: TokenStream,
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
#[derive(Clone)]
pub(super) struct Lower<'a> {
    pub(super) root: &'a Path,
    pub(super) name: &'a Ident,
    pub(super) formals: &'a [Binding],
    pub(super) bindings: HashMap<String, Binding>,
    pub(super) builder: &'a Ident,
    pub(super) result: &'a Ident,
    pub(super) exit: &'a Ident,
    pub(super) fish: TokenStream,
    pub(super) loop_labels: Option<(Ident, Ident)>,
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
            let lhs = binary.left.to_token_stream();
            let rhs = binary.right.to_token_stream();
            let code = self.0.host_binary(lhs, &binary.op, rhs);
            *expr = syn::parse_quote!(#code);
        }
    }
    fn visit_item_mut(&mut self, _: &mut syn::Item) {}
}
struct PatternNames<'a>(&'a mut HashMap<String, Binding>);
impl<'ast> Visit<'ast> for PatternNames<'_> {
    fn visit_pat_ident(&mut self, p: &'ast syn::PatIdent) {
        self.0.insert(p.ident.to_string(), Binding::Host);
        syn::visit::visit_pat_ident(self, p);
    }
}

impl Lower<'_> {
    // Each runtime arm builds independently. A host-selected return, break
    // or continue can skip the rest of that arm without skipping its siblings.
    fn region(&self) -> syn::Lifetime {
        syn::Lifetime::new(
            &format!("'__lancy_arm_{}", self.region_depth),
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
        quote!(#[allow(unused_labels)] #label: { #code; })
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
            return Ok(quote!(#expr));
        }
        Ok(quote!(#e))
    }
    fn host(&self, v: Value, at: impl ToTokens) -> Result<TokenStream> {
        match v.stage {
            Stage::Host | Stage::Literal(_) => Ok(v.code),
            _ => err(at, "expected a comptime value; found a runtime expression"),
        }
    }
    pub(super) fn reg(&self, v: Value, want: Scalar, at: impl ToTokens) -> Result<TokenStream> {
        let b = self.builder;
        let code = v.code;
        match v.stage {
            Stage::Reg(t)
                if t == want
                    || (t != Scalar::Index
                        && want != Scalar::Index
                        && (t == Scalar::Inferred || want == Scalar::Inferred)) =>
            {
                Ok(code)
            }
            Stage::Host | Stage::Literal(Scalar::Int) if want == Scalar::Index => Ok(quote!({
                let n: usize = #code;
                let word = n as i32;
                if word as usize != n {
                    return Err("runtime usize value exceeds the scalar register width".into());
                }
                #b.constant(word)
            })),
            Stage::Literal(Scalar::Int) if want != Scalar::Bool => {
                Ok(quote!({let n=#code;#b.constant(n)}))
            }
            Stage::Literal(Scalar::Bool) if matches!(want, Scalar::Bool | Scalar::Inferred) => {
                Ok(quote!({let n=#code;#b.constant(-i32::from(n))}))
            }
            Stage::Host => Ok(match want {
                Scalar::Index => unreachable!(),
                Scalar::Int => {
                    quote!({let n=*::core::borrow::Borrow::<i32>::borrow(&(#code));#b.constant(n)})
                }
                Scalar::Bool => {
                    quote!({let n=*::core::borrow::Borrow::<bool>::borrow(&(#code));#b.constant(-i32::from(n))})
                }
                Scalar::Inferred => {
                    let root = self.root;
                    quote!({let n=#root::frontend::scalar::ScalarConstant::bits(#code);#b.constant(n)})
                }
            }),
            _ => err(
                at,
                match want {
                    Scalar::Int => "expected a runtime i32",
                    Scalar::Index => "expected a usize slice index",
                    Scalar::Bool => "expected a runtime bool condition",
                    Scalar::Inferred => "expected a runtime scalar",
                },
            ),
        }
    }
    fn bin(&self, op: &str, x: TokenStream, y: TokenStream) -> TokenStream {
        let (b, root) = (self.builder, self.root);
        let op = Ident::new(op, proc_macro2::Span::call_site());
        quote!({let (a,z)=(#x,#y); #b.binary(#root::hir::Binary::#op,a,z)})
    }
    fn stage_join(&self, a: Stage, z: Stage, at: impl ToTokens) -> Result<Stage> {
        use Stage::*;
        match (a, z) {
            (Never, other) | (other, Never) => Ok(other),
            (a, z) if a == z => Ok(a),
            (Buffer(a), Buffer(b)) => Ok(Buffer(a && b)),
            (Reg(t), Literal(u)) | (Literal(u), Reg(t)) if t == u => Ok(Reg(t)),
            (Reg(Scalar::Inferred), Reg(t) | Literal(t))
            | (Reg(t) | Literal(t), Reg(Scalar::Inferred)) => Ok(Reg(t)),
            (Host, Literal(_)) | (Literal(_), Host) => Ok(Host),
            (Host, Reg(t)) | (Reg(t), Host) => Ok(Reg(t)),
            _ => err(at, "branch arms have different result types"),
        }
    }
    fn path(&self, p: &syn::ExprPath) -> Value {
        let binding = p
            .path
            .get_ident()
            .and_then(|i| self.bindings.get(&i.to_string()));
        match binding {
            Some(Binding::Reg(t, true)) => {
                // A later operand can assign this variable. Snapshot its value
                // at the Rust evaluation point before constructing the RHS.
                let b = self.builder;
                let snapshot = super::private("__lancy_snapshot");
                Value::new(
                    Stage::Reg(*t),
                    quote!({let #snapshot=#b.local(); #b.assign(#snapshot,#p);#snapshot}),
                )
            }
            Some(Binding::Reg(t, false)) => Value::new(Stage::Reg(*t), p),
            Some(Binding::Buffer(writable)) => Value::new(Stage::Buffer(*writable), p),
            _ => Value::new(Stage::Host, p),
        }
    }
    pub(super) fn block(&self, block: &Block) -> Result<Value> {
        let mut scope = self.clone();
        let mut prefix = TokenStream::new();
        let mut tail = Value::unit(quote!({}));
        for (i, s) in block.stmts.iter().enumerate() {
            match s {
                Stmt::Local(local) => {
                    let Some(init) = &local.init else {
                        return err(local, "initialize kernel locals at their declaration");
                    };
                    if init.diverge.is_some() || !local.attrs.is_empty() {
                        return err(local, "expected a kernel let binding");
                    }
                    let v = scope.expr(&init.expr)?;
                    let (pat, ty) = match &local.pat {
                        Pat::Type(t) => (&*t.pat, Some(&*t.ty)),
                        p => (p, None),
                    };
                    if matches!(v.stage, Stage::Host | Stage::Literal(_))
                        && !self.flow.locals.contains(&super::flow::key(local))
                    {
                        let mut names = PatternNames(&mut scope.bindings);
                        names.visit_pat(pat);
                        let pat = &local.pat;
                        let code = v.code;
                        prefix.extend(quote!(let #pat=#code;));
                    } else if let Stage::Buffer(writable) = v.stage {
                        let Pat::Ident(p) = pat else {
                            return err(pat, "a slice binding requires a name");
                        };
                        if p.mutability.is_some() || p.subpat.is_some() || p.by_ref.is_some() {
                            return err(p, "a runtime slice binding must be a fixed reference");
                        }
                        let writable = ty.and_then(super::slice_type).unwrap_or(writable);
                        let id = &p.ident;
                        let code = v.code;
                        scope
                            .bindings
                            .insert(id.to_string(), Binding::Buffer(writable));
                        prefix.extend(quote!(let #id = #code;));
                    } else {
                        let Pat::Ident(p) = pat else {
                            return err(pat, "runtime locals require a name");
                        };
                        if p.subpat.is_some() || p.by_ref.is_some() {
                            return err(p, "expected a kernel local name");
                        }
                        let t = match v.stage {
                            Stage::Reg(t) | Stage::Literal(t) => t,
                            Stage::Host => {
                                if ty.is_some_and(|t| crate::scalar_type(t, "bool")) {
                                    Scalar::Bool
                                } else if ty.is_some_and(|t| crate::scalar_type(t, "i32")) {
                                    Scalar::Int
                                } else {
                                    Scalar::Inferred
                                }
                            }
                            _ => return err(p, "initializer must produce a scalar value"),
                        };
                        if let Some(ty) = ty
                            && t != Scalar::Inferred
                            && !crate::scalar_type(
                                ty,
                                match t {
                                    Scalar::Int => "i32",
                                    Scalar::Bool => "bool",
                                    Scalar::Index => "usize",
                                    Scalar::Inferred => unreachable!(),
                                },
                            )
                        {
                            return err(ty, "type annotation differs from the runtime initializer");
                        }
                        let id = &p.ident;
                        let rhs = scope.reg(v, t, &init.expr)?;
                        let mutable = p.mutability.is_some();
                        scope
                            .bindings
                            .insert(id.to_string(), Binding::Reg(t, mutable));
                        let b = self.builder;
                        if mutable {
                            prefix.extend(quote!(let #id={let init=#rhs; let dst=#b.local(); #b.assign(dst,init);dst};));
                        } else {
                            prefix.extend(quote!(let #id=#rhs;));
                        }
                    }
                }
                Stmt::Expr(e, semi) => {
                    let v = scope.expr(e)?;
                    if v.stage == Stage::Never || (i + 1 == block.stmts.len() && semi.is_none()) {
                        tail = v;
                        break;
                    }
                    let code = v.code;
                    prefix.extend(quote!(#code;));
                }
                _ => return err(s, "expected a let binding or scalar kernel expression"),
            }
        }
        let code = tail.code;
        Ok(Value::new(tail.stage, quote!({#prefix #code})))
    }
    fn expr(&self, e: &Expr) -> Result<Value> {
        let b = self.builder;
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
            Expr::Paren(x) => self.expr(&x.expr),
            Expr::Group(x) => self.expr(&x.expr),
            Expr::Block(x) if x.label.is_none() => self.block(&x.block),
            Expr::If(x) => self.if_expr(x),
            Expr::Match(x) => self.match_expr(x),
            Expr::While(x) if x.label.is_none() => {
                if self.flow.loops.contains(&super::flow::key(&x.body)) {
                    self.loop_expr(Some(&x.cond), &x.body)
                } else {
                    self.known_loop(e)
                }
            }
            Expr::Loop(x) if x.label.is_none() => {
                if self.flow.loops.contains(&super::flow::key(&x.body)) {
                    self.loop_expr(None, &x.body)
                } else {
                    self.known_loop(e)
                }
            }
            Expr::ForLoop(x) if x.label.is_none() => {
                if self.flow.loops.contains(&super::flow::key(&x.body)) {
                    self.for_expr(x)
                } else {
                    self.known_loop(e)
                }
            }
            Expr::Continue(x) if x.label.is_none() => self.loop_jump(e, true),
            Expr::Break(x) if x.label.is_none() && x.expr.is_none() => self.loop_jump(e, false),
            Expr::Return(x) => {
                let Some(arg) = &x.expr else {
                    return err(e, "a kernel must return an i32");
                };
                let ret = self.reg(self.expr(arg)?, Scalar::Int, arg)?;
                let (exit, result) = (self.exit, self.result);
                // A host-known return transfers control in the generator too.
                // The common exit remains the target of earlier runtime returns.
                let finish = if self.region_depth == 0 {
                    quote!(#b.mark(#exit); return Ok(#result);)
                } else {
                    let region = self.region();
                    quote!(break #region;)
                };
                Ok(Value::new(
                    Stage::Never,
                    quote!({let value=#ret; #b.assign(#result,value); #b.jump(#exit); #finish}),
                ))
            }
            Expr::Binary(x) => self.binary(x),
            Expr::Assign(x) => self.assignment(&x.left, &x.right, None),
            Expr::Unary(x) => {
                let v = self.expr(&x.expr)?;
                let op = &x.op;
                let t = match v.stage {
                    Stage::Reg(t) => t,
                    Stage::Host | Stage::Literal(_) => {
                        let code = v.code;
                        let stage = if matches!(op, UnOp::Deref(_)) {
                            Stage::Host
                        } else {
                            v.stage
                        };
                        return Ok(Value::new(stage, quote!(#op (#code))));
                    }
                    _ => return err(x, "expected an operand"),
                };
                let code = match op {
                    UnOp::Neg(_) if t != Scalar::Bool => {
                        self.bin("Sub", quote!(#b.constant(0)), v.code)
                    }
                    UnOp::Not(_) => self.bin("Xor", v.code, quote!(#b.constant(-1))),
                    _ => return err(x, "unsupported runtime unary operator"),
                };
                Ok(Value::new(Stage::Reg(t), code))
            }
            Expr::Call(x) => self.call(x),
            Expr::Index(x) => self.index(x),
            Expr::Cast(x) => self.cast(x),
            Expr::MethodCall(x) => self.method(x),
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
        let binding = p
            .path
            .get_ident()
            .and_then(|i| self.bindings.get(&i.to_string()));
        if matches!(binding, Some(Binding::Host)) {
            let rhs = self.host(self.expr(right)?, right)?;
            let code = if let Some((_, op)) = op {
                quote!(#left #op #rhs)
            } else {
                quote!(#left = #rhs)
            };
            return Ok(Value::unit(code));
        }
        let Some(Binding::Reg(t, true)) = binding else {
            return err(left, "assignment requires a mutable runtime variable");
        };
        let rhs = self.reg(self.expr(right)?, *t, right)?;
        let b = self.builder;
        let code = if let Some((op, _)) = op {
            self.bin(op, quote!(#left), rhs)
        } else {
            rhs
        };
        let assigned = super::private("__lancy_assigned_value");
        Ok(Value::unit(
            quote!({let #assigned=#code; #b.assign(#left,#assigned);}),
        ))
    }
    fn binary(&self, x: &syn::ExprBinary) -> Result<Value> {
        use BinOp::*;
        let op = &x.op;
        let assign = match op {
            AddAssign(_) => Some("Add"),
            SubAssign(_) => Some("Sub"),
            MulAssign(_) => Some("Mul"),
            DivAssign(_) => Some("Div"),
            RemAssign(_) => Some("Rem"),
            BitAndAssign(_) => Some("And"),
            BitOrAssign(_) => Some("Or"),
            BitXorAssign(_) => Some("Xor"),
            ShlAssign(_) => Some("Shl"),
            ShrAssign(_) => Some("Shr"),
            _ => None,
        };
        if let Some(operation) = assign {
            return self.assignment(&x.left, &x.right, Some((operation, op)));
        }
        let left = self.expr(&x.left)?;
        let rhs_scope = if matches!(op, And(_) | Or(_)) && matches!(left.stage, Stage::Reg(_)) {
            self.nested_region()
        } else {
            self.clone()
        };
        let right = rhs_scope.expr(&x.right)?;
        let t = match (left.stage, right.stage) {
            (Stage::Reg(t), _) | (_, Stage::Reg(t)) => t,
            (Stage::Host | Stage::Literal(_), Stage::Literal(_) | Stage::Host) => {
                let (lhs, rhs) = (left.code, right.code);
                return Ok(Value::new(Stage::Host, self.host_binary(lhs, op, rhs)));
            }
            _ => return err(x, "expected scalar operands"),
        };
        if left.stage == Stage::Reg(Scalar::Index) || right.stage == Stage::Reg(Scalar::Index) {
            return self.index_comparison(x, left, right);
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
                let literal = Value::new(Stage::Literal(Scalar::Bool), quote!(false));
                let true_ = Value::new(Stage::Literal(Scalar::Bool), quote!(true));
                return if matches!(op, And(_)) {
                    self.choice(left, right, literal, x)
                } else {
                    self.choice(left, true_, right, x)
                };
            }
            _ => return err(op, "unsupported runtime operator"),
        };
        let lhs = self.reg(left, input, &x.left)?;
        let rhs = self.reg(right, input, &x.right)?;
        Ok(Value::new(
            Stage::Reg(output),
            self.bin(operation, lhs, rhs),
        ))
    }
    fn host_binary(&self, lhs: TokenStream, op: &BinOp, rhs: TokenStream) -> TokenStream {
        let operation = match op {
            BinOp::Div(_) if self.region_depth > 0 => quote!(Div),
            BinOp::Rem(_) if self.region_depth > 0 => quote!(Rem),
            _ => return quote!((#lhs) #op (#rhs)),
        };
        let (b, root, region) = (self.builder, self.root, self.region());
        quote!({
            use #root::frontend::scalar::DivisionGuard as _;
            let operands = (#lhs, #rhs);
            if let Some((a, z)) = (&operands).faulting_i32_division() {
                let (a, z) = (#b.constant(a), #b.constant(z));
                let fault = #b.binary(#root::hir::Binary::#operation, a, z);
                // This arm always faults. Terminate its CFG and stop staging
                // subsequent host code without inventing a result value.
                #b.ret(fault);
                break #region;
            }
            operands.0 #op operands.1
        })
    }
    fn call(&self, c: &syn::ExprCall) -> Result<Value> {
        let Expr::Path(p) = &*c.func else {
            return err(&c.func, "expected a kernel function path");
        };
        let (b, root) = (self.builder, self.root);
        let support = quote!(#root::frontend::scalar);
        let mut values = c
            .args
            .iter()
            .map(|a| self.expr(a))
            .collect::<Result<Vec<_>>>()?;
        if values
            .iter()
            .all(|v| matches!(v.stage, Stage::Host | Stage::Literal(_)))
        {
            let host_args = values.into_iter().map(|v| v.code);
            return Ok(Value::new(Stage::Host, quote!(#p(#(#host_args),*))));
        }
        if p.path.is_ident(self.name) {
            if !self.formals.iter().any(|p| matches!(p, Binding::Host)) {
                return err(
                    c,
                    "runtime recursion is unsupported; recursive expansion needs a terminating #[comptime] plan",
                );
            }
            if c.args.len() != self.formals.len() {
                return err(c, "wrong number of recursive arguments");
            }
            let args = c
                .args
                .iter()
                .zip(self.formals)
                .zip(values.drain(..))
                .map(|((a, formal), v)| match formal {
                    Binding::Host => self.host(v, a),
                    Binding::Reg(t, _) => self.reg(v, *t, a),
                    Binding::Buffer(writable) => match v.stage {
                        Stage::Buffer(actual) if !*writable || actual => Ok(v.code),
                        _ => err(a, "expected a runtime slice of the required mutability"),
                    },
                })
                .collect::<Result<Vec<_>>>()?;
            let fish = &self.fish;
            return Ok(Value::new(
                Stage::Reg(Scalar::Int),
                quote!(Self::__lancy_inline #fish (#(#args,)* #b)?),
            ));
        }
        let mut target = p.path.clone();
        let part = target.segments.last_mut().unwrap();
        let fish = std::mem::take(&mut part.arguments);
        let args = c
            .args
            .iter()
            .zip(values)
            .map(|(a, v)| {
                let code = v.code;
                match v.stage {
                    Stage::Host | Stage::Literal(_) => Ok(quote!(#support::Host(#code))),
                    Stage::Reg(Scalar::Int | Scalar::Inferred) | Stage::Buffer(_) => Ok(code),
                    _ => err(
                        a,
                        "expected a comptime value, a runtime i32, or a runtime slice",
                    ),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Value::new(
            Stage::Reg(Scalar::Int),
            quote!(#target::__lancy_call #fish (#(#args,)* #b)?),
        ))
    }
}
