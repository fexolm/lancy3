//! A host expression keeps its Rust type. A row expression produces a symbolic
//! operand, tagged here as an integer or a comparison mask.
use proc_macro2::{Ident, TokenStream};
use quote::{ToTokens, quote};
use std::collections::HashMap;
use syn::visit::Visit;
use syn::{BinOp, Block, Error, Expr, Pat, Path, Result, Stmt, UnOp};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Lane {
    Int,
    Bool,
}

pub(super) struct Value {
    code: TokenStream,
    lane: Option<Lane>,
}

impl Value {
    fn host(code: impl ToTokens) -> Self {
        Self {
            code: code.into_token_stream(),
            lane: None,
        }
    }

    fn row(code: TokenStream, lane: Lane) -> Self {
        Self {
            code,
            lane: Some(lane),
        }
    }

    fn require_host(self, at: impl ToTokens, message: &str) -> Result<TokenStream> {
        if self.lane.is_some() {
            Err(Error::new_spanned(at, message))
        } else {
            Ok(self.code)
        }
    }

    pub(super) fn operand(self, want: Lane, root: &Path, at: impl ToTokens) -> Result<TokenStream> {
        let code = self.code;
        match self.lane {
            Some(actual) if actual == want => Ok(code),
            Some(_) => Err(Error::new_spanned(
                at,
                match want {
                    Lane::Int => "expected a row i32, found a boolean mask",
                    Lane::Bool => "expected a boolean condition, found a row i32",
                },
            )),
            None => Ok(match want {
                Lane::Int => quote!(#root::frontend::batch::Operand::integer(#code)),
                Lane::Bool => quote!(#root::frontend::batch::Operand::boolean(#code)),
            }),
        }
    }

    fn join(self, lane: Option<Lane>, root: &Path, at: impl ToTokens) -> Result<TokenStream> {
        match lane {
            Some(lane) => self.operand(lane, root, at),
            None => Ok(self.code),
        }
    }
}

#[derive(Clone)]
pub(super) struct Lower<'a> {
    root: &'a Path,
    name: &'a Ident,
    ctx: &'a Ident,
    helper: &'a TokenStream,
    host_params: &'a [bool],
    bindings: HashMap<String, Option<Lane>>,
}

// Every name introduced by a host pattern shadows the surrounding stage,
// including bindings inside tuples, struct variants and or-patterns.
struct Bind<'a> {
    bindings: &'a mut HashMap<String, Option<Lane>>,
    lane: Option<Lane>,
}

impl<'ast> Visit<'ast> for Bind<'_> {
    fn visit_pat_ident(&mut self, p: &'ast syn::PatIdent) {
        self.bindings.insert(p.ident.to_string(), self.lane);
        syn::visit::visit_pat_ident(self, p);
    }
}

impl<'a> Lower<'a> {
    pub(super) fn new(
        root: &'a Path,
        name: &'a Ident,
        ctx: &'a Ident,
        helper: &'a TokenStream,
        host_params: &'a [bool],
        bindings: HashMap<String, Option<Lane>>,
    ) -> Self {
        Self {
            root,
            name,
            ctx,
            helper,
            host_params,
            bindings,
        }
    }

    fn bind(&mut self, pat: &Pat, lane: Option<Lane>) {
        Bind {
            bindings: &mut self.bindings,
            lane,
        }
        .visit_pat(pat);
    }

    fn host(&self, e: &Expr) -> Result<TokenStream> {
        self.expr(e)?.require_host(
            e,
            "expected a comptime Rust value here, found a runtime row expression",
        )
    }

    pub(super) fn block(&self, block: &Block) -> Result<Value> {
        let mut scope = self.clone();
        let mut prefix = TokenStream::new();
        let mut tail = Value::host(quote!(()));
        for (i, stmt) in block.stmts.iter().enumerate() {
            match stmt {
                Stmt::Local(local) => {
                    let Some(init) = &local.init else {
                        return Err(Error::new_spanned(
                            local,
                            "kernel let bindings require an initializer",
                        ));
                    };
                    if init.diverge.is_some() || !local.attrs.is_empty() {
                        return Err(Error::new_spanned(
                            local,
                            "use a plain let binding or a comptime match",
                        ));
                    }
                    let val = scope.expr(&init.expr)?;
                    let mut pat = &local.pat;
                    if let Some(lane) = val.lane {
                        if let Pat::Type(typed) = pat {
                            let ty = if lane == Lane::Int { "i32" } else { "bool" };
                            if !crate::scalar_type(&typed.ty, ty) {
                                return Err(Error::new_spanned(
                                    typed,
                                    format!("expected a runtime {ty}"),
                                ));
                            }
                            pat = &typed.pat;
                        }
                        if !matches!(pat, Pat::Ident(p) if p.by_ref.is_none() && p.mutability.is_none() && p.subpat.is_none())
                            && !matches!(pat, Pat::Wild(_))
                        {
                            return Err(Error::new_spanned(
                                pat,
                                "use an immutable name for a runtime value",
                            ));
                        }
                    }
                    scope.bind(pat, val.lane);
                    let code = val.code;
                    prefix.extend(quote!(let #pat = #code;));
                }
                Stmt::Expr(expr, semi) => {
                    let value = scope.expr(expr)?;
                    if i + 1 == block.stmts.len() && semi.is_none() {
                        tail = value;
                    } else {
                        let code = value.code;
                        prefix.extend(quote!(#code;));
                    }
                }
                _ => {
                    return Err(Error::new_spanned(
                        stmt,
                        "expected a let binding or a kernel expression",
                    ));
                }
            }
        }
        let code = tail.code;
        Ok(Value {
            code: quote!({ #prefix #code }),
            lane: tail.lane,
        })
    }

    fn expr(&self, e: &Expr) -> Result<Value> {
        match e {
            Expr::Path(p) => {
                let lane = p
                    .path
                    .get_ident()
                    .and_then(|i| self.bindings.get(&i.to_string()))
                    .copied()
                    .flatten();
                Ok(Value {
                    code: quote!(#p),
                    lane,
                })
            }
            Expr::Lit(_) => Ok(Value::host(e)),
            Expr::Paren(p) => self.expr(&p.expr),
            Expr::Group(g) => self.expr(&g.expr),
            Expr::Block(b) if b.label.is_none() => self.block(&b.block),
            Expr::Binary(b) => self.binary(b),
            Expr::Unary(u) => self.unary(u),
            Expr::Match(m) => self.match_expr(m),
            Expr::If(i) => self.if_expr(i),
            Expr::Call(c) => self.call(c),
            Expr::Field(f) => {
                let base = self.host(&f.base)?;
                let member = &f.member;
                Ok(Value::host(quote!((#base).#member)))
            }
            Expr::Reference(r) if r.mutability.is_none() => {
                let value = self.host(&r.expr)?;
                Ok(Value::host(quote!(&(#value))))
            }
            Expr::Index(i) => {
                let base = self.host(&i.expr)?;
                let index = self.host(&i.index)?;
                Ok(Value::host(quote!((#base)[#index])))
            }
            Expr::MethodCall(m) => {
                let receiver = self.host(&m.receiver)?;
                let args = m
                    .args
                    .iter()
                    .map(|a| self.host(a))
                    .collect::<Result<Vec<_>>>()?;
                let method = &m.method;
                let generics = &m.turbofish;
                Ok(Value::host(
                    quote!((#receiver).#method #generics (#(#args),*)),
                ))
            }
            Expr::Cast(c) => {
                let value = self.host(&c.expr)?;
                let ty = &c.ty;
                Ok(Value::host(quote!((#value) as #ty)))
            }
            Expr::Tuple(t) => {
                let fields = t
                    .elems
                    .iter()
                    .map(|v| self.host(v))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Value::host(quote!((#(#fields,)*))))
            }
            _ => Err(Error::new_spanned(
                e,
                "unsupported kernel expression; put host-only logic in a Rust helper",
            )),
        }
    }

    fn call(&self, c: &syn::ExprCall) -> Result<Value> {
        let recursive = matches!(&*c.func, Expr::Path(p) if p.path.is_ident(self.name))
            && !self.bindings.contains_key(&self.name.to_string());
        if !recursive {
            let func = self.host(&c.func)?;
            let args = c
                .args
                .iter()
                .map(|a| self.host(a))
                .collect::<Result<Vec<_>>>()?;
            return Ok(Value::host(quote!((#func)(#(#args),*))));
        }
        if !self.host_params.contains(&true) {
            return Err(Error::new_spanned(
                c,
                "recursive expansion needs a terminating #[comptime] argument",
            ));
        }
        if c.args.len() != self.host_params.len() {
            return Err(Error::new_spanned(
                c,
                "recursive calls use the original kernel signature",
            ));
        }
        let args = c
            .args
            .iter()
            .zip(self.host_params)
            .map(|(arg, &host)| {
                if host {
                    self.host(arg)
                } else {
                    self.expr(arg)?.operand(Lane::Int, self.root, arg)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let helper = self.helper;
        let ctx = self.ctx;
        Ok(Value::row(quote!(#helper(#(#args,)* #ctx)), Lane::Int))
    }

    fn match_expr(&self, m: &syn::ExprMatch) -> Result<Value> {
        let scrutinee = self.expr(&m.expr)?.require_host(
            &m.expr,
            "match requires a #[comptime] value; use if for a runtime row condition",
        )?;
        let mut arms = Vec::new();
        let mut result_lane = None;
        for arm in &m.arms {
            let mut scope = self.clone();
            scope.bind(&arm.pat, None);
            let guard = arm
                .guard
                .as_ref()
                .map(|(_, e)| scope.host(e).map(|g| quote!(if #g)))
                .transpose()?;
            let body = scope.expr(&arm.body)?;
            result_lane = result_lane.or(body.lane);
            arms.push((arm, guard, body));
        }
        let arms = arms
            .into_iter()
            .map(|(arm, guard, body)| {
                let pat = &arm.pat;
                let attrs = &arm.attrs;
                let code = body.join(result_lane, self.root, &arm.body)?;
                Ok(quote!(#(#attrs)* #pat #guard => #code,))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Value {
            code: quote!(match #scrutinee { #(#arms)* }),
            lane: result_lane,
        })
    }

    fn if_expr(&self, i: &syn::ExprIf) -> Result<Value> {
        let cond = self.expr(&i.cond)?;
        let yes = self.block(&i.then_branch)?;
        let Some((_, otherwise)) = &i.else_branch else {
            let test = cond.require_host(&i.cond, "a runtime if must have an else value")?;
            let yes = yes.code;
            return Ok(Value::host(quote!(if #test { #yes; })));
        };
        let no = self.expr(otherwise)?;
        let branch_lane = yes.lane.or(no.lane);
        if cond.lane.is_none() {
            let test = cond.code;
            let yes = yes.join(branch_lane, self.root, &i.then_branch)?;
            let no = no.join(branch_lane, self.root, otherwise)?;
            return Ok(Value {
                code: quote!(if #test { #yes } else { #no }),
                lane: branch_lane,
            });
        }
        let test = cond.operand(Lane::Bool, self.root, &i.cond)?;
        let result = branch_lane.unwrap_or(Lane::Int);
        let yes = yes.operand(result, self.root, &i.then_branch)?;
        let no = no.operand(result, self.root, otherwise)?;
        let ctx = self.ctx;
        // Evaluate nested expressions before borrowing the builder for select.
        Ok(Value::row(
            quote!({ let (mask, yes, no) = (#test, #yes, #no); #ctx.select(mask, yes, no) }),
            result,
        ))
    }

    fn unary(&self, u: &syn::ExprUnary) -> Result<Value> {
        let val = self.expr(&u.expr)?;
        let op = &u.op;
        let Some(lane) = val.lane else {
            let code = val.code;
            return Ok(Value::host(quote!(#op (#code))));
        };
        let root = self.root;
        let zero = quote!(#root::frontend::batch::Operand::Literal(0));
        let ones = quote!(#root::frontend::batch::Operand::Literal(-1));
        let code = match op {
            UnOp::Neg(_) if lane == Lane::Int => self.emit_binary("sub", zero, val.code, false),
            UnOp::Not(_) => self.emit_binary("xor", val.code, ones, false),
            _ => {
                return Err(Error::new_spanned(
                    u,
                    "unsupported unary operator for this runtime value",
                ));
            }
        };
        Ok(Value::row(code, lane))
    }

    fn binary(&self, b: &syn::ExprBinary) -> Result<Value> {
        let a = self.expr(&b.left)?;
        let c = self.expr(&b.right)?;
        let op = &b.op;
        let Some(inferred) = a.lane.or(c.lane) else {
            let (a, c) = (a.code, c.code);
            return Ok(Value::host(quote!((#a) #op (#c))));
        };
        use BinOp::*;
        let (method, input, output) = match op {
            Add(_) => ("add", Lane::Int, Lane::Int),
            Sub(_) => ("sub", Lane::Int, Lane::Int),
            Mul(_) => ("mul", Lane::Int, Lane::Int),
            BitAnd(_) => ("and", inferred, inferred),
            BitOr(_) => ("or", inferred, inferred),
            BitXor(_) => ("xor", inferred, inferred),
            And(_) => ("and", Lane::Bool, Lane::Bool),
            Or(_) => ("or", Lane::Bool, Lane::Bool),
            Eq(_) | Ne(_) => ("eq", inferred, Lane::Bool),
            Lt(_) | Gt(_) | Le(_) | Ge(_) => ("lt", Lane::Int, Lane::Bool),
            _ => {
                return Err(Error::new_spanned(
                    op,
                    "this operator has no runtime i32 HIR instruction",
                ));
            }
        };
        let lhs = a.operand(input, self.root, &b.left)?;
        let rhs = c.operand(input, self.root, &b.right)?;
        let mut code = self.emit_binary(method, lhs, rhs, matches!(op, Gt(_) | Le(_)));
        if matches!(op, Ne(_) | Le(_) | Ge(_)) {
            let root = self.root;
            code = self.emit_binary(
                "xor",
                code,
                quote!(#root::frontend::batch::Operand::Literal(-1)),
                false,
            );
        }
        Ok(Value::row(code, output))
    }

    fn emit_binary(
        &self,
        method: &str,
        lhs: TokenStream,
        rhs: TokenStream,
        swap: bool,
    ) -> TokenStream {
        let (ctx, root) = (self.ctx, self.root);
        let method = Ident::new(method, proc_macro2::Span::call_site());
        let args = if swap {
            quote!(right, left)
        } else {
            quote!(left, right)
        };
        quote!({ let (left, right) = (#lhs, #rhs); #ctx.binary(#root::hir::Builder::#method, #args) })
    }
}
