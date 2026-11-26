//! Integer types Rust infers for unannotated literal locals. rustc has
//! already checked the kernel's Rust function, so a use beside a `usize`
//! operand (`.len()` of a slice argument, `as usize`, a `usize` argument or
//! local, a `usize` literal) or as a slice index makes the literal local a
//! `usize`: `let mut i = 0; while i < xs.len()`. Declarations are resolved,
//! so a shadowed name is a different local.
use super::{Body, DeclId, Origin, Param};
use syn::visit::Visit;
use syn::{BinOp, Expr, Lit, Pat};

pub(super) fn usize_decls(body: &Body<'_>) -> Vec<bool> {
    let mut collect = Collect {
        body,
        literal: vec![false; body.decls.len()],
        initialized: Vec::new(),
        uses: Vec::new(),
        indices: Vec::new(),
    };
    collect.visit_block(body.block());
    let mut found: Vec<bool> = body
        .decls
        .iter()
        .map(|decl| decl.origin == Origin::Param(Param::Scalar { index: true }))
        .collect();
    loop {
        let mut marks = Vec::new();
        let known = |e: &Expr| is_usize(body, &found, e);
        for (decl, init) in &collect.initialized {
            if !found[decl.0 as usize] && known(init) {
                marks.push(*decl);
            }
        }
        let mut operands = Vec::new();
        for (left, right) in &collect.uses {
            if known(right) {
                operands.push(*left);
            } else if known(left) {
                operands.push(*right);
            }
        }
        operands.extend(collect.indices.iter().copied());
        for operand in operands {
            mark(body, operand, &collect.literal, &mut marks);
        }
        marks.retain(|decl| !found[decl.0 as usize]);
        if marks.is_empty() {
            return found;
        }
        for decl in marks {
            found[decl.0 as usize] = true;
        }
    }
}

struct Collect<'b, 'a> {
    body: &'b Body<'a>,
    /// Declarations of `let [mut] x = <unsuffixed integer literal>`.
    literal: Vec<bool>,
    /// A single-name `let` and its initializer.
    initialized: Vec<(DeclId, &'b Expr)>,
    /// Operand pairs Rust types alike.
    uses: Vec<(&'b Expr, &'b Expr)>,
    /// Indices of slice arguments.
    indices: Vec<&'b Expr>,
}

impl<'b> Visit<'b> for Collect<'b, '_> {
    fn visit_local(&mut self, local: &'b syn::Local) {
        if let Pat::Ident(name) = &local.pat
            && name.subpat.is_none()
            && let Some(init) = &local.init
            && let Some(decls) = self
                .body
                .ids
                .let_decls
                .get(&Body::lookup(&self.body.ids.locals, local))
        {
            let decl = decls[0];
            match &*init.expr {
                Expr::Lit(literal) if matches!(&literal.lit, Lit::Int(int) if int.suffix().is_empty()) =>
                {
                    self.literal[decl.0 as usize] = true;
                }
                init => self.initialized.push((decl, init)),
            }
        }
        syn::visit::visit_local(self, local);
    }
    fn visit_expr_binary(&mut self, binary: &'b syn::ExprBinary) {
        if !matches!(
            binary.op,
            BinOp::Shl(_)
                | BinOp::Shr(_)
                | BinOp::ShlAssign(_)
                | BinOp::ShrAssign(_)
                | BinOp::And(_)
                | BinOp::Or(_)
        ) {
            self.uses.push((&binary.left, &binary.right));
        }
        syn::visit::visit_expr_binary(self, binary);
    }
    fn visit_expr_assign(&mut self, assign: &'b syn::ExprAssign) {
        self.uses.push((&assign.left, &assign.right));
        syn::visit::visit_expr_assign(self, assign);
    }
    fn visit_expr_index(&mut self, index: &'b syn::ExprIndex) {
        if slice(self.body, &index.expr) {
            self.indices.push(&index.index);
        }
        syn::visit::visit_expr_index(self, index);
    }
    fn visit_macro(&mut self, mac: &'b syn::Macro) {
        if mac.path.is_ident("comptime") {
            self.visit_expr(self.body.comptime(mac));
        }
    }
    fn visit_item(&mut self, _: &'b syn::Item) {}
}

fn param(body: &Body<'_>, e: &Expr) -> Option<Param> {
    let Expr::Path(path) = e else { return None };
    match body.decl(body.decl_of(path)?).origin {
        Origin::Param(param) => Some(param),
        _ => None,
    }
}

fn slice(body: &Body<'_>, e: &Expr) -> bool {
    param(body, e) == Some(Param::Slice)
}

fn arithmetic(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::Add(_)
            | BinOp::Sub(_)
            | BinOp::Mul(_)
            | BinOp::Div(_)
            | BinOp::Rem(_)
            | BinOp::BitAnd(_)
            | BinOp::BitOr(_)
            | BinOp::BitXor(_)
    )
}

fn is_usize(body: &Body<'_>, found: &[bool], e: &Expr) -> bool {
    match e {
        Expr::Paren(p) => is_usize(body, found, &p.expr),
        Expr::Group(g) => is_usize(body, found, &g.expr),
        Expr::Cast(c) => {
            matches!(&*c.ty, syn::Type::Path(t) if t.qself.is_none() && t.path.is_ident("usize"))
        }
        Expr::Lit(l) => matches!(&l.lit, Lit::Int(int) if int.suffix() == "usize"),
        Expr::MethodCall(m) => m.method == "len" && m.args.is_empty() && slice(body, &m.receiver),
        Expr::Binary(b) if arithmetic(&b.op) => {
            is_usize(body, found, &b.left) || is_usize(body, found, &b.right)
        }
        Expr::Path(path) => body.decl_of(path).is_some_and(|d| found[d.0 as usize]),
        _ => false,
    }
}

/// Mark the literal locals that `e`, a `usize` operand, consists of.
fn mark(body: &Body<'_>, e: &Expr, literal: &[bool], marks: &mut Vec<DeclId>) {
    match e {
        Expr::Paren(p) => mark(body, &p.expr, literal, marks),
        Expr::Group(g) => mark(body, &g.expr, literal, marks),
        Expr::Binary(b) if arithmetic(&b.op) => {
            mark(body, &b.left, literal, marks);
            mark(body, &b.right, literal, marks);
        }
        Expr::Path(path) => {
            if let Some(decl) = body.decl_of(path)
                && literal[decl.0 as usize]
            {
                marks.push(decl);
            }
        }
        _ => {}
    }
}
