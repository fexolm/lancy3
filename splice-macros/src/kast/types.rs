//! Number types Rust infers for unannotated literal locals. rustc has
//! already checked the kernel's Rust function, so a literal local used
//! beside an operand of a known type (a typed parameter or local, `.len()` of
//! a slice argument, a cast, a suffixed literal) in a comparison, arithmetic
//! or an assignment has that type, and one that indexes a slice argument is a
//! `usize`: `let mut i = 0; while i < xs.len()`. Declarations are resolved,
//! so a shadowed name is a different local.
use super::{Body, DeclId, Origin, Param};
use syn::visit::Visit;
use syn::{BinOp, Expr, Lit, Pat, Type};

/// A primitive Rust type, the one table every classification of the macro
/// reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Primitive {
    Bool,
    Char,
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    F32,
    F64,
}

impl Primitive {
    const ALL: [(Self, &'static str); 16] = [
        (Self::Bool, "bool"),
        (Self::Char, "char"),
        (Self::I8, "i8"),
        (Self::I16, "i16"),
        (Self::I32, "i32"),
        (Self::I64, "i64"),
        (Self::I128, "i128"),
        (Self::Isize, "isize"),
        (Self::U8, "u8"),
        (Self::U16, "u16"),
        (Self::U32, "u32"),
        (Self::U64, "u64"),
        (Self::U128, "u128"),
        (Self::Usize, "usize"),
        (Self::F32, "f32"),
        (Self::F64, "f64"),
    ];

    /// The primitive `ty` names: `i32`, or the full path
    /// `[::]core::primitive::i32` (also through `std`).
    pub(crate) fn of(ty: &Type) -> Option<Self> {
        let Type::Path(path) = ty else { return None };
        if path.qself.is_some() {
            return None;
        }
        let path = &path.path;
        let name = match path.segments.len() {
            1 if path.leading_colon.is_none() => &path.segments[0],
            3 if matches!(path.segments[0].ident.to_string().as_str(), "core" | "std")
                && path.segments[1].ident == "primitive" =>
            {
                &path.segments[2]
            }
            _ => return None,
        };
        if !name.arguments.is_none() {
            return None;
        }
        Self::named(&name.ident.to_string())
    }

    pub(crate) fn named(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .find_map(|(primitive, spelling)| (*spelling == name).then_some(*primitive))
    }
}

/// The type of each declaration that has one Rust infers from its uses.
pub(super) fn local_types(body: &Body<'_>) -> Vec<Option<Primitive>> {
    let mut collect = Collect {
        body,
        literal: vec![None; body.decls.len()],
        initialized: Vec::new(),
        uses: Vec::new(),
        indices: Vec::new(),
        elements: Vec::new(),
    };
    collect.visit_block(body.block());
    let mut found: Vec<Option<Primitive>> = body
        .decls
        .iter()
        .map(|decl| match decl.origin {
            Origin::Param(Param::Scalar(ty)) => ty,
            _ => None,
        })
        .collect();
    for (decl, element) in &collect.elements {
        found[decl.0 as usize] = Some(*element);
    }
    loop {
        let mut marks = Vec::new();
        let known = |e: &Expr| type_of(body, &found, e);
        for (decl, init) in &collect.initialized {
            if found[decl.0 as usize].is_none()
                && let Some(ty) = known(init)
            {
                marks.push((*decl, ty));
            }
        }
        let mut operands = Vec::new();
        for (left, right) in &collect.uses {
            if let Some(ty) = known(right) {
                operands.push((*left, ty));
            } else if let Some(ty) = known(left) {
                operands.push((*right, ty));
            }
        }
        operands.extend(
            collect
                .indices
                .iter()
                .map(|index| (*index, Primitive::Usize)),
        );
        for (operand, ty) in operands {
            mark(body, operand, ty, &collect.literal, &mut marks);
        }
        marks.retain(|(decl, _)| found[decl.0 as usize].is_none());
        if marks.is_empty() {
            return found;
        }
        for (decl, ty) in marks {
            found[decl.0 as usize] = Some(ty);
        }
    }
}

struct Collect<'b, 'a> {
    body: &'b Body<'a>,
    /// Declarations of `let [mut] x = <unsuffixed number literal>`: whether
    /// the literal is a float.
    literal: Vec<Option<bool>>,
    /// A single-name `let` and its initializer.
    initialized: Vec<(DeclId, &'b Expr)>,
    /// Operand pairs Rust types alike.
    uses: Vec<(&'b Expr, &'b Expr)>,
    /// Indices of slice arguments.
    indices: Vec<&'b Expr>,
    /// Bindings of slice elements a `for` loop reads.
    elements: Vec<(DeclId, Primitive)>,
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
                    self.literal[decl.0 as usize] = Some(false);
                }
                Expr::Lit(literal) if matches!(&literal.lit, Lit::Float(float) if float.suffix().is_empty()) =>
                {
                    self.literal[decl.0 as usize] = Some(true);
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
    fn visit_expr_for_loop(&mut self, f: &'b syn::ExprForLoop) {
        match (&*f.pat, &*f.expr) {
            (Pat::Ident(name), Expr::Range(range)) if name.subpat.is_none() => {
                if let Some(end) = &range.end {
                    self.initialized.push((self.body.binding(name), end));
                }
            }
            (Pat::Reference(reference), iterable) => {
                if let Pat::Ident(name) = &*reference.pat
                    && let Some(element) = iterated_element(self.body, iterable)
                {
                    self.elements.push((self.body.binding(name), element));
                }
            }
            // `for x in xs`: `x` is a reference whose `*x` has the element type.
            (Pat::Ident(name), iterable) if name.subpat.is_none() => {
                if let Some(element) = iterated_element(self.body, iterable) {
                    self.elements.push((self.body.binding(name), element));
                }
            }
            _ => {}
        }
        syn::visit::visit_expr_for_loop(self, f);
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
    matches!(param(body, e), Some(Param::Slice(_)))
}

/// The element type of the slice argument a `for` loop reads, as in
/// `for &x in xs` or `for &x in xs.iter()`.
fn iterated_element(body: &Body<'_>, e: &Expr) -> Option<Primitive> {
    let e = match e {
        Expr::MethodCall(call) if call.method == "iter" && call.args.is_empty() => &*call.receiver,
        e => e,
    };
    match param(body, e)? {
        Param::Slice(element) => element,
        _ => None,
    }
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

/// The type rustc gives `e`, when its operands name it.
fn type_of(body: &Body<'_>, found: &[Option<Primitive>], e: &Expr) -> Option<Primitive> {
    match e {
        Expr::Paren(p) => type_of(body, found, &p.expr),
        Expr::Group(g) => type_of(body, found, &g.expr),
        Expr::Unary(u)
            if matches!(
                u.op,
                syn::UnOp::Neg(_) | syn::UnOp::Not(_) | syn::UnOp::Deref(_)
            ) =>
        {
            type_of(body, found, &u.expr)
        }
        Expr::Cast(c) => Primitive::of(&c.ty),
        Expr::Lit(l) => match &l.lit {
            Lit::Int(int) => Primitive::named(int.suffix()),
            Lit::Float(float) => Primitive::named(float.suffix()),
            _ => None,
        },
        Expr::MethodCall(m)
            if m.method == "len" && m.args.is_empty() && slice(body, &m.receiver) =>
        {
            Some(Primitive::Usize)
        }
        Expr::Binary(b) if arithmetic(&b.op) => {
            type_of(body, found, &b.left).or_else(|| type_of(body, found, &b.right))
        }
        Expr::Path(path) => body.decl_of(path).and_then(|d| found[d.0 as usize]),
        Expr::Index(index) => match param(body, &index.expr)? {
            Param::Slice(element) => element,
            _ => None,
        },
        _ => None,
    }
}

impl Primitive {
    pub(crate) fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
    pub(crate) fn is_number(self) -> bool {
        !matches!(self, Self::Bool | Self::Char)
    }
    /// The Rust name of the type.
    pub(crate) fn name(self) -> &'static str {
        Self::ALL
            .iter()
            .find_map(|(primitive, spelling)| (*primitive == self).then_some(*spelling))
            .expect("every primitive has a name")
    }
    /// Whether a kernel register can hold the type: what
    /// `splice::KernelScalar` is implemented for. 128-bit integers and
    /// `isize` are not.
    pub(crate) fn is_kernel_scalar(self) -> bool {
        !matches!(self, Self::I128 | Self::U128 | Self::Isize)
    }
    /// The kernel scalar types, for diagnostics: `bool, char, i8, …`.
    pub(crate) fn kernel_scalars() -> String {
        Self::ALL
            .iter()
            .filter(|(primitive, _)| primitive.is_kernel_scalar())
            .map(|(_, name)| *name)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Mark the literal locals that `e`, an operand of type `ty`, consists of.
fn mark(
    body: &Body<'_>,
    e: &Expr,
    ty: Primitive,
    literal: &[Option<bool>],
    marks: &mut Vec<(DeclId, Primitive)>,
) {
    match e {
        Expr::Paren(p) => mark(body, &p.expr, ty, literal, marks),
        Expr::Group(g) => mark(body, &g.expr, ty, literal, marks),
        Expr::Unary(u)
            if matches!(
                u.op,
                syn::UnOp::Neg(_) | syn::UnOp::Not(_) | syn::UnOp::Deref(_)
            ) =>
        {
            mark(body, &u.expr, ty, literal, marks);
        }
        Expr::Binary(b) if arithmetic(&b.op) => {
            mark(body, &b.left, ty, literal, marks);
            mark(body, &b.right, ty, literal, marks);
        }
        Expr::Path(path) => {
            if let Some(decl) = body.decl_of(path)
                && ty.is_number()
                && literal[decl.0 as usize] == Some(ty.is_float())
            {
                marks.push((decl, ty));
            }
        }
        _ => {}
    }
}
