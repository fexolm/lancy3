//! The kernel body as one normalized model. A body is parsed once:
//! `comptime!` arguments and `vec!` elements become expressions, every expression, local and
//! loop gets a [`NodeId`], and every use of a local or parameter is linked to
//! its declaration through a scope tree ([`resolve`]). The staging analysis
//! ([`stage`]) and the integer type inference ([`types`]) work on these ids,
//! and the lowering asks the body for facts about the nodes it lowers.
//!
//! The ids belong to the nodes of the one block the body was built from. A
//! query for a node outside it (a clone, a reparsed fragment) is an internal
//! error, not a silent default.
mod resolve;
mod stage;
pub(crate) mod types;

use proc_macro2::Ident;
use std::collections::HashMap;
use syn::{Block, Expr, ExprPath, Local, Macro, Result};

/// The position of a node in the pre-order walk of one body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct NodeId(u32);

/// A declaration: a parameter or a name bound by a pattern in the body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DeclId(u32);

impl PartialOrd for DeclId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DeclId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

/// How a parameter takes part in staging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Param {
    /// A `#[comptime]` value the generator knows.
    Known,
    /// A runtime scalar of a primitive type, if it has one.
    Scalar(Option<types::Primitive>),
    /// A runtime slice, and the primitive type of its elements if it has one.
    Slice(Option<types::Primitive>),
    /// Another runtime object (context, column table).
    Object,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Origin {
    Param(Param),
    /// A name bound by the pattern of `let`, with that `let`'s id.
    Let(NodeId),
    /// A name bound by a `match` arm, `for`, `if let`, `while let` or closure.
    Pattern,
}

#[derive(Debug)]
struct Decl {
    name: Ident,
    origin: Origin,
    mutable: bool,
}

/// The node addresses of one owned block. Kept private so that only this
/// module turns a node reference into an id.
#[derive(Default)]
struct Ids {
    exprs: HashMap<usize, NodeId>,
    locals: HashMap<usize, NodeId>,
    macros: HashMap<usize, NodeId>,
    /// Every path expression of the body: the declaration it names, or
    /// `None` for an item.
    uses: HashMap<usize, Option<DeclId>>,
    /// The declaration each binding identifier of a pattern introduces.
    bindings: HashMap<usize, DeclId>,
    /// The declarations each `let` binds.
    let_decls: HashMap<NodeId, Vec<DeclId>>,
    /// The declarations bound by the pattern of a `match` arm, `for`,
    /// `if let`/`while let` or closure parameter, by the pattern's address.
    patterns: HashMap<usize, std::ops::Range<u32>>,
    /// The declarations whose fields the body reads (`v.0`, `v.name`).
    projected: std::collections::HashSet<DeclId>,
    next: u32,
}

fn address<T>(node: &T) -> usize {
    std::ptr::from_ref(node) as usize
}

impl Ids {
    fn fresh(&mut self) -> NodeId {
        let id = NodeId(self.next);
        self.next += 1;
        id
    }
}

pub(crate) struct Body<'a> {
    block: &'a Block,
    ids: Ids,
    decls: Vec<Decl>,
    /// Parsed `comptime!` arguments and `vec!` elements (an array or repeat
    /// expression) by the id of their macro. Boxed so that the ids of their
    /// nodes stay valid.
    macros: HashMap<NodeId, Box<Expr>>,
    stages: stage::Stages,
    local_types: Vec<Option<types::Primitive>>,
}

impl<'a> Body<'a> {
    /// Normalize `block`, the body of a kernel with `params` in order.
    pub(crate) fn new(block: &'a Block, params: &[(Ident, Param)]) -> Result<Self> {
        let (ids, decls, macros) = resolve::resolve(block, params)?;
        let mut body = Self {
            block,
            ids,
            decls,
            macros,
            stages: stage::Stages::default(),
            local_types: Vec::new(),
        };
        body.stages = stage::analyze(&body);
        body.local_types = types::local_types(&body);
        Ok(body)
    }

    pub(crate) fn block(&self) -> &'a Block {
        self.block
    }

    fn lookup<T: Clone, N: quote::ToTokens>(map: &HashMap<usize, T>, node: &N) -> T {
        map.get(&address(node))
            .unwrap_or_else(|| {
                panic!(
                    "internal error: a node outside the normalized kernel body: {}",
                    node.to_token_stream()
                )
            })
            .clone()
    }

    /// The id of an expression of the body.
    pub(crate) fn expr(&self, expr: &Expr) -> NodeId {
        Self::lookup(&self.ids.exprs, expr)
    }

    /// The parsed argument of a `comptime!` invocation in the body.
    pub(crate) fn comptime(&self, mac: &Macro) -> &Expr {
        &self.macros[&Self::lookup(&self.ids.macros, mac)]
    }

    /// The elements of a `vec!` invocation in the body: an array or a
    /// repeat expression.
    pub(crate) fn vec_elements(&self, mac: &Macro) -> &Expr {
        &self.macros[&Self::lookup(&self.ids.macros, mac)]
    }

    /// The declaration a path of the body names, if it names a parameter
    /// or a local rather than an item.
    pub(crate) fn decl_of(&self, path: &ExprPath) -> Option<DeclId> {
        Self::lookup(&self.ids.uses, path)
    }

    /// The declaration a binding identifier of a pattern of the body introduces.
    pub(crate) fn binding(&self, pat: &syn::PatIdent) -> DeclId {
        Self::lookup(&self.ids.bindings, pat)
    }

    /// The declaration of parameter `index` of the kernel.
    pub(crate) fn param(index: usize) -> DeclId {
        DeclId(index as u32)
    }

    /// The name a declaration binds.
    pub(crate) fn name(&self, decl: DeclId) -> &Ident {
        &self.decl(decl).name
    }

    /// Whether the kernel has parameters.
    pub(crate) fn has_params(&self) -> bool {
        self.decls
            .first()
            .is_some_and(|decl| matches!(decl.origin, Origin::Param(_)))
    }

    /// Whether the declaration is a parameter of the kernel.
    pub(crate) fn is_param(&self, decl: DeclId) -> bool {
        matches!(self.decl(decl).origin, Origin::Param(_))
    }

    /// The parameter a path names, unless a local of the body shadows it.
    pub(crate) fn parameter(&self, path: &ExprPath) -> Option<&Ident> {
        let decl = self.decl(self.decl_of(path)?);
        matches!(decl.origin, Origin::Param(_)).then_some(&decl.name)
    }

    /// Whether the generator must keep the local `let` as an IR register:
    /// its value or an assignment to it depends on a runtime input.
    pub(crate) fn runtime_local(&self, local: &Local) -> bool {
        self.stages
            .locals
            .contains(&Self::lookup(&self.ids.locals, local))
    }

    /// Whether the generator knows the value of `expr` while it expands the
    /// kernel ([`stage`]): it runs the expression as Rust.
    pub(crate) fn known(&self, expr: &Expr) -> bool {
        self.stages.exprs[self.expr(expr).0 as usize] == stage::Level::Known
    }

    /// Whether the loop whose body is `body` must be lowered by the
    /// generator (as a runtime loop or unrolled) instead of running as Rust.
    pub(crate) fn staged_loop(&self, loop_expr: &Expr) -> bool {
        self.stages.loops.contains(&self.expr(loop_expr))
    }

    /// The type Rust infers from its uses for the unannotated local `let`.
    pub(crate) fn literal_type(&self, local: &Local) -> Option<types::Primitive> {
        let id = Self::lookup(&self.ids.locals, local);
        self.ids
            .let_decls
            .get(&id)?
            .iter()
            .find_map(|decl| self.local_types[decl.0 as usize])
    }

    /// Whether the body reads a field of a name the `let` binds.
    pub(crate) fn projected_local(&self, local: &Local) -> bool {
        let id = Self::lookup(&self.ids.locals, local);
        self.ids
            .let_decls
            .get(&id)
            .is_some_and(|decls| decls.iter().any(|decl| self.ids.projected.contains(decl)))
    }

    /// The declarations a binding pattern of the body introduces.
    fn pattern_decls(&self, pat: &syn::Pat) -> impl Iterator<Item = DeclId> {
        Self::lookup(&self.ids.patterns, pat).map(DeclId)
    }

    /// Whether the declaration is a `mut` binding.
    fn mutable(&self, decl: DeclId) -> bool {
        self.decl(decl).mutable
    }

    fn decl(&self, id: DeclId) -> &Decl {
        &self.decls[id.0 as usize]
    }
}

/// Whether `path` names the standard `vec!` macro.
pub(crate) fn vec_macro(path: &syn::Path) -> bool {
    path.is_ident("vec")
        || (path.segments.len() == 2
            && (path.segments[0].ident == "std" || path.segments[0].ident == "alloc")
            && path.segments[1].ident == "vec")
}

/// The elements of `vec![a, b]` or `vec![v; n]`, as an array or a repeat
/// expression.
pub(crate) fn vec_elements(mac: &Macro) -> Result<Expr> {
    let tokens = &mac.tokens;
    syn::parse2(quote::quote!([#tokens]))
}

#[cfg(test)]
mod tests;
