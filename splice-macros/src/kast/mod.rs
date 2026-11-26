//! The kernel body as one normalized model. A body is parsed once:
//! `comptime!` arguments become expressions, every expression, local and
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

/// How a parameter takes part in staging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Param {
    /// A `#[comptime]` value the generator knows.
    Known,
    /// A runtime scalar; `usize` when `index`.
    Scalar { index: bool },
    /// A runtime slice.
    Slice,
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
    origin: Origin,
}

/// The node addresses of one owned block. Kept private so that only this
/// module turns a node reference into an id.
#[derive(Default)]
struct Ids {
    exprs: HashMap<usize, NodeId>,
    locals: HashMap<usize, NodeId>,
    macros: HashMap<usize, NodeId>,
    uses: HashMap<usize, DeclId>,
    /// The declarations each `let` binds.
    let_decls: HashMap<NodeId, Vec<DeclId>>,
    /// The declarations bound by the pattern of a `match` arm, `for`,
    /// `if let`/`while let` or closure parameter, by the pattern's address.
    patterns: HashMap<usize, std::ops::Range<u32>>,
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
    /// Parsed `comptime!` arguments by the id of their macro. Boxed so that
    /// the ids of their nodes stay valid.
    comptime: HashMap<NodeId, Box<Expr>>,
    stages: stage::Stages,
    usize_decls: Vec<bool>,
}

impl<'a> Body<'a> {
    /// Normalize `block`, the body of a kernel with `params` in order.
    pub(crate) fn new(block: &'a Block, params: &[(Ident, Param)]) -> Result<Self> {
        let (ids, decls, comptime) = resolve::resolve(block, params)?;
        let mut body = Self {
            block,
            ids,
            decls,
            comptime,
            stages: stage::Stages::default(),
            usize_decls: Vec::new(),
        };
        body.stages = stage::analyze(&body);
        body.usize_decls = types::usize_decls(&body);
        Ok(body)
    }

    pub(crate) fn block(&self) -> &'a Block {
        self.block
    }

    fn lookup<T: Clone, N>(map: &HashMap<usize, T>, node: &N) -> T {
        map.get(&address(node))
            .expect("internal error: a node outside the normalized kernel body")
            .clone()
    }

    /// The id of an expression of the body.
    pub(crate) fn expr(&self, expr: &Expr) -> NodeId {
        Self::lookup(&self.ids.exprs, expr)
    }

    /// The parsed argument of a `comptime!` invocation in the body.
    pub(crate) fn comptime(&self, mac: &Macro) -> &Expr {
        &self.comptime[&Self::lookup(&self.ids.macros, mac)]
    }

    /// The declaration a single-identifier path names, if it names a
    /// parameter or a local rather than an item.
    fn decl_of(&self, path: &ExprPath) -> Option<DeclId> {
        self.ids.uses.get(&address(path)).copied()
    }

    /// Whether the generator must keep the local `let` as an IR register:
    /// its value or an assignment to it depends on a runtime input.
    pub(crate) fn runtime_local(&self, local: &Local) -> bool {
        self.stages
            .locals
            .contains(&Self::lookup(&self.ids.locals, local))
    }

    /// Whether the loop whose body is `body` must be lowered by the
    /// generator (as a runtime loop or unrolled) instead of running as Rust.
    pub(crate) fn staged_loop(&self, loop_expr: &Expr) -> bool {
        self.stages.loops.contains(&self.expr(loop_expr))
    }

    /// Whether Rust infers `usize` for the unannotated integer local `let`.
    pub(crate) fn usize_local(&self, local: &Local) -> bool {
        let id = Self::lookup(&self.ids.locals, local);
        self.ids
            .let_decls
            .get(&id)
            .is_some_and(|decls| decls.iter().any(|decl| self.usize_decls[decl.0 as usize]))
    }

    /// The declarations a binding pattern of the body introduces.
    fn pattern_decls(&self, pat: &syn::Pat) -> impl Iterator<Item = DeclId> {
        Self::lookup(&self.ids.patterns, pat).map(DeclId)
    }

    fn decl(&self, id: DeclId) -> &Decl {
        &self.decls[id.0 as usize]
    }
}

#[cfg(test)]
mod tests;
