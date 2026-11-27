//! The optimization pipeline: an ordered list of passes over one function
//! and the analyses they share. A pass declares what it kept valid, and an
//! analysis it invalidated is recomputed when the next pass asks for it.
use super::bounds::{self, Bounds};
use super::facts::Facts;
use super::{checkpoint, dead, loops, memory, simplify};
use crate::Trap;
use crate::hir::transforms::{ControlFlowGraph, DomTree};
use crate::hir::{Block, Function, IrError, Op, Value};
use splice_target::Arch;
use std::collections::HashMap;

/// What a pass left valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Preserved {
    /// Every analysis, including the ones the pass computed or updated.
    All,
    /// Blocks and edges, but not the facts about values.
    Cfg,
    Nothing,
}

/// An analysis given by the caller or computed by the pipeline.
pub(super) enum Held<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<T> std::ops::Deref for Held<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => value,
        }
    }
}

/// Analyses shared by the passes of one pipeline run, computed on first use.
pub(super) struct Analyses<'a> {
    cfg: Option<Held<'a, ControlFlowGraph>>,
    dom: Option<Held<'a, DomTree>>,
    facts: Option<Facts>,
    /// Bounds built by a pass with extra knowledge (the loop bounds of
    /// `loops`). Without them, a pass builds the plain bounds itself.
    bounds: Option<Bounds>,
    /// The architecture the function is compiled for; register budgets
    /// depend on it.
    pub(super) arch: Arch,
}

impl<'a> Analyses<'a> {
    /// Start from a CFG and dominator tree that describe `f`, such as the
    /// verifier's.
    pub(super) fn new(
        cfg: Held<'a, ControlFlowGraph>,
        dom: Option<&'a DomTree>,
        arch: Arch,
    ) -> Self {
        Self {
            cfg: Some(cfg),
            dom: dom.map(Held::Borrowed),
            facts: None,
            bounds: None,
            arch,
        }
    }

    pub(super) fn invalidate(&mut self, preserved: Preserved) {
        if preserved != Preserved::All {
            self.facts = None;
            self.bounds = None;
        }
        if preserved == Preserved::Nothing {
            self.cfg = None;
            self.dom = None;
        }
    }

    pub(super) fn cfg(&mut self, f: &Function) -> Result<&ControlFlowGraph, IrError> {
        if self.cfg.is_none() {
            self.cfg = Some(Held::Owned(ControlFlowGraph::compute(f)?));
        }
        Ok(self.cfg.as_deref().unwrap())
    }

    pub(super) fn graph(&mut self, f: &Function) -> Result<(&ControlFlowGraph, &DomTree), IrError> {
        self.cfg(f)?;
        let cfg = self.cfg.as_deref().unwrap();
        let dom = self
            .dom
            .get_or_insert_with(|| Held::Owned(DomTree::compute(cfg)));
        Ok((cfg, dom))
    }

    /// The CFG, dominators, value facts and any bounds a pass left behind.
    pub(super) fn all(&mut self, f: &Function) -> Result<Shared<'_>, IrError> {
        self.graph(f)?;
        let cfg = self.cfg.as_deref().unwrap();
        let dom = self.dom.as_deref().unwrap();
        let facts = self.facts.get_or_insert_with(|| Facts::new(f, cfg, dom));
        Ok(Shared {
            cfg,
            dom,
            facts,
            bounds: &mut self.bounds,
        })
    }

    /// Keep the facts and bounds a pass computed for the current function.
    pub(super) fn keep(&mut self, facts: Facts, bounds: Bounds) {
        self.facts = Some(facts);
        self.bounds = Some(bounds);
    }
}

pub(super) struct Shared<'s> {
    pub(super) cfg: &'s ControlFlowGraph,
    pub(super) dom: &'s DomTree,
    pub(super) facts: &'s mut Facts,
    pub(super) bounds: &'s mut Option<Bounds>,
}

struct Pass {
    name: &'static str,
    run: fn(&mut Function, &mut Analyses<'_>) -> Result<Preserved, IrError>,
}

/// Division, memory operations and checks remain ordered on every
/// executable path. The passes after `loops` only remove checks, loads,
/// stores and dead values, or forward values that the facts already
/// equate, so the facts stay valid for the passes after them.
const PIPELINE: &[Pass] = &[
    Pass {
        name: "sccp and local_cfg",
        run: simplify::propagate,
    },
    Pass {
        name: "loops",
        run: loops::run,
    },
    Pass {
        name: "acyclic parameter forwarding",
        run: |f, a| {
            let shared = a.all(f)?;
            // Physical forwarding across loop edges can lengthen hot live ranges.
            if !shared.cfg.has_cycle() {
                shared.facts.forward_acyclic_params(f, shared.cfg);
            }
            Ok(Preserved::All)
        },
    },
    Pass {
        name: "bounds",
        run: |f, a| {
            let s = a.all(f)?;
            bounds::remove_full_packet_checks(f, s.cfg, s.dom, s.facts, s.bounds);
            Ok(Preserved::All)
        },
    },
    Pass {
        name: "redundant checks",
        run: |f, a| {
            let s = a.all(f)?;
            remove_redundant_checks(f, s.dom, s.facts);
            Ok(Preserved::All)
        },
    },
    Pass {
        name: "memory",
        run: |f, a| {
            let s = a.all(f)?;
            memory::optimize(f, s.cfg, s.dom, s.facts, s.bounds.as_ref());
            Ok(Preserved::All)
        },
    },
    Pass {
        name: "dead",
        run: |f, a| {
            let s = a.all(f)?;
            dead::eliminate(f, s.cfg, s.facts);
            Ok(Preserved::All)
        },
    },
];

/// Run every pass in order. In test and debug builds each pass's output is
/// verified, so a pass that breaks HIR is named where it ran.
pub(super) fn optimize(f: &mut Function, mut analyses: Analyses<'_>) -> Result<(), IrError> {
    for pass in PIPELINE {
        let preserved = (pass.run)(f, &mut analyses)?;
        analyses.invalidate(preserved);
        checkpoint(f, pass.name);
    }
    // Structural passes retire edges, calls and packets; drop them before
    // the function is verified and lowered again.
    f.compact();
    Ok(())
}

/// Remove a check that a dominating check of the same values already made:
/// a bounds check, or a conditional trap of the same kind. Values compare by
/// their facts root, so checks through forwarded parameters match (HIR-18).
fn remove_redundant_checks(f: &mut Function, dom: &DomTree, facts: &Facts) {
    #[derive(PartialEq, Eq, Hash)]
    enum Check {
        Bounds(Value, Value),
        Trap(Value, Trap),
    }
    let mut checks = HashMap::new();
    for &b in dom.preorder() {
        f.rewrite_instructions(b, |op, _, _| {
            let check = match op {
                Op::BoundsCheck(i, n) => Check::Bounds(facts.root(i), facts.root(n)),
                Op::TrapIf(c, trap) => Check::Trap(facts.root(c), trap),
                _ => return Some(op),
            };
            let previous = checks.entry(check).or_insert(Block::INVALID);
            if dom.dominates(*previous, b) {
                return None;
            }
            *previous = b;
            Some(op)
        });
    }
}
