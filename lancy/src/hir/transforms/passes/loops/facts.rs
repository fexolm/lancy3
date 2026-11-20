use super::{Definition, definitions};
use crate::hir::transforms::{ControlFlowGraph, DomTree};
use crate::hir::{Block, Function, Inst, Op, Value};
use crate::support::slotmap::DenseMap;

pub(super) struct Facts {
    pub(super) defs: DenseMap<Value, Definition>,
    roots: DenseMap<Value, Value>,
}

impl Facts {
    pub(super) fn new(f: &Function, cfg: &ControlFlowGraph, dom: &DomTree) -> Self {
        let defs = definitions(f);
        let mut roots = DenseMap::from_fn(f.value_count(), |v| v);
        // First collapse straight forwarding chains. A second sweep then
        // sees unchanged backedge arguments as the header parameter itself.
        // Only link to a definition strictly dominating the destination.
        let mut incoming = DenseMap::filled(f.value_count(), None);
        for _ in 0..2 {
            for candidate in incoming.values_mut() {
                *candidate = None;
            }
            for &block in cfg.reverse_postorder() {
                for edge in f.terminator(block).edges() {
                    let (target, args) = f.edge(edge);
                    for (&param, &arg) in f.params(target).iter().zip(args) {
                        let arg = roots[arg];
                        if roots[param] == param && arg != param {
                            let prior = incoming[param].get_or_insert(arg);
                            if *prior != arg {
                                *prior = Value::INVALID;
                            }
                        }
                    }
                }
            }
            for &block in cfg.reverse_postorder() {
                for &p in f.params(block) {
                    if let Some(v) = incoming[p]
                        && v != Value::INVALID
                        && defs[v].block != block
                        && dom.dominates(defs[v].block, block)
                    {
                        roots[p] = v;
                    }
                }
            }
            for v in roots.keys() {
                let mut root = v;
                while roots[root] != root {
                    let parent = roots[root];
                    roots[root] = roots[parent];
                    root = roots[root];
                }
                roots[v] = root;
            }
        }
        Self { defs, roots }
    }

    pub(super) fn root(&self, v: Value) -> Value {
        self.roots[v]
    }

    pub(super) fn available(&self, v: Value, at: Block, dom: &DomTree) -> Option<Value> {
        let v = self.root(v);
        dom.dominates(self.defs[v].block, at).then_some(v)
    }

    pub(super) fn op(&self, f: &Function, v: Value) -> Option<Op> {
        let inst = self.defs[self.root(v)].inst;
        (inst != Inst::INVALID).then(|| *f.op(inst))
    }

    pub(super) fn constant(&self, f: &Function, v: Value) -> Option<i32> {
        match self.op(f, v) {
            Some(Op::Const(n)) => Some(n),
            _ => None,
        }
    }

    pub(super) fn append(&mut self, f: &mut Function, block: Block, op: Op) -> Value {
        let inst = f.append_inst(block, op);
        let value = f.result(inst).expect("value-producing operation");
        debug_assert_eq!(self.defs.len(), value.index());
        self.defs.insert(Definition { block, inst });
        self.roots.insert(value);
        value
    }
}
