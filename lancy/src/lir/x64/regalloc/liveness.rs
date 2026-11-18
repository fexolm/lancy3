use crate::hir::Value;
use crate::lir::{BlockId, block::Exit, ssa::SsaBody};
use crate::support::slotmap::DenseMap;

#[derive(Default)]
pub(super) struct Lifetime {
    pub(super) def_block: Option<BlockId>,
    pub(super) first: usize,
    pub(super) last: usize,
    uses: Vec<BlockId>,
}

pub(super) fn intervals(body: &SsaBody) -> DenseMap<Value, Lifetime> {
    let mut lives = DenseMap::from_fn(body.values, |_| Lifetime::default());
    let mut predecessors = DenseMap::from_fn(body.blocks.len(), |_| Vec::new());
    let mut bounds = DenseMap::filled(body.blocks.len(), (0, 0));
    let mut position = 0;
    for (id, b) in body.blocks.iter() {
        let entry = position;
        for &param in body.params[id] {
            let v = &mut lives[param];
            v.def_block = Some(id);
            v.first = entry;
            v.last = v.last.max(entry);
        }
        position += 1;
        for &op in &body.ops[b.ops.clone()] {
            op.inputs(&body.calls, |v| record_use(&mut lives[v], id, position));
            if let Some(dst) = op.dst() {
                let v = &mut lives[dst];
                v.first = position + 1;
                v.def_block = Some(id);
                v.last = v.last.max(v.first);
            }
            position += 2;
        }
        let mut used = |v: Value| record_use(&mut lives[v], id, position);
        match b.exit {
            Exit::Return(v) => used(v),
            Exit::Branch(cond, ..) => used(cond),
            Exit::Jump(_) => {}
        }
        for e in b.exit.successors() {
            let edge = &body.edges[e];
            predecessors[edge.target].push(id);
            edge.args.iter().copied().for_each(&mut used);
        }
        bounds[id] = (entry, position);
        position += 1;
    }
    let mut seen = DenseMap::filled(body.blocks.len(), Value::INVALID);
    for (id, v) in lives.iter_mut() {
        // Cross-block uses are live-in up to the block defining this value.
        // The per-value stamp avoids clearing a CFG-sized bitmap for each use.
        while let Some(b) = v.uses.pop() {
            if Some(b) == v.def_block || seen[b] == id {
                continue;
            }
            seen[b] = id;
            v.first = v.first.min(bounds[b].0);
            for &p in &predecessors[b] {
                v.last = v.last.max(bounds[p].1);
                v.uses.push(p);
            }
        }
    }
    lives
}

fn record_use(v: &mut Lifetime, b: BlockId, pos: usize) {
    v.last = v.last.max(pos);
    // Most temporaries never leave their defining block. Only cross-block
    // uses need a predecessor walk or storage for a list of blocks.
    if Some(b) != v.def_block && v.uses.last() != Some(&b) {
        v.uses.push(b);
    }
}
