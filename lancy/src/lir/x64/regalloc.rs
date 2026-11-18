//! Conservative linear intervals with backward CFG liveness, including loop
//! backedges. Definitions and uses occupy distinct positions in an instruction.
//! Four volatile registers are assigned by linear scan; remaining values spill.
use super::frame::{Frame, Home, LINK_SLOTS, buffer_header_words};
use crate::hir::{Function, Value};
use crate::lir::ssa::{Exit, SsaBody};

#[derive(Default)]
struct Lifetime {
    def_block: usize,
    first: usize,
    last: usize,
    uses: Vec<usize>,
}

pub(super) struct Allocation {
    pub homes: Vec<Home>,
    pub frame: Frame,
}

pub(super) fn allocate(f: &Function, body: &SsaBody, root: bool) -> Allocation {
    let mut lives: Vec<Lifetime> = (0..body.values)
        .map(|_| Lifetime {
            def_block: usize::MAX,
            ..Lifetime::default()
        })
        .collect();
    let mut predecessors = vec![Vec::new(); body.blocks.len()];
    let mut bounds = Vec::with_capacity(body.blocks.len());
    let mut position = 0;
    for (id, b) in body.blocks.iter().enumerate() {
        let entry = position;
        for &param in body.params[id] {
            let v = &mut lives[param.index()];
            v.def_block = id;
            v.first = entry;
            v.last = v.last.max(entry);
        }
        position += 1;
        for &op in &body.ops[b.ops.clone()] {
            op.inputs(&body.calls, |v| {
                record_use(&mut lives[v.index()], id, position)
            });
            if let Some(dst) = op.dst() {
                let v = &mut lives[dst.index()];
                v.first = position + 1;
                v.def_block = id;
                v.last = v.last.max(v.first);
            }
            position += 2;
        }
        let mut used = |v: Value| record_use(&mut lives[v.index()], id, position);
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
        bounds.push((entry, position));
        position += 1;
    }
    let mut seen = vec![usize::MAX; body.blocks.len()];
    for (id, v) in lives.iter_mut().enumerate() {
        // Cross-block uses are live-in up to the block defining this value.
        // The per-value stamp avoids clearing a CFG-sized bitmap for each use.
        while let Some(b) = v.uses.pop() {
            if b == v.def_block || seen[b] == id {
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
    let mut order: Vec<_> = (0..lives.len())
        .filter(|&i| lives[i].def_block != usize::MAX)
        .collect();
    order.sort_unstable_by_key(|&id| (lives[id].first, id));
    let status_slot = if root { 0 } else { LINK_SLOTS + f.arity() };
    let mut slots = status_slot + buffer_header_words(f.buffer_kinds().len());
    let mut occupied_until = [None; 4];
    let mut homes = vec![Home::Slot(0); lives.len()];
    for id in order {
        let live = &lives[id];
        homes[id] = if let Some(r) = occupied_until
            .iter()
            .position(|end| end.is_none_or(|p| p < live.first))
        {
            occupied_until[r] = Some(live.last);
            Home::Register(8 + r as u8)
        } else {
            let slot = slots;
            slots += 1;
            Home::Slot(slot as u32)
        };
    }
    let save_base = slots;
    if !body.calls.is_empty() {
        slots += occupied_until.len();
    }
    let args = body.params[0].iter().map(|v| homes[v.index()]).collect();
    Allocation {
        homes,
        frame: Frame {
            args,
            slots,
            status_slot,
            save_base,
        },
    }
}
fn record_use(v: &mut Lifetime, b: usize, pos: usize) {
    v.last = v.last.max(pos);
    // Most temporaries never leave their defining block. Only cross-block
    // uses need a predecessor walk or storage for a list of blocks.
    if b != v.def_block && v.uses.last() != Some(&b) {
        v.uses.push(b);
    }
}
