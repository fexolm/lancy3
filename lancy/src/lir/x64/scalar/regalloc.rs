//! Conservative linear intervals with backward CFG liveness, including loop
//! backedges. Definitions and uses occupy distinct positions in an instruction.
//! Homes and spill slots can therefore be reused at a last operand use.
use super::frame::{Frame, Home, LINK_SLOTS, buffer_header_words};
use crate::hir::ssa::{Function, Value};
use crate::lir::ssa::{Exit, Op, SsaBody};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Default)]
struct Lifetime {
    def_block: usize,
    first: usize,
    last: usize,
    uses: Vec<usize>,
    used: bool,
    literal: Option<i32>,
}

pub(super) struct Allocation {
    pub homes: Vec<Home>,
    pub frame: Frame,
    pub call_masks: Vec<u8>,
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
    let mut call_points = vec![0; body.calls.len()];
    let mut position = 0;
    for (id, b) in body.blocks.iter().enumerate() {
        let entry = position;
        for &param in body.params[id] {
            let v = &mut lives[param.index()];
            v.def_block = id;
            v.first = entry;
        }
        position += 1;
        for &op in &body.ops[b.ops.clone()] {
            op.inputs(&body.calls, |v| {
                record_use(&mut lives[v.index()], id, position)
            });
            if let Op::Call(_, site) = op {
                call_points[site as usize] = position;
            }
            if let Some(dst) = op.dst() {
                let v = &mut lives[dst.index()];
                v.first = position + 1;
                v.def_block = id;
                if let Op::Imm(_, n) = op {
                    v.literal = Some(n);
                }
            }
            position += 2;
        }
        let mut used = |v: Value| record_use(&mut lives[v.index()], id, position);
        match b.exit {
            Exit::Return(v) => used(v),
            Exit::Branch(cond, ..) => {
                cond.map(&mut used);
            }
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
    for (id, v) in lives
        .iter_mut()
        .enumerate()
        .filter(|(_, v)| v.used && v.literal.is_none())
    {
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
        .filter(|&i| lives[i].used && lives[i].literal.is_none())
        .collect();
    order.sort_unstable_by_key(|&i| (lives[i].first, i));
    let mut homes: Vec<_> = lives
        .iter()
        .map(|v| v.literal.map_or(Home::Dead, Home::Immediate))
        .collect();
    let mut occupants: [Option<usize>; 4] = [None; 4];
    for &id in &order {
        let start = lives[id].first;
        let reg = occupants
            .iter()
            .position(|o| o.is_none_or(|v| lives[v].last < start))
            .or_else(|| {
                let (r, &old) = occupants
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, o)| lives[o.unwrap()].last)?;
                let old = old.unwrap();
                if lives[old].last <= lives[id].last {
                    return None;
                }
                homes[old] = Home::Dead; // Assign its spill home in the second scan.
                Some(r)
            });
        if let Some(r) = reg {
            occupants[r] = Some(id);
            homes[id] = Home::Register(8 + r as u8);
        }
    }
    let status_slot = if root { 0 } else { LINK_SLOTS + f.arity() };
    let mut slots = status_slot;
    if f.needs_status() {
        slots += buffer_header_words(f.buffer_kinds().len());
    }
    let mut spills: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    for &i in &order {
        if homes[i] != Home::Dead {
            continue;
        }
        let slot = if spills.peek().is_some_and(|s| s.0.0 < lives[i].first) {
            spills.pop().unwrap().0.1
        } else {
            let n = slots;
            slots += 1;
            n
        };
        // Emission checks the complete workspace against the signed x64 displacement limit.
        homes[i] = Home::Slot(slot as u32);
        spills.push(Reverse((lives[i].last, slot)));
    }
    // Calls are numbered in instruction order. Only a register whose value
    // survives this call needs a save. A dying argument can be overwritten.
    let mut call_masks = vec![0u8; call_points.len()];
    let mut saved = 0;
    for id in order {
        let Home::Register(reg) = homes[id] else {
            continue;
        };
        let bit = 1 << (reg - 8);
        let interval = &lives[id];
        let lo = call_points.partition_point(|&p| p < interval.first);
        let hi = call_points.partition_point(|&p| p < interval.last);
        for mask in &mut call_masks[lo..hi] {
            *mask |= bit;
            saved |= bit;
        }
    }
    let save_base = slots;
    if saved != 0 {
        slots += (u8::BITS - saved.leading_zeros()) as usize;
    }
    let args = body.params[0].iter().map(|v| homes[v.index()]).collect();
    Allocation {
        homes,
        frame: Frame {
            args,
            slots,
            status_slot,
            save_base,
            has_status: f.needs_status(),
        },
        call_masks,
    }
}
fn record_use(v: &mut Lifetime, b: usize, pos: usize) {
    v.used = true;
    v.last = v.last.max(pos);
    // Most temporaries never leave their defining block. Only cross-block
    // uses need a predecessor walk or storage for a list of blocks.
    if b != v.def_block && v.uses.last() != Some(&b) {
        v.uses.push(b);
    }
}
