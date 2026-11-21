//! Place fallthrough chains late, leaving the allocator's block numbering intact.
use super::{Function, TargetInst};
use regalloc2::{Block, Function as _};
use std::cmp::Reverse;

struct Edge {
    from: Block,
    to: Block,
    weight: u64,
    unconditional: bool,
}

fn edges<I: TargetInst>(f: &Function<I>) -> Vec<Edge> {
    const MAX_WEIGHT: u64 = 1 << 40;
    let mut counts = vec![0u64; f.num_blocks()];
    counts[f.entry_block().index()] = 1024;
    let mut edges = Vec::new();
    for from in (0..f.num_blocks()).map(Block::new) {
        let count = counts[from.index()].max(1);
        let current_depth = f.loop_depth(from);
        let targets = f.block_succs(from);
        let benefit = |index: usize| {
            (
                f.loop_depth(targets[index]) >= current_depth,
                f.leads_to_backedge(from, index),
            )
        };
        for (i, &to) in targets.iter().enumerate() {
            let eighths = if targets.len() == 1 {
                8
            } else {
                match benefit(i).cmp(&benefit(1 - i)) {
                    std::cmp::Ordering::Greater => 7,
                    std::cmp::Ordering::Less => 1,
                    std::cmp::Ordering::Equal => 4,
                }
            };
            let weight = (count * eighths / 8).max(1);
            if to > from {
                let entering = f.loop_depth(to).saturating_sub(current_depth).min(6);
                let target = &mut counts[to.index()];
                *target = (*target + (weight << (3 * entering))).min(MAX_WEIGHT);
            }
            edges.push(Edge {
                from,
                to,
                weight,
                unconditional: targets.len() == 1,
            });
        }
    }
    edges
}

fn locality(edge: &Edge, gap: usize, adjacent: bool) -> u128 {
    // Use selected instruction counts as a cheap proxy for byte distance.
    // Adjacent branches can drop a jump; short jumps keep related code nearby.
    const NEAR: usize = 96;
    let gain = if adjacent {
        if edge.unconditional { 105 } else { 100 }
    } else {
        10 * NEAR.saturating_sub(gap) / NEAR
    };
    u128::from(edge.weight) * gain as u128
}

fn score<I: TargetInst>(f: &Function<I>, order: &[Block], edges: &[Edge]) -> u128 {
    let mut at = 0;
    let mut offsets = vec![(0usize, 0usize); order.len()];
    for &b in order {
        let end = at + f.block_insns(b).len();
        offsets[b.index()] = (at, end);
        at = end;
    }
    edges
        .iter()
        .map(|edge| {
            let end = offsets[edge.from.index()].1;
            let target = offsets[edge.to.index()].0;
            locality(edge, end.abs_diff(target), end == target)
        })
        .sum()
}

pub(super) fn place<I: TargetInst>(f: &Function<I>) -> Vec<Block> {
    let original: Vec<_> = (0..f.num_blocks()).map(Block::new).collect();
    if original.len() < 3 {
        return original;
    }
    let mut edges = edges(f);
    // Prefer already-adjacent edges on ties. Link only chain tails to chain
    // heads. Forward links keep entry first and preserve loop headers.
    edges.sort_unstable_by_key(|e| {
        Reverse((
            locality(e, 0, true),
            e.to.index() == e.from.index() + 1,
            Reverse(e.from.index()),
            Reverse(e.to.index()),
        ))
    });
    let mut previous = vec![false; original.len()];
    let mut next = vec![None; original.len()];
    for e in &edges {
        if e.to > e.from && !previous[e.to.index()] && next[e.from.index()].is_none() {
            next[e.from.index()] = Some(e.to);
            previous[e.to.index()] = true;
        }
    }
    let mut order = Vec::with_capacity(original.len());
    for &head in &original {
        if previous[head.index()] {
            continue;
        }
        let mut block = Some(head);
        while let Some(b) = block {
            order.push(b);
            block = next[b.index()];
        }
    }
    if order != original && score(f, &order, &edges) > score(f, &original, &edges) {
        order
    } else {
        original
    }
}

#[cfg(test)]
pub(super) fn listing<I: TargetInst>(f: &Function<I>) -> String {
    use std::fmt::Write;
    let mut text = String::new();
    for b in place(f) {
        writeln!(text, "b{} depth={}", b.index(), f.loop_depth(b)).unwrap();
    }
    text
}
