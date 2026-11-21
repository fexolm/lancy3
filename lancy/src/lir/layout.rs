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

fn loop_depths<I: TargetInst>(f: &Function<I>) -> Vec<u8> {
    let size = f.num_blocks();
    let mut depths = vec![0u8; size];
    let mut seen = vec![Block::invalid(); size];
    let mut stack = Vec::new();
    // In RPO a natural loop's backward edge points to its header. The
    // predecessor walk excludes exits and stops at the header. Irreducible
    // regions can give an approximation, so use this only for edge weights.
    let mut budget = size.saturating_mul(8);
    for header in (0..size).rev().map(Block::new) {
        stack.extend(
            f.block_preds(header)
                .iter()
                .copied()
                .filter(|&b| b >= header),
        );
        while let Some(block) = stack.pop() {
            if seen[block.index()] == header {
                continue;
            }
            if budget == 0 {
                return depths;
            }
            budget -= 1;
            seen[block.index()] = header;
            depths[block.index()] = (depths[block.index()] + 1).min(6);
            if block != header {
                stack.extend(
                    f.block_preds(block)
                        .iter()
                        .copied()
                        .filter(|&b| b >= header),
                );
            }
        }
    }
    depths
}

fn edges<I: TargetInst>(f: &Function<I>, depth: &[u8]) -> Vec<Edge> {
    const MAX_WEIGHT: u64 = 1 << 40;
    let mut counts = vec![0u64; f.num_blocks()];
    counts[0] = 1024u64 << (3 * depth[0]);
    let mut edges = Vec::new();
    for from in (0..f.num_blocks()).map(Block::new) {
        let count = counts[from.index()].max(1);
        let current_depth = depth[from.index()];
        let targets = f.block_succs(from);
        let benefit = |to: Block| (depth[to.index()] >= current_depth, to <= from);
        for (i, &to) in targets.iter().enumerate() {
            let eighths = if targets.len() == 1 {
                8
            } else {
                match benefit(to).cmp(&benefit(targets[1 - i])) {
                    std::cmp::Ordering::Greater => 7,
                    std::cmp::Ordering::Less => 1,
                    std::cmp::Ordering::Equal => 4,
                }
            };
            let weight = (count * eighths / 8).max(1);
            if to > from {
                let entering = depth[to.index()].saturating_sub(current_depth);
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
    let mut edges = edges(f, &loop_depths(f));
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
