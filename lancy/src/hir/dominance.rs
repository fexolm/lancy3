//! Iterative CFG analyses for SSA dominance verification.

pub(super) fn reverse_postorder(entry: usize, successors: &[[Option<usize>; 2]]) -> Vec<usize> {
    let mut seen = vec![false; successors.len()];
    let mut pending = vec![(entry, false)];
    let mut order = Vec::new();
    while let Some((block, exiting)) = pending.pop() {
        if exiting {
            order.push(block);
        } else if !std::mem::replace(&mut seen[block], true) {
            pending.push((block, true));
            pending.extend(
                successors[block]
                    .into_iter()
                    .rev()
                    .flatten()
                    .map(|b| (b, false)),
            );
        }
    }
    order.reverse();
    order
}

/// Cooper-Harvey-Kennedy immediate dominators. The entry dominates itself;
/// unreachable arena slots retain usize::MAX.
pub(super) fn immediate(order: &[usize], predecessors: &[Vec<usize>]) -> Vec<usize> {
    let mut rank = vec![usize::MAX; predecessors.len()];
    let mut idom = vec![usize::MAX; predecessors.len()];
    for (i, &block) in order.iter().enumerate() {
        rank[block] = i;
    }
    idom[order[0]] = order[0];
    loop {
        let mut changed = false;
        for &block in &order[1..] {
            let mut preds = predecessors[block]
                .iter()
                .copied()
                .filter(|&p| idom[p] != usize::MAX);
            let mut common = preds
                .next()
                .expect("reachable block has processed predecessor in RPO");
            for mut other in preds {
                while common != other {
                    if rank[common] > rank[other] {
                        common = idom[common];
                    } else {
                        other = idom[other];
                    }
                }
            }
            changed |= idom[block] != common;
            idom[block] = common;
        }
        if !changed {
            return idom;
        }
    }
}
