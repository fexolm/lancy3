use crate::hir::ssa::dominance;
use crate::input::{self, Instr};
use std::ops::Range;

pub(super) struct Cfg {
    // Index 0 represents invocation, index 1 the first source instruction.
    pub(super) ranges: Vec<Range<usize>>,
    pub(super) succ: Vec<[Option<usize>; 2]>,
    pub(super) pred: Vec<Vec<usize>>,
    pub(super) rpo: Vec<usize>,
    pub(super) idom: Vec<usize>,
}

impl Cfg {
    pub(super) fn new(input: &input::Function) -> Self {
        let code = input.instructions();
        let mut leaders = vec![0];
        for i in 1..code.len() {
            if matches!(code[i], Instr::Label(_))
                || matches!(
                    code[i - 1],
                    Instr::Jump(_) | Instr::Branch { .. } | Instr::Return(_)
                )
            {
                leaders.push(i);
            }
        }
        leaders.push(code.len());
        let mut ranges = Vec::with_capacity(leaders.len());
        ranges.push(0..0);
        ranges.extend(leaders.windows(2).map(|w| w[0]..w[1]));
        let mut succ = vec![[None; 2]; ranges.len()];
        succ[0][0] = Some(1);
        let target = |label: input::Label| {
            Some(
                1 + leaders
                    .binary_search(&input.targets[label.0 as usize])
                    .unwrap(),
            )
        };
        for (b, range) in ranges.iter().enumerate().skip(1) {
            succ[b] = match code[range.end - 1] {
                Instr::Jump(l) => [target(l), None],
                Instr::Branch { yes, no, .. } => [target(yes), target(no)],
                Instr::Return(_) => [None, None],
                _ => [(b + 1 < ranges.len()).then_some(b + 1), None],
            };
        }
        let rpo = dominance::reverse_postorder(0, &succ);
        let mut pred = vec![Vec::new(); ranges.len()];
        for &b in &rpo {
            for next in succ[b].iter().flatten().copied() {
                if pred[next].last() != Some(&b) {
                    pred[next].push(b);
                }
            }
        }
        let idom = dominance::immediate(&rpo, &pred);
        Self {
            ranges,
            succ,
            pred,
            rpo,
            idom,
        }
    }

    pub(super) fn frontiers(&self) -> Vec<Vec<usize>> {
        let mut frontiers = vec![Vec::new(); self.ranges.len()];
        for &join in &self.rpo {
            if self.pred[join].len() < 2 {
                continue;
            }
            for &pred in &self.pred[join] {
                let mut b = pred;
                while b != self.idom[join] {
                    if frontiers[b].last() == Some(&join) {
                        break;
                    }
                    frontiers[b].push(join);
                    b = self.idom[b];
                }
            }
        }
        frontiers
    }
}
