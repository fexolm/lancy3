use super::cfg::Cfg;
use crate::input::{self, Reg};
use std::collections::BTreeMap;

struct Sites {
    defs: Vec<usize>,
    reads_before_def: Vec<usize>,
}

pub(super) fn place_parameters(input: &input::Function, cfg: &Cfg) -> Vec<Vec<Reg>> {
    let mut params = vec![Vec::new(); cfg.ranges.len()];
    let mut first_def = vec![usize::MAX; input.registers];
    first_def[..input.arity()].fill(0);
    // Typical scalar temporaries have a single definition. Allocate sparse
    // use/def lists only for variables actually assigned in distinct blocks.
    let mut candidates = BTreeMap::<u32, Sites>::new();
    for &b in &cfg.rpo {
        for op in &input.code[cfg.ranges[b].clone()] {
            if let Some(Reg(r)) = op.written() {
                let first = &mut first_def[r as usize];
                if *first == usize::MAX {
                    *first = b;
                } else if *first != b {
                    let sites = candidates.entry(r).or_insert_with(|| Sites {
                        defs: vec![*first],
                        reads_before_def: Vec::new(),
                    });
                    if sites.defs.last() != Some(&b) {
                        sites.defs.push(b);
                    }
                }
            }
        }
    }
    if candidates.is_empty() {
        return params;
    }
    let mut last_def = first_def;
    last_def.fill(usize::MAX);
    for &b in &cfg.rpo {
        for op in &input.code[cfg.ranges[b].clone()] {
            op.read(|Reg(r)| {
                if last_def[r as usize] != b
                    && let Some(sites) = candidates.get_mut(&r)
                    && sites.reads_before_def.last() != Some(&b)
                {
                    sites.reads_before_def.push(b);
                }
            });
            if let Some(Reg(r)) = op.written() {
                last_def[r as usize] = b;
            }
        }
    }
    let frontier = cfg.frontiers();
    // Stamps reuse the same scratch arrays across variables. No rectangular
    // block-by-register liveness or rename table is needed.
    let mut defined = vec![usize::MAX; cfg.ranges.len()];
    let mut live_in = defined.clone();
    let mut placed = defined.clone();
    for (variable, mut sites) in candidates {
        let stamp = variable as usize;
        for &b in &sites.defs {
            defined[b] = stamp;
        }
        while let Some(b) = sites.reads_before_def.pop() {
            if live_in[b] != stamp {
                live_in[b] = stamp;
                sites.reads_before_def.extend(
                    cfg.pred[b]
                        .iter()
                        .copied()
                        .filter(|&p| defined[p] != stamp && live_in[p] != stamp),
                );
            }
        }
        while let Some(b) = sites.defs.pop() {
            for &join in &frontier[b] {
                if live_in[join] == stamp && placed[join] != stamp {
                    placed[join] = stamp;
                    params[join].push(Reg(variable));
                    sites.defs.push(join);
                }
            }
        }
    }
    params
}
