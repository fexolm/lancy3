//! Profiling of live input columns to propose guarded constants.
use crate::hir::{Assumption, Op, Program};

const SAMPLE_ROWS: usize = 64;
const MAX_GUARDS: usize = 4;

// Cache only inputs that can influence the returned value. Dead builder
// expressions must not consume guard slots or cause spurious invalidations.
pub(super) fn live_inputs(program: &Program) -> Vec<u16> {
    let mut live = vec![false; program.result().index() + 1];
    let mut used = vec![false; program.input_count()];
    live[program.result().index()] = true;
    for i in (0..live.len()).rev() {
        if live[i] {
            let op = program.ops()[i];
            if let Op::Input(col) = op {
                used[usize::from(col)] = true;
            }
            for arg in op.operands() {
                live[arg.index()] = true;
            }
        }
    }
    used.into_iter()
        .enumerate()
        .filter_map(|(col, used)| used.then_some(col as u16))
        .collect()
}

pub(super) fn profile(used: &[u16], columns: &[&[i32]]) -> Vec<Assumption> {
    let mut guards = Vec::new();
    for &column in used {
        let values = columns[usize::from(column)];
        let first = values[0];
        if values.iter().take(SAMPLE_ROWS).all(|&v| v == first) {
            guards.push(Assumption {
                column,
                value: first,
            });
            if guards.len() == MAX_GUARDS {
                break;
            }
        }
    }
    guards
}
