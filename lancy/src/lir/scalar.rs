//! Layout of scalar control flow during lowering. HIR already names registers
//! and labels explicitly. The layout annotates that stream without copying it.

use crate::hir::{Function, Instr, Label, Reg};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReturnValue {
    Reg(Reg),
    Constant(i32),
}

// A backwards pass identifies suffixes whose only observable action is a
// return. Substitute copies and constants, including across forward joins.
// Backedges, calls and data-dependent operations stop the scan.
fn return_paths(f: &Function) -> Vec<Option<ReturnValue>> {
    let mut paths = vec![None; f.code.len() + 1];
    for (pc, op) in f.code.iter().enumerate().rev() {
        let following = paths[pc + 1];
        paths[pc] = match op {
            Instr::Return(reg) => Some(ReturnValue::Reg(*reg)),
            Instr::Label(_) => following,
            Instr::Jump(target) => paths[f.targets[target.0 as usize]],
            Instr::Const(dst, value) if following == Some(ReturnValue::Reg(*dst)) => {
                Some(ReturnValue::Constant(*value))
            }
            Instr::Copy(dst, src) if following == Some(ReturnValue::Reg(*dst)) => {
                Some(ReturnValue::Reg(*src))
            }
            Instr::Const(..) | Instr::Copy(..) => following,
            _ => None,
        };
    }
    paths
}

// Put early-return arms at the end of the native function so a sequence of
// guards has a straight-through path. Only move an arm bounded by the
// branch and its immediately adjacent alternative; other label entries stay put.
fn early_exits(f: &Function, paths: &[Option<ReturnValue>]) -> Vec<Option<(ReturnValue, Label)>> {
    let mut exits = vec![None; f.targets.len()];
    for (pc, instr) in f.code.iter().enumerate() {
        let Instr::Branch { yes, no, .. } = instr else {
            continue;
        };
        let Some(Instr::Label(first)) = f.code.get(pc + 1) else {
            continue;
        };
        let Some(value) = paths[pc + 1] else { continue };
        let other = if first == yes {
            no
        } else if first == no {
            yes
        } else {
            continue;
        };
        let next = f.code[pc + 2..].iter().find_map(|op| match op {
            Instr::Label(l) => Some(l),
            _ => None,
        });
        if next == Some(other) {
            exits[first.0 as usize] = Some((value, *other));
        }
    }
    exits
}

pub(super) struct Layout {
    pub(super) returns: Vec<Option<ReturnValue>>,
    pub(super) deferred: Vec<Option<(ReturnValue, Label)>>,
    pub(super) branch_entries: Vec<bool>,
}

impl Layout {
    pub(super) fn new(f: &Function) -> Self {
        let paths = return_paths(f);
        let exits = early_exits(f, &paths);
        let mut entry = vec![false; f.targets.len()];
        for (pc, op) in f.code.iter().enumerate() {
            match op {
                Instr::Branch { yes, no, .. } => {
                    entry[yes.0 as usize] = true;
                    entry[no.0 as usize] = true;
                }
                Instr::Jump(target) if paths[pc].is_none() => entry[target.0 as usize] = true,
                _ => {}
            }
        }
        Self {
            returns: paths,
            deferred: exits,
            branch_entries: entry,
        }
    }

    pub(super) fn fallthrough(&self, next: Option<&Instr>) -> Option<Label> {
        match next {
            Some(Instr::Label(l)) => {
                Some(self.deferred[l.0 as usize].map_or(*l, |(_, target)| target))
            }
            _ => None,
        }
    }
}
