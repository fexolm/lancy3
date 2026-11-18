use super::super::Instr;
use std::collections::HashSet;

// Remove unused pure instructions, including chains of unused temporaries.
// Mutable registers can have many reaching definitions. Conservatively keep
// all definitions of a required register. Loads can trap and calls can
// diverge or modify buffers even when their result is unused.
pub(super) fn remove_dead_values(code: &mut Vec<Instr>, registers: usize) {
    let observable =
        |op: &Instr| op.written().is_none() || matches!(op, Instr::Call { .. }) || op.may_trap();
    let mut definitions = vec![Vec::new(); registers];
    let mut todo = Vec::new();
    for (pc, op) in code.iter().enumerate() {
        if let Some(r) = op.written() {
            definitions[r.0 as usize].push(pc);
        }
        if observable(op) {
            op.read(|r| todo.push(r));
        }
    }
    let mut needed = vec![false; registers];
    while let Some(r) = todo.pop() {
        if std::mem::replace(&mut needed[r.0 as usize], true) {
            continue;
        }
        for &pc in &definitions[r.0 as usize] {
            code[pc].read(|r| todo.push(r));
        }
    }
    code.retain(|op| observable(op) || op.written().is_some_and(|r| needed[r.0 as usize]));
}

pub(super) fn remove_implicit_jumps(code: &mut Vec<Instr>) {
    let mut adjacent = HashSet::new();
    let mut keep = vec![true; code.len()];
    for (pc, op) in code.iter().enumerate().rev() {
        match op {
            Instr::Label(l) => {
                adjacent.insert(*l);
            }
            Instr::Jump(l) if adjacent.contains(l) => keep[pc] = false,
            _ => adjacent.clear(),
        }
    }
    let mut kept = keep.into_iter();
    code.retain(|_| kept.next().unwrap());
    let mut referenced = HashSet::new();
    for op in code.iter() {
        match op {
            Instr::Jump(l) => {
                referenced.insert(*l);
            }
            Instr::Branch { yes, no, .. } => {
                referenced.insert(*yes);
                referenced.insert(*no);
            }
            _ => {}
        }
    }
    code.retain(|op| !matches!(op, Instr::Label(l) if !referenced.contains(l)));
}
