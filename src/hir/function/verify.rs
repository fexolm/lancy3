//! Resolve labels, check calls and register indices, and verify definite initialization.

use super::{BufferKind, Function, Instr, Label, buffer};
use std::collections::VecDeque;

pub(super) fn finish(
    arity: usize,
    buffer_kinds: Vec<BufferKind>,
    registers: usize,
    labels: usize,
    code: Vec<Instr>,
) -> Result<Function, String> {
    if registers > u32::MAX as usize {
        return Err("too many scalar registers".into());
    }
    if buffer_kinds.len() > u32::MAX as usize {
        return Err("too many buffer parameters".into());
    }
    let mut targets = vec![usize::MAX; labels];
    for (i, op) in code.iter().enumerate() {
        if let Instr::Label(l) = op {
            let slot = targets.get_mut(l.0 as usize).ok_or("invalid label index")?;
            if *slot != usize::MAX {
                return Err(format!("duplicate label {l:?}"));
            }
            *slot = i;
        }
        let mut bad = false;
        op.read(|v| bad |= v.0 as usize >= registers);
        if bad || op.written().is_some_and(|v| v.0 as usize >= registers) {
            return Err(format!("invalid register at instruction {i}"));
        }
        match op {
            Instr::BufferLen { buffer, .. }
            | Instr::Load { buffer, .. }
            | Instr::Store { buffer, .. } => {
                let kind = buffer_kinds
                    .get(buffer.index())
                    .ok_or_else(|| format!("invalid buffer at instruction {i}"))?;
                if matches!(op, Instr::Store { .. }) && *kind != BufferKind::Write {
                    return Err(format!("store to read-only buffer at instruction {i}"));
                }
            }
            Instr::Call {
                function,
                args,
                buffers,
                ..
            } => {
                if function.arity != args.len() {
                    return Err(format!(
                        "call at instruction {i} expects {} arguments, got {}",
                        function.arity,
                        args.len()
                    ));
                }
                buffer::check_forwarding(&buffer_kinds, function.buffer_kinds(), buffers)
                    .map_err(|e| format!("call at instruction {i}: {e}"))?;
            }
            _ => {}
        }
    }
    for op in &code {
        let check = |l: Label| {
            targets
                .get(l.0 as usize)
                .filter(|p| **p != usize::MAX)
                .ok_or_else(|| format!("undefined label {l:?}"))
        };
        match op {
            Instr::Jump(l) => {
                check(*l)?;
            }
            Instr::Branch { yes, no, .. } => {
                check(*yes)?;
                check(*no)?;
            }
            _ => {}
        }
    }
    let mut function = Function {
        arity,
        buffer_kinds,
        needs_status: false,
        registers,
        code,
        targets,
    };
    function.refresh_status();
    check_initialization(&function)?;
    Ok(function)
}

/// Forward must-analysis. A register at a join must have been assigned on
/// every reachable incoming edge, including edges from loop back to header.
fn check_initialization(f: &Function) -> Result<(), String> {
    if f.code.is_empty() {
        return Err("empty scalar function".into());
    }
    let mut leaders = vec![0];
    for (i, op) in f.code.iter().enumerate().skip(1) {
        if matches!(op, Instr::Label(_))
            || matches!(
                f.code[i - 1],
                Instr::Return(_) | Instr::Branch { .. } | Instr::Jump(_)
            )
        {
            leaders.push(i);
        }
    }
    let target_block = |l: Label| leaders.binary_search(&f.targets[l.0 as usize]).unwrap();
    let mut entry = vec![0u64; f.registers.div_ceil(64)];
    for r in 0..f.arity {
        entry[r / 64] |= 1 << (r % 64);
    }
    let mut states = vec![None; leaders.len()];
    states[0] = Some(entry);
    let mut queue = VecDeque::from([0]);
    while let Some(block) = queue.pop_front() {
        let mut bits = states[block].as_ref().unwrap().clone();
        let end = leaders.get(block + 1).copied().unwrap_or(f.code.len());
        for pc in leaders[block]..end {
            let op = &f.code[pc];
            let mut missing = None;
            op.read(|r| {
                if bits[r.0 as usize / 64] & (1 << (r.0 % 64)) == 0 {
                    missing = Some(r);
                }
            });
            if let Some(r) = missing {
                return Err(format!("uninitialized {r:?} at instruction {pc}"));
            }
            if let Some(r) = op.written() {
                bits[r.0 as usize / 64] |= 1 << (r.0 % 64);
            }
        }
        let mut propagate = |next: usize| {
            if let Some(existing) = &mut states[next] {
                let mut changed = false;
                for (a, b) in existing.iter_mut().zip(&bits) {
                    let common = *a & b;
                    changed |= *a != common;
                    *a = common;
                }
                if changed {
                    queue.push_back(next);
                }
            } else {
                states[next] = Some(bits.clone());
                queue.push_back(next);
            }
        };
        match f.code[end - 1] {
            Instr::Return(_) => {}
            Instr::Branch { yes, no, .. } => {
                propagate(target_block(yes));
                propagate(target_block(no));
            }
            Instr::Jump(l) => propagate(target_block(l)),
            _ if end < f.code.len() => propagate(block + 1),
            _ => return Err("reachable path falls off the end of the function".into()),
        }
    }
    Ok(())
}
