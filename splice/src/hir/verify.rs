//! Resolve labels, check calls and register indices, and verify definite initialization.

use super::{Function, Instr, Label, ValueType};
use crate::simd::Element;
use std::collections::VecDeque;

impl Function {
    pub(super) fn verify(mut self) -> Result<Self, String> {
        if self.register_count() > u32::MAX as usize {
            return Err("too many registers".into());
        }
        if self.buffer_kinds.len() > u32::MAX as usize {
            return Err("too many buffer parameters".into());
        }
        if self.code.is_empty() {
            return Err("empty scalar function".into());
        }
        let mut targets = vec![None; self.labels];
        let mut starts = Vec::new();
        let mut new_block = true;
        for (i, op) in self.code.iter().enumerate() {
            if new_block || matches!(op, Instr::Label(_)) {
                starts.push(i);
            }
            new_block = matches!(op, Instr::Return(_) | Instr::Branch { .. } | Instr::Jump(_));
            if let Instr::Label(l) = op {
                let slot = targets.get_mut(l.0 as usize).ok_or("invalid label index")?;
                if slot.replace(starts.len() - 1).is_some() {
                    return Err(format!("duplicate label {l:?}"));
                }
            }
            if let Instr::Call {
                dst,
                function,
                args,
                buffers,
            } = op
            {
                super::check_call(
                    function.arity(),
                    function.buffer_kinds(),
                    args.iter().map(|&r| self.checked_reg_type(r)),
                    &self.buffer_kinds,
                    buffers,
                )
                .map_err(|e| format!("call at instruction {i}: {e}"))?;
                if self.checked_reg_type(*dst) != Some(ValueType::Scalar) {
                    return Err(format!(
                        "invalid result register for call at instruction {i}"
                    ));
                }
                continue;
            }
            if op
                .reads()
                .chain(op.written())
                .any(|r| r.0 as usize >= self.register_count())
            {
                return Err(format!("invalid register at instruction {i}"));
            }
            match op {
                Instr::Debug(point) => {
                    let location = point.location;
                    if location.file.is_empty()
                        || location.directory.is_empty()
                        || location.file.contains('\0')
                        || location.directory.contains('\0')
                        || location.line == 0
                    {
                        return Err(format!("invalid debug source location at instruction {i}"));
                    }
                    if point.variables.iter().any(|var| {
                        matches!(var.value, crate::debug::Value::Slice(buffer)
                            if buffer.index() >= self.buffer_kinds.len())
                    }) {
                        return Err(format!("invalid debug buffer at instruction {i}"));
                    }
                }
                Instr::Simd { dst, inst, args } => {
                    let types: Vec<_> = args.iter().map(|&r| self.reg_type(r)).collect();
                    inst.validate(&types, &self.buffer_kinds)?;
                    if dst.map(|r| self.reg_type(r)) != inst.output() {
                        return Err("invalid SIMD result type".into());
                    }
                }
                Instr::Copy(d, s) if self.reg_type(*d) != self.reg_type(*s) => {
                    return Err("register copy type mismatch".into());
                }
                Instr::Copy(..) => {}
                _ if op
                    .reads()
                    .chain(op.written())
                    .any(|r| self.reg_type(r) != ValueType::Scalar) =>
                {
                    return Err("expected a scalar register".into());
                }
                _ => {}
            }
            if let Instr::BufferLen { buffer, .. }
            | Instr::Load { buffer, .. }
            | Instr::Store { buffer, .. } = op
            {
                let kind = self
                    .buffer_kinds
                    .get(buffer.index())
                    .ok_or_else(|| format!("invalid buffer at instruction {i}"))?;
                if !matches!(op, Instr::BufferLen { .. }) && kind.element() != Element::I32 {
                    return Err("scalar element access requires an i32 buffer".into());
                }
                if matches!(op, Instr::Store { .. }) && !kind.writable() {
                    return Err(format!("store to read-only buffer at instruction {i}"));
                }
            }
        }
        starts.push(self.code.len());
        check_initialization(&self, &starts, &targets)?;
        self.needs_status = !self.buffer_kinds.is_empty() || self.code.iter().any(Instr::may_trap);
        Ok(self)
    }
}

/// Forward must-analysis. A register at a join must have been assigned on
/// every reachable incoming edge, including edges from loop back to header.
fn check_initialization(
    f: &Function,
    starts: &[usize],
    targets: &[Option<usize>],
) -> Result<(), String> {
    let target = |l: Label| {
        targets
            .get(l.0 as usize)
            .copied()
            .flatten()
            .ok_or_else(|| format!("undefined label {l:?}"))
    };
    // Resolve every edge, including jumps in unreachable blocks. Fallthrough
    // past the last block is an error only if execution can reach it.
    let mut edges = Vec::new();
    for (block, &end) in starts.iter().skip(1).enumerate() {
        edges.push(match f.code[end - 1] {
            Instr::Return(_) => [None, None],
            Instr::Branch { yes, no, .. } => [Some(target(yes)?), Some(target(no)?)],
            Instr::Jump(l) => [Some(target(l)?), None],
            _ => [Some(block + 1), None],
        });
    }
    let mut entry = vec![0u64; f.register_count().div_ceil(64)];
    for r in 0..f.arity {
        entry[r / 64] |= 1 << (r % 64);
    }
    let mut states = vec![None; edges.len()];
    states[0] = Some(entry);
    let mut queue = VecDeque::from([0]);
    while let Some(block) = queue.pop_front() {
        let mut bits = states[block].as_ref().unwrap().clone();
        for pc in starts[block]..starts[block + 1] {
            let op = &f.code[pc];
            if let Some(r) = op
                .reads()
                .find(|r| bits[r.0 as usize / 64] & (1 << (r.0 % 64)) == 0)
            {
                return Err(format!("uninitialized {r:?} at instruction {pc}"));
            }
            if let Some(r) = op.written() {
                bits[r.0 as usize / 64] |= 1 << (r.0 % 64);
            }
        }
        for next in edges[block].into_iter().flatten() {
            if next == edges.len() {
                return Err("reachable path falls off the end of the function".into());
            }
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
        }
    }
    Ok(())
}
