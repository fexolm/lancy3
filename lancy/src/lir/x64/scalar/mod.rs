//! x86-64 machine emission and the internal, workspace-based call convention.
mod checked;
mod encode;
mod frame;
mod machine;
mod memory;
mod regalloc;
#[cfg(test)]
mod tests;

use crate::hir::{
    Binary,
    ssa::{Function, Op as HirOp},
};
use crate::lir::ssa::{Condition, Exit, Op, SsaBody};
use checked::Failures;
use encode::{Code, EAX, ECX, ENDBR64, comparison};
pub(crate) use frame::buffer_header_words;
use frame::{Home, LINK_SLOTS};
use machine::Machine;
use std::collections::HashMap;
use std::sync::Arc;

struct Graph {
    functions: Vec<Machine>,
    workspace: usize,
}
impl Graph {
    fn lower(root: &Function) -> Result<Self, String> {
        let mut ids = HashMap::from([(std::ptr::from_ref(root), 0)]);
        let mut source = vec![root];
        let mut functions = Vec::new();
        while let Some(&f) = source.get(functions.len()) {
            f.verify()?;
            for b in f.blocks() {
                for inst in f.instructions(b) {
                    if let HirOp::Call(c) = f.op(inst) {
                        let (target, _, _) = f.call(*c);
                        ids.entry(Arc::as_ptr(target)).or_insert_with(|| {
                            let n = source.len();
                            source.push(target);
                            n
                        });
                    }
                }
            }
            let body = SsaBody::lower(f, &ids);
            let allocation = regalloc::allocate(f, &body, functions.is_empty());
            functions.push(Machine::new(body, allocation));
        }
        // The internal ABI reuses the child workspace between sibling calls.
        let mut states = vec![0u8; functions.len()];
        let mut peaks = vec![0usize; functions.len()];
        let mut tasks = vec![(0, false)];
        while let Some((id, done)) = tasks.pop() {
            let f = &functions[id];
            if done {
                let child = f.calls.iter().map(|c| peaks[c.target]).max().unwrap_or(0);
                peaks[id] = f
                    .frame
                    .slots
                    .checked_add(child)
                    .ok_or("scalar frame overflow")?;
                states[id] = 2;
            } else {
                match states[id] {
                    2 => continue,
                    1 => return Err("recursive runtime calls require a dynamic stack".into()),
                    _ => {}
                }
                states[id] = 1;
                tasks.push((id, true));
                tasks.extend(f.calls.iter().map(|c| (c.target, false)));
            }
        }
        if peaks[0] > i32::MAX as usize / 4 || root.arity() > i32::MAX as usize / 4 {
            return Err("scalar workspace exceeds x64 displacement range".into());
        }
        Ok(Self {
            functions,
            workspace: peaks[0],
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EntryArgs {
    Slice,
    Three,
}
fn load_three_args(code: &mut Code, homes: &[Home]) {
    // Move the scratch pointer before assigning r8..r11 to virtual registers.
    // The preserved source registers below cannot be overwritten by store().
    #[cfg(windows)]
    {
        code.put(&[0x89, 0xc8]); // eax = ecx (first)
        code.put(&[0x89, 0xd1]); // ecx = edx (second)
        code.put(&[0x4c, 0x89, 0xca]); // rdx = r9 (workspace)
        code.store(homes[2], 8); // third is still in r8d
        code.store(homes[0], EAX);
        code.store(homes[1], ECX);
    }
    #[cfg(not(windows))]
    {
        code.put(&[0x89, 0xd0]); // eax = edx (third)
        code.put(&[0x48, 0x89, 0xca]); // rdx = rcx (workspace)
        code.store(homes[0], 7); // edi
        code.store(homes[1], 6); // esi
        code.store(homes[2], EAX);
    }
}

/// Scheduling uses verified block IDs, while each branch fixup names a machine
/// block (including any critical-edge bridges introduced during allocation).
fn emit_function(code: &mut Code, graph: &Graph, id: usize, abi: EntryArgs) -> Result<(), String> {
    let f = &graph.functions[id];
    let frame = &f.frame;
    if id == 0 && abi == EntryArgs::Three {
        load_three_args(code, &frame.args);
    } else {
        for (i, &dest) in frame.args.iter().enumerate() {
            if dest == Home::Dead {
                continue;
            }
            if id == 0 {
                let reg = if let Home::Register(r) = dest { r } else { EAX };
                code.memory_op(&[0x8b], reg, ECX, i);
                code.store(dest, reg);
            } else {
                code.assign(dest, Home::Slot((LINK_SLOTS + i) as u32));
            }
        }
    }
    let returns = f.returns();
    let mut order = Vec::new();
    let mut seen = vec![false; f.blocks.len()];
    let mut queue = vec![0];
    while let Some(b) = queue.pop() {
        if std::mem::replace(&mut seen[b], true) {
            continue;
        }
        order.push(b);
        if returns[b].is_some() {
            continue;
        }
        match f.blocks[b].exit {
            Exit::Jump(n) if returns[n].is_none() => queue.push(n),
            Exit::Branch(_, yes, no) => {
                // Prefer a computing arm as fallthrough over an early return.
                if returns[yes].is_some() {
                    queue.extend([yes, no]);
                } else {
                    queue.extend([no, yes]);
                }
            }
            _ => {}
        }
    }
    let mut offsets = vec![0; f.blocks.len()];
    let mut branches = Vec::new();
    let mut failures = Failures::default();
    for (index, &b) in order.iter().enumerate() {
        offsets[b] = code.bytes.len();
        if let Some(v) = returns[b] {
            code.return_value(v, id == 0);
            continue;
        }
        let block = &f.blocks[b];
        for &op in &f.ops[block.ops.clone()] {
            match op {
                Op::Imm(..) => unreachable!("literal materialization occurs at the use"),
                Op::Move(dst, src) => code.assign(dst, src),
                Op::Alu(dst, kind @ (Binary::Div | Binary::Rem), a, b) => {
                    code.divide(kind, dst, a, b, &mut failures)?
                }
                Op::Alu(dst, kind, a, b) => code.binary(kind, dst, a, b),
                Op::Len(dst, b) => code.buffer_len(frame.buffer_slot(b.index()), dst),
                Op::Load(dst, b, i) => {
                    failures
                        .bounds
                        .push(code.load_element(frame.buffer_slot(b.index()), i, dst))
                }
                Op::Store(b, i, v) => {
                    failures
                        .bounds
                        .push(code.store_element(frame.buffer_slot(b.index()), i, v))
                }
                Op::Call(dst, index) => {
                    let site = &f.calls[index as usize];
                    let child = &graph.functions[site.target].frame;
                    // A child with status has a header (status pointer followed by buffers).
                    if child.has_status {
                        code.pass_header(frame, child, &site.buffers);
                    }
                    code.call(
                        frame,
                        site.target,
                        child,
                        dst,
                        &site.args,
                        f.saves[index as usize],
                    )?;
                }
            }
        }
        let next = order.get(index + 1).copied();
        match block.exit {
            Exit::Return(v) => code.return_value(v, id == 0),
            Exit::Jump(n) => {
                if let Some(v) = returns[n] {
                    code.return_value(v, id == 0);
                } else if Some(n) != next {
                    branches.push((code.rel32(&[0xe9]), n));
                }
            }
            Exit::Branch(c, yes, no) => {
                let cc = match c {
                    Condition::Nonzero(v) => {
                        code.cmp(v, Home::Immediate(0));
                        5
                    }
                    Condition::Compare(op, a, b) => {
                        code.cmp(a, b);
                        comparison(op).unwrap()
                    }
                    Condition::Mask {
                        source,
                        mask,
                        nonzero,
                    } => {
                        code.test_mask(source, mask);
                        if nonzero { 5 } else { 4 }
                    }
                };
                if next == Some(yes) {
                    branches.push((code.rel32(&[0x0f, 0x80 | (cc ^ 1)]), no));
                } else {
                    branches.push((code.rel32(&[0x0f, 0x80 | cc]), yes));
                    if next != Some(no) {
                        branches.push((code.rel32(&[0xe9]), no));
                    }
                }
            }
        }
    }
    for (offset, b) in branches {
        code.patch(offset, offsets[b])?;
    }
    code.failure_exits(frame, failures)
}
pub(crate) fn emit_ssa(function: &Function) -> Result<(Vec<u8>, usize), String> {
    emit_with_abi(function, EntryArgs::Slice)
}
pub(crate) fn emit_ssa3(function: &Function) -> Result<(Vec<u8>, usize), String> {
    if function.arity() != 3 || function.needs_status() {
        return Err("typed x64 entry expects 3 scalar arguments and no error-status frame".into());
    }
    emit_with_abi(function, EntryArgs::Three)
}
fn emit_with_abi(f: &Function, abi: EntryArgs) -> Result<(Vec<u8>, usize), String> {
    let graph = Graph::lower(f)?;
    let mut code = Code::default();
    code.put(&ENDBR64);
    #[cfg(not(windows))]
    if abi == EntryArgs::Slice {
        code.put(&[0x48, 0x89, 0xf9, 0x48, 0x89, 0xf2]);
    }
    let mut entries = Vec::with_capacity(graph.functions.len());
    for id in 0..graph.functions.len() {
        entries.push(code.bytes.len());
        emit_function(&mut code, &graph, id, abi)?;
    }
    for (offset, id) in std::mem::take(&mut code.calls) {
        code.patch(offset, entries[id])?;
    }
    Ok((code.bytes, graph.workspace))
}
