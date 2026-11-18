//! x86-64 machine emission and the internal, workspace-based call convention.
mod checked;
mod encode;
mod frame;
mod machine;
mod memory;
mod regalloc;
#[cfg(test)]
mod tests;

use crate::hir::{Binary, Function, Op as HirOp};
use crate::lir::ssa::{Exit, Op, SsaBody};
use checked::Failures;
use encode::{Code, EAX, ECX, ENDBR64};
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

fn emit_function(code: &mut Code, graph: &Graph, id: usize) -> Result<(), String> {
    let f = &graph.functions[id];
    let frame = &f.frame;
    for (i, &dest) in frame.args.iter().enumerate() {
        if id == 0 {
            let reg = if let Home::Register(r) = dest { r } else { EAX };
            code.memory_op(&[0x8b], reg, ECX, i);
            code.store(dest, reg);
        } else {
            code.assign(dest, Home::Slot((LINK_SLOTS + i) as u32));
        }
    }
    let mut offsets = vec![0; f.blocks.len()];
    let mut branches = Vec::new();
    let mut failures = Failures::default();
    for (b, block) in f.blocks.iter().enumerate() {
        offsets[b] = code.bytes.len();
        for &op in &f.ops[block.ops.clone()] {
            match op {
                Op::Imm(dst, n) => code.constant(dst, n),
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
                    code.pass_header(frame, child, &site.buffers);
                    code.call(frame, site.target, dst, &site.args)?;
                }
            }
        }
        match block.exit {
            Exit::Return(v) => code.return_value(v, id == 0),
            Exit::Jump(target) => branches.push((code.rel32(&[0xe9]), target)),
            Exit::Branch(cond, yes, no) => {
                code.load(EAX, cond);
                code.put(&[0x85, 0xc0]); // test eax,eax
                branches.push((code.rel32(&[0x0f, 0x85]), yes));
                branches.push((code.rel32(&[0xe9]), no));
            }
        }
    }
    for (offset, b) in branches {
        code.patch(offset, offsets[b])?;
    }
    code.failure_exits(frame, failures)
}
pub(crate) fn emit(f: &Function) -> Result<(Vec<u8>, usize), String> {
    let graph = Graph::lower(f)?;
    let mut code = Code::default();
    code.put(&ENDBR64);
    #[cfg(not(windows))]
    code.put(&[0x48, 0x89, 0xf9, 0x48, 0x89, 0xf2]);
    let mut entries = Vec::with_capacity(graph.functions.len());
    for id in 0..graph.functions.len() {
        entries.push(code.bytes.len());
        emit_function(&mut code, &graph, id)?;
    }
    for (offset, id) in std::mem::take(&mut code.calls) {
        code.patch(offset, entries[id])?;
    }
    Ok((code.bytes, graph.workspace))
}
