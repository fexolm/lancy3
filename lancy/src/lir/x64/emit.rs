//! x86-64 machine emission and the internal, workspace-based call convention.
mod code;

use super::frame::{EAX, ECX, Home, LINK_SLOTS};
use super::graph::Graph;
use crate::CompileError;
use crate::hir::{Binary, Function};
use crate::lir::{FunctionId, block::Exit, instruction::Op};
use crate::support::slotmap::DenseMap;
use code::{Code, ENDBR64, Failures};

fn emit_function(code: &mut Code, graph: &Graph, id: FunctionId) -> Result<(), CompileError> {
    let f = &graph.functions[id];
    let frame = &f.frame;
    for (i, &dest) in frame.args.iter().enumerate() {
        if id == graph.root {
            let reg = if let Home::Register(r) = dest { r } else { EAX };
            code.memory_op(&[0x8b], reg, ECX, i);
            code.store(dest, reg);
        } else {
            code.assign(dest, Home::Slot((LINK_SLOTS + i) as u32));
        }
    }
    let mut offsets = DenseMap::filled(f.blocks.len(), 0);
    let mut branches = Vec::new();
    let mut failures = Failures::default();
    for (b, block) in f.blocks.iter() {
        offsets[b] = code.bytes.len();
        for &op in &f.ops[block.ops.clone()] {
            match op {
                Op::Imm(dst, n) => code.constant(dst, n),
                Op::Move(dst, src) => code.assign(dst, src),
                Op::Alu(dst, kind @ (Binary::Div | Binary::Rem), a, b) => {
                    code.divide(kind, dst, a, b, &mut failures)?
                }
                Op::Alu(dst, kind, a, b) => code.binary(kind, dst, a, b),
                Op::BoundsCheck(index, len) => failures.bounds.push(code.bounds_check(index, len)),
                Op::PtrOffset(dst, base, index, scale) => code.ptr_offset(dst, base, index, scale),
                Op::Load(dst, ty, addr) => code.load_at(dst, ty, addr),
                Op::Store(ty, addr, value) => code.store_at(ty, addr, value),
                Op::Call(dst, index) => {
                    let site = &f.calls[index];
                    let child = &graph.functions[site.target].frame;
                    code.pass_status(frame, child);
                    code.call(frame, site.target, dst, &site.args)?;
                }
            }
        }
        match block.exit {
            Exit::Return(v) => code.return_value(v, id == graph.root),
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

pub(crate) fn emit(f: &Function) -> Result<(Vec<u8>, usize), CompileError> {
    let graph = Graph::lower(f)?;
    let mut code = Code::default();
    code.put(&ENDBR64);
    #[cfg(not(windows))]
    code.put(&[0x48, 0x89, 0xf9, 0x48, 0x89, 0xf2]);
    let mut entries = DenseMap::filled(graph.functions.len(), 0);
    for id in graph.functions.keys() {
        entries[id] = code.bytes.len();
        emit_function(&mut code, &graph, id)?;
    }
    for (offset, id) in std::mem::take(&mut code.calls) {
        code.patch(offset, entries[id])?;
    }
    Ok((code.bytes, graph.workspace))
}
