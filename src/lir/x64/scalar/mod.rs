//! Scalar x86-64 functions with control flow and a host-owned call workspace.
//! The C entry point accepts (arguments, workspace) and returns one i32.
//! Static callees share one native body per Function. A native jump enters
//! that body with its own frame; the return continuation lives in the frame.
//! No generated instruction adjusts RSP or touches a nonvolatile register.

mod checked;
mod encode;
mod frame;
mod memory;
#[cfg(test)]
mod tests;

use crate::hir::{Binary, Function, Instr, Label, Reg};
use crate::lir::scalar::Layout;
use checked::Failures;
use encode::{Code, EAX, ECX, ENDBR64, RDX, comparison};
pub(crate) use frame::buffer_header_words;
use frame::{Frame, Graph, Home, LINK_SLOTS};
use std::collections::HashMap;

#[derive(Default)]
struct Labels {
    bound: HashMap<u32, usize>,
    fixups: Vec<(usize, u32)>,
}

impl Labels {
    fn jump(&mut self, code: &mut Code, cc: Option<u8>, to: Label) {
        let pos = match cc {
            Some(cc) => code.rel32(&[0x0f, 0x80 | cc]),
            None => code.rel32(&[0xe9]),
        };
        self.fixups.push((pos, to.0));
    }

    fn branch(
        &mut self,
        code: &mut Code,
        cc: u8,
        yes: Label,
        no: Label,
        fallthrough: Option<Label>,
    ) {
        if Some(yes) == fallthrough {
            self.jump(code, Some(cc ^ 1), no);
        } else {
            self.jump(code, Some(cc), yes);
            if Some(no) != fallthrough {
                self.jump(code, None, no);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EntryArgs {
    Slice,
    Three,
}

// Lower (x & mask) ==/!= 0 directly to TEST and Jcc. A fixed HIR literal
// between the AND and comparison is rematerialized and emits no instruction.
fn mask_branch(
    f: &Function,
    frame: &Frame,
    mut i: usize,
    masked: Reg,
) -> Option<(usize, u8, Label, Label)> {
    while let Some(Instr::Const(dst, _)) = f.code.get(i) {
        if !matches!(frame.homes[dst.0 as usize], Home::Dead | Home::Immediate(_)) {
            return None;
        }
        i += 1;
    }
    let Instr::Bin(cmp, op @ (Binary::Eq | Binary::Ne), a, b) = f.code.get(i)? else {
        return None;
    };
    let zero = if *a == masked {
        *b
    } else if *b == masked {
        *a
    } else {
        return None;
    };
    if frame.homes[zero.0 as usize] != Home::Immediate(0) || frame.reads[cmp.0 as usize] != 1 {
        return None;
    }
    let Instr::Branch { cond, yes, no } = f.code.get(i + 1)? else {
        return None;
    };
    (*cond == *cmp).then_some((i + 2, if *op == Binary::Eq { 4 } else { 5 }, *yes, *no))
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

fn body(code: &mut Code, graph: &Graph<'_>, id: usize, arguments: EntryArgs) -> Result<(), String> {
    let f = graph.functions[id];
    let frame = &graph.frames[id];
    let home = |r: Reg| frame.homes[r.0 as usize];
    if id == 0 && arguments == EntryArgs::Three {
        load_three_args(code, &frame.homes);
    } else {
        for arg in 0..f.arity {
            let dest = frame.homes[arg];
            if id != 0 {
                code.assign(dest, Home::Slot(LINK_SLOTS + arg));
            } else if dest != Home::Dead {
                let reg = match dest {
                    Home::Register(r) => r,
                    _ => EAX,
                };
                code.memory_op(&[0x8b], reg, ECX, arg);
                code.store(dest, reg);
            }
        }
    }
    let mut labels = Labels::default();
    let mut failures = Failures::default();
    let layout = Layout::new(f);
    let fallthrough = |i| layout.fallthrough(f.code.get(i));
    let mut returned = false;
    let mut i = 0;
    while let Some(instr) = f.code.get(i) {
        let path = layout.returns[i];
        i += 1;
        if let Instr::Label(label) = instr {
            if let Some((_, next)) = layout.deferred[label.0 as usize] {
                i = f.targets[next.0 as usize];
                continue;
            }
            returned &= !layout.branch_entries[label.0 as usize];
            labels.bound.insert(label.0, code.bytes.len());
            continue;
        }
        if returned {
            continue;
        }
        if let Some(value) = path {
            code.return_value(value, frame, id == 0);
            returned = true;
            continue;
        }
        match instr {
            Instr::Const(dst, n) => code.assign(home(*dst), Home::Immediate(*n)),
            Instr::Copy(dst, src) => code.assign(home(*dst), home(*src)),
            Instr::BufferLen { dst, buffer } => {
                code.buffer_len(frame.buffer_slot(buffer.index()), home(*dst));
            }
            Instr::Load { dst, buffer, index } => {
                failures.bounds.push(code.load_element(
                    frame.buffer_slot(buffer.index()),
                    home(*index),
                    home(*dst),
                ));
            }
            Instr::Store {
                buffer,
                index,
                value,
            } => {
                failures.bounds.push(code.store_element(
                    frame.buffer_slot(buffer.index()),
                    home(*index),
                    home(*value),
                ));
            }
            Instr::Bin(dst, op @ (Binary::Div | Binary::Rem), lhs, rhs) => {
                code.divide(*op, home(*dst), home(*lhs), home(*rhs), &mut failures)?;
            }
            Instr::Bin(dst, op, lhs, rhs) => {
                if *op == Binary::And
                    && frame.reads[dst.0 as usize] == 1
                    && let (source, Home::Immediate(mask)) | (Home::Immediate(mask), source) =
                        (home(*lhs), home(*rhs))
                    && let Some((next, cc, yes, no)) = mask_branch(f, frame, i, *dst)
                {
                    code.test_mask(source, mask);
                    i = next;
                    labels.branch(code, cc, yes, no, fallthrough(i));
                    continue;
                }
                // When the sole consumer is the next branch, consume the
                // processor flags directly instead of creating a boolean mask.
                if let Some(cc) = comparison(*op)
                    && let Some(Instr::Branch { cond, yes, no }) = f.code.get(i)
                    && cond.0 == dst.0
                    && frame.reads[dst.0 as usize] == 1
                {
                    code.cmp(home(*lhs), home(*rhs));
                    i += 1;
                    labels.branch(code, cc, *yes, *no, fallthrough(i));
                    continue;
                }
                code.binary(*op, home(*dst), home(*lhs), home(*rhs));
            }
            Instr::Label(_) | Instr::Return(_) => unreachable!(),
            Instr::Jump(to) => {
                if !matches!(f.code.get(i), Some(Instr::Label(l)) if l.0 == to.0) {
                    labels.jump(code, None, *to);
                }
            }
            Instr::Branch { cond, yes, no } => {
                match home(*cond) {
                    Home::Immediate(n) => labels.jump(code, None, if n != 0 { *yes } else { *no }),
                    Home::Register(r) => {
                        code.register_op(&[0x85], r, r);
                        labels.branch(code, 5, *yes, *no, fallthrough(i));
                    }
                    Home::Slot(s) => {
                        code.memory_op(&[0x83], 7, RDX, s); // cmp dword [rdx+off],0
                        code.bytes.push(0);
                        labels.branch(code, 5, *yes, *no, fallthrough(i));
                    }
                    Home::Dead => unreachable!(),
                }
            }
            Instr::Call {
                dst,
                function,
                args,
                buffers,
            } => {
                let child = graph.ids[&std::sync::Arc::as_ptr(function)];
                if function.needs_status() {
                    code.pass_header(frame, &graph.frames[child], buffers);
                }
                code.call(frame, child, &graph.frames[child], home(*dst), args)?;
            }
        }
    }
    for (label, exit) in layout.deferred.into_iter().enumerate() {
        if let Some((value, _)) = exit {
            labels.bound.insert(label as u32, code.bytes.len());
            code.return_value(value, frame, id == 0);
        }
    }
    for (offset, label) in labels.fixups {
        let target = labels
            .bound
            .get(&label)
            .ok_or("undefined scalar branch target")?;
        code.patch(offset, *target)?;
    }
    code.failure_exits(frame, failures)?;
    Ok(())
}

pub(crate) fn emit(function: &Function) -> Result<(Vec<u8>, usize), String> {
    emit_with_abi(function, EntryArgs::Slice)
}

/// extern "C" fn(i32, i32, i32, *mut i32) -> i32. Only the external
/// prologue changes; the emitted control flow and callee ABI are identical.
pub(crate) fn emit3(function: &Function) -> Result<(Vec<u8>, usize), String> {
    if function.arity != 3 || function.needs_status() {
        return Err(
            "the typed x64 entry point expects three scalar arguments and no error-status frame"
                .into(),
        );
    }
    emit_with_abi(function, EntryArgs::Three)
}

fn emit_with_abi(function: &Function, arguments: EntryArgs) -> Result<(Vec<u8>, usize), String> {
    let graph = Graph::discover(function);
    let workspace = graph.workspace()?;
    let mut code = Code::default();
    code.put(&ENDBR64);
    #[cfg(not(windows))]
    if arguments == EntryArgs::Slice {
        code.put(&[0x48, 0x89, 0xf9, 0x48, 0x89, 0xf2]); // RCX=args, RDX=scratch
    }
    let mut entries = Vec::with_capacity(graph.functions.len());
    for id in 0..graph.functions.len() {
        entries.push(code.bytes.len());
        body(&mut code, &graph, id, arguments)?;
    }
    for (offset, id) in std::mem::take(&mut code.calls) {
        code.patch(offset, entries[id])?;
    }
    Ok((code.bytes, workspace))
}
