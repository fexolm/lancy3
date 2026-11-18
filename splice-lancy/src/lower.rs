use lancy::input as native;
use splice::hir::{Binary, Buffer, BufferKind, Function, Instr, Label, Reg};
use std::collections::HashMap;
use std::sync::Arc;

/// Resolve mutable Splice registers and control flow into a Lancy SSA graph.
pub fn lower(root: &Function) -> Result<lancy::hir::Function, String> {
    lancy::hir::from_input(&import(root)?)
}

fn import(root: &Function) -> Result<native::Function, String> {
    let mut compiled = HashMap::new();
    let mut pending = vec![(root, false)];
    while let Some((function, ready)) = pending.pop() {
        let key = std::ptr::from_ref(function);
        if compiled.contains_key(&key) {
            continue;
        }
        if !ready {
            pending.push((function, true));
            for op in function.instructions() {
                if let Instr::Call { function, .. } = op {
                    pending.push((function, false));
                }
            }
            continue;
        }
        let ops = function
            .instructions()
            .iter()
            .map(|op| match op {
                Instr::Const(d, n) => native::Instr::Const(reg(*d), *n),
                Instr::Copy(d, v) => native::Instr::Copy(reg(*d), reg(*v)),
                Instr::Bin(d, op, a, b) => {
                    native::Instr::Bin(reg(*d), binary(*op), reg(*a), reg(*b))
                }
                Instr::BufferLen { dst, buffer: b } => native::Instr::BufferLen {
                    dst: reg(*dst),
                    buffer: buffer(*b),
                },
                Instr::Load {
                    dst,
                    buffer: b,
                    index,
                } => native::Instr::Load {
                    dst: reg(*dst),
                    buffer: buffer(*b),
                    index: reg(*index),
                },
                Instr::Store {
                    buffer: b,
                    index,
                    value,
                } => native::Instr::Store {
                    buffer: buffer(*b),
                    index: reg(*index),
                    value: reg(*value),
                },
                Instr::Label(l) => native::Instr::Label(label(*l)),
                Instr::Jump(l) => native::Instr::Jump(label(*l)),
                Instr::Branch { cond, yes, no } => native::Instr::Branch {
                    cond: reg(*cond),
                    yes: label(*yes),
                    no: label(*no),
                },
                Instr::Return(v) => native::Instr::Return(reg(*v)),
                Instr::Call {
                    dst,
                    function,
                    args,
                    buffers,
                } => native::Instr::Call {
                    dst: reg(*dst),
                    function: Arc::clone(&compiled[&Arc::as_ptr(function)]),
                    args: args.iter().copied().map(reg).collect(),
                    buffers: buffers.iter().copied().map(buffer).collect(),
                },
            })
            .collect();
        let kinds = function
            .buffer_kinds()
            .iter()
            .map(|k| match k {
                BufferKind::Read => native::BufferKind::Read,
                BufferKind::Write => native::BufferKind::Write,
            })
            .collect();
        let native = native::Function::from_instructions(
            function.arity(),
            kinds,
            function.register_count(),
            function.label_count(),
            ops,
        )?;
        if std::ptr::eq(function, root) {
            return Ok(native);
        }
        compiled.insert(key, Arc::new(native));
    }
    unreachable!("the traversal always includes the root")
}

fn reg(r: Reg) -> native::Reg {
    native::Reg(r.0)
}
fn label(l: Label) -> native::Label {
    native::Label(l.0)
}
fn buffer(b: Buffer) -> native::Buffer {
    native::Buffer(b.0)
}

pub(super) fn binary(op: Binary) -> native::Binary {
    use native::Binary as N;
    match op {
        Binary::Add => N::Add,
        Binary::Sub => N::Sub,
        Binary::Mul => N::Mul,
        Binary::Div => N::Div,
        Binary::Rem => N::Rem,
        Binary::And => N::And,
        Binary::Or => N::Or,
        Binary::Xor => N::Xor,
        Binary::Eq => N::Eq,
        Binary::Ne => N::Ne,
        Binary::Lt => N::Lt,
        Binary::Le => N::Le,
        Binary::Gt => N::Gt,
        Binary::Ge => N::Ge,
        Binary::Shl => N::Shl,
        Binary::Shr => N::Shr,
        Binary::Min => N::Min,
        Binary::Max => N::Max,
    }
}
