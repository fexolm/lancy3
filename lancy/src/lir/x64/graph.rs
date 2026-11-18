use super::{machine::Machine, regalloc};
use crate::{
    CompileError,
    hir::{Function, Op},
    lir::{FunctionId, ssa::SsaBody},
    support::slotmap::DenseMap,
};
use std::{collections::HashMap, sync::Arc};

pub(super) struct Graph {
    pub(super) functions: DenseMap<FunctionId, Machine>,
    pub(super) root: FunctionId,
    pub(super) workspace: usize,
}
impl Graph {
    pub(super) fn lower(root: &Function) -> Result<Self, CompileError> {
        let mut source = DenseMap::new();
        let root_id = source.insert(root);
        let mut ids = HashMap::from([(std::ptr::from_ref(root), root_id)]);
        let mut functions = DenseMap::new();
        // Discovery appends new bodies to source. Lower them in that same order so
        // a call keeps its FunctionId through allocation and machine emission.
        while source.len() > functions.len() {
            let id = FunctionId::from_index(functions.len());
            let f = source[id];
            f.verify()?;
            for b in f.blocks() {
                for inst in f.instructions(b) {
                    if let Op::Call(c) = f.op(inst) {
                        let (target, _) = f.call(*c);
                        ids.entry(Arc::as_ptr(target))
                            .or_insert_with(|| source.insert(target));
                    }
                }
            }
            let body = SsaBody::lower(f, &ids);
            let allocation = regalloc::allocate(f, &body, id == root_id);
            let machine_id = functions.insert(Machine::new(body, allocation));
            debug_assert_eq!(machine_id, id);
        }
        // The internal ABI reuses the child workspace between sibling calls.
        let mut states = DenseMap::filled(functions.len(), Visit::New);
        let mut peaks = DenseMap::filled(functions.len(), 0usize);
        let mut tasks = vec![(root_id, false)];
        while let Some((id, done)) = tasks.pop() {
            let f = &functions[id];
            if done {
                let child = f.calls.values().map(|c| peaks[c.target]).max().unwrap_or(0);
                peaks[id] = f
                    .frame
                    .slots
                    .checked_add(child)
                    .ok_or(CompileError::FrameOverflow)?;
                states[id] = Visit::Done;
            } else {
                match states[id] {
                    Visit::Done => continue,
                    Visit::Active => return Err(CompileError::RecursiveCall),
                    Visit::New => {}
                }
                states[id] = Visit::Active;
                tasks.push((id, true));
                tasks.extend(f.calls.values().map(|c| (c.target, false)));
            }
        }
        let workspace = peaks[root_id];
        let slots = workspace.max(root.arity());
        if slots > i32::MAX as usize / 8 {
            return Err(CompileError::WorkspaceTooLarge { slots });
        }
        Ok(Self {
            functions,
            root: root_id,
            workspace,
        })
    }
}

#[derive(Clone, Copy)]
enum Visit {
    New,
    Active,
    Done,
}
