//! Register assignment and host workspace layout for a static call graph.

use crate::hir::{Function, Instr};
use std::collections::HashMap;

pub(super) const LINK_SLOTS: usize = 2; // one 64-bit continuation in i32 slots

/// Header of a status-bearing frame, including a status pointer even when
/// the function has no buffer parameters.
pub(crate) fn buffer_header_words(count: usize) -> usize {
    2 * (1 + count)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Home {
    Register(u8),
    Slot(usize),
    Immediate(i32),
    Dead,
}

#[derive(Default)]
struct Uses {
    reads: usize,
    definitions: usize,
    literal: Option<i32>,
}

pub(super) struct Frame {
    pub(super) homes: Vec<Home>,
    pub(super) reads: Vec<usize>,
    pub(super) slots: usize,
    pub(super) saves: Vec<(u8, usize)>,
    // Memory functions carry a shared error-status pointer followed by
    // their local descriptor pointers. A child also has a link and arguments.
    pub(super) status_slot: usize,
}

impl Frame {
    pub(super) fn buffer_slot(&self, buffer: usize) -> usize {
        self.status_slot + 2 + 2 * buffer
    }
}

pub(super) fn frame(function: &Function, root: bool) -> Frame {
    let mut usage: Vec<Uses> = (0..function.registers).map(|_| Uses::default()).collect();
    for arg in &mut usage[..function.arity] {
        arg.definitions = 1;
    }
    let mut calls = false;
    for instr in &function.code {
        let write = match instr {
            Instr::Const(dst, n) => {
                usage[dst.0 as usize].literal = Some(*n);
                Some(dst)
            }
            Instr::Copy(dst, src) => {
                usage[src.0 as usize].reads += 1;
                Some(dst)
            }
            Instr::Bin(dst, _, a, b) => {
                usage[a.0 as usize].reads += 1;
                usage[b.0 as usize].reads += 1;
                Some(dst)
            }
            Instr::BufferLen { dst, .. } => Some(dst),
            Instr::Load { dst, index, .. } => {
                usage[index.0 as usize].reads += 1;
                Some(dst)
            }
            Instr::Store { index, value, .. } => {
                usage[index.0 as usize].reads += 1;
                usage[value.0 as usize].reads += 1;
                None
            }
            Instr::Call { dst, args, .. } => {
                calls = true;
                for arg in args {
                    usage[arg.0 as usize].reads += 1;
                }
                Some(dst)
            }
            Instr::Branch { cond: v, .. } | Instr::Return(v) => {
                usage[v.0 as usize].reads += 1;
                None
            }
            Instr::Label(_) | Instr::Jump(_) => None,
        };
        if let Some(dst) = write {
            usage[dst.0 as usize].definitions += 1;
        }
    }

    let mut homes = vec![Home::Dead; function.registers];
    let mut live = Vec::new();
    for (i, u) in usage.iter().enumerate() {
        if u.reads > 0 {
            if let (1, Some(n)) = (u.definitions, u.literal) {
                homes[i] = Home::Immediate(n);
            } else {
                live.push(i);
            }
        }
    }
    // Four permanently assigned caller-saved registers work across arbitrary
    // backedges. Other mutable values reside in the caller-provided workspace.
    let best = live.len().min(4);
    let rank = |i: &usize| (std::cmp::Reverse(usage[*i].reads), *i);
    if best < live.len() {
        live.select_nth_unstable_by_key(best, rank);
    }
    let status_slot = if root { 0 } else { LINK_SLOTS + function.arity };
    let mut slots = status_slot;
    if function.needs_status() {
        slots += buffer_header_words(function.buffer_kinds().len());
    }
    let mut saves = Vec::new();
    for (k, i) in live.into_iter().enumerate() {
        homes[i] = if k < best {
            let reg = 8 + k as u8;
            if calls {
                saves.push((reg, slots));
                slots += 1;
            }
            Home::Register(reg)
        } else if !root && i < function.arity {
            Home::Slot(LINK_SLOTS + i)
        } else {
            let slot = slots;
            slots += 1;
            Home::Slot(slot)
        };
    }
    Frame {
        homes,
        reads: usage.into_iter().map(|u| u.reads).collect(),
        slots,
        saves,
        status_slot,
    }
}

pub(super) struct Graph<'a> {
    pub(super) functions: Vec<&'a Function>,
    pub(super) ids: HashMap<*const Function, usize>,
    pub(super) frames: Vec<Frame>,
}

impl<'a> Graph<'a> {
    pub(super) fn discover(root: &'a Function) -> Self {
        let mut functions = vec![root];
        let mut ids = HashMap::from([(std::ptr::from_ref(root), 0)]);
        let mut frames = Vec::new();
        let mut i = 0;
        while i < functions.len() {
            let f = functions[i];
            frames.push(frame(f, i == 0));
            for instr in &f.code {
                if let Instr::Call { function, .. } = instr {
                    ids.entry(std::sync::Arc::as_ptr(function))
                        .or_insert_with(|| {
                            let next = functions.len();
                            functions.push(function);
                            next
                        });
                }
            }
            i += 1;
        }
        Self {
            functions,
            ids,
            frames,
        }
    }

    fn children(&self, id: usize) -> impl Iterator<Item = usize> + '_ {
        self.functions[id]
            .code
            .iter()
            .filter_map(|instr| match instr {
                Instr::Call { function, .. } => Some(self.ids[&std::sync::Arc::as_ptr(function)]),
                _ => None,
            })
    }

    pub(super) fn workspace(&self) -> Result<usize, String> {
        // Static nonrecursive call graph. Size for the deepest live call chain,
        // reusing sibling frames. Iterative traversal also handles deep DAGs.
        let mut state = vec![0; self.functions.len()];
        let mut peak = vec![0usize; self.functions.len()];
        let mut todo = vec![(0, false)];
        while let Some((id, exiting)) = todo.pop() {
            if exiting {
                let child = self.children(id).map(|c| peak[c]).max().unwrap_or(0);
                peak[id] = self.frames[id]
                    .slots
                    .checked_add(child)
                    .ok_or("scalar frame size overflow")?;
                state[id] = 2;
            } else if state[id] == 0 {
                state[id] = 1;
                todo.push((id, true));
                todo.extend(self.children(id).map(|c| (c, false)));
            } else if state[id] == 1 {
                return Err("recursive runtime call graph requires a dynamic stack".into());
            }
        }
        if peak[0] > i32::MAX as usize / 4 || self.functions[0].arity > i32::MAX as usize / 4 {
            return Err("scalar workspace or argument array exceeds x64 displacement range".into());
        }
        Ok(peak[0])
    }
}
