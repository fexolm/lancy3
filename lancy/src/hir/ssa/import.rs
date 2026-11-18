//! Construct pruned SSA from a verified mutable-register input. Joins get
//! parameters only for live-in variables with multiple reaching definitions.

mod cfg;
mod parameters;

use super::{Block, Function, Op, Terminator, Value};
use crate::input::{self, Instr, Reg};
use cfg::Cfg;
use parameters::place_parameters;
use std::{collections::HashMap, sync::Arc};

#[cfg(test)]
mod tests;

type Callees = HashMap<*const input::Function, Arc<Function>>;

/// Convert the shared call DAG bottom-up; repeated calls retain one callee.
pub fn from_input(input: &input::Function) -> Result<Function, String> {
    let mut callees = Callees::new();
    let mut pending = vec![(input, false)];
    while let Some((function, ready)) = pending.pop() {
        let key = std::ptr::from_ref(function);
        if callees.contains_key(&key) {
            continue;
        }
        if ready {
            let body = convert(function, &callees)?;
            if std::ptr::eq(function, input) {
                return Ok(body);
            }
            callees.insert(key, Arc::new(body));
        } else {
            pending.push((function, true));
            pending.extend(function.instructions().iter().filter_map(|op| match op {
                Instr::Call { function, .. } => Some((function.as_ref(), false)),
                _ => None,
            }));
        }
    }
    unreachable!("the root is converted last")
}

/// A single current-name table plus undo records for dominator-tree traversal.
struct Names {
    current: Vec<Option<Value>>,
    undo: Vec<(Reg, Option<Value>)>,
}

impl Names {
    fn read(&self, r: Reg) -> Value {
        self.current[r.0 as usize].expect("verified use has a dominating definition")
    }

    fn define(&mut self, r: Reg, v: Value) {
        self.undo.push((r, self.current[r.0 as usize].replace(v)));
    }

    fn restore(&mut self, checkpoint: usize) {
        for (r, old) in self.undo.drain(checkpoint..).rev() {
            self.current[r.0 as usize] = old;
        }
    }
}

fn instruction(
    graph: &mut Function,
    block: Block,
    op: &Instr,
    names: &mut Names,
    callees: &Callees,
) {
    let (dst, new) = match op {
        Instr::Const(dst, n) => (*dst, Op::Const(*n)),
        Instr::Copy(dst, src) => {
            names.define(*dst, names.read(*src));
            return;
        }
        Instr::Bin(dst, op, a, b) => (*dst, Op::Bin(*op, names.read(*a), names.read(*b))),
        Instr::BufferLen { dst, buffer } => (*dst, Op::BufferLen(*buffer)),
        Instr::Load { dst, buffer, index } => (*dst, Op::Load(*buffer, names.read(*index))),
        Instr::Store {
            buffer,
            index,
            value,
        } => {
            graph.append(
                block,
                Op::Store(*buffer, names.read(*index), names.read(*value)),
            );
            return;
        }
        Instr::Call {
            dst,
            function,
            args,
            buffers,
        } => {
            let target = Arc::clone(&callees[&Arc::as_ptr(function)]);
            let args: Vec<_> = args.iter().map(|&r| names.read(r)).collect();
            (*dst, Op::Call(graph.add_call(target, &args, buffers)))
        }
        Instr::Branch { .. } | Instr::Jump(_) | Instr::Return(_) | Instr::Label(_) => return,
    };
    names.define(
        dst,
        graph.append(block, new).expect("operation defines a value"),
    );
}

fn convert(input: &input::Function, callees: &Callees) -> Result<Function, String> {
    let cfg = Cfg::new(input);
    let parameters = place_parameters(input, &cfg);
    let mut graph = Function::new(input.arity(), input.buffer_kinds().to_vec());
    let mut blocks = vec![graph.entry(); cfg.ranges.len()];
    // Split the ABI entry only if a reachable branch can return to the first
    // source instruction. All other functions keep the source block layout.
    let split_entry = cfg.pred[1].len() > 1;
    for (b, block) in blocks.iter_mut().enumerate().skip(1) {
        if (b != 1 || split_entry) && cfg.idom[b] != usize::MAX {
            *block = graph.create_block();
        }
    }
    let mut children = vec![Vec::new(); blocks.len()];
    for &b in &cfg.rpo[1..] {
        children[cfg.idom[b]].push(b);
        for _ in &parameters[b] {
            graph.append_param(blocks[b]);
        }
    }
    let mut names = Names {
        current: vec![None; input.registers],
        undo: Vec::new(),
    };
    for (arg, &value) in graph.params(graph.entry()).iter().enumerate() {
        names.current[arg] = Some(value);
    }
    // An exit event restores the parent's definitions before visiting a
    // sibling. Deep loops and call DAGs do not use the Rust call stack.
    let mut pending = vec![(0, None)];
    while let Some((b, checkpoint)) = pending.pop() {
        if let Some(checkpoint) = checkpoint {
            names.restore(checkpoint);
            continue;
        }
        pending.push((b, Some(names.undo.len())));
        pending.extend(children[b].iter().rev().map(|&child| (child, None)));
        if b == 0 && !split_entry {
            continue;
        }
        let block = blocks[b];
        let body = &input.code[cfg.ranges[b].clone()];
        for (&reg, &value) in parameters[b].iter().zip(graph.params(block)) {
            names.define(reg, value);
        }
        for op in body {
            instruction(&mut graph, block, op, &mut names, callees);
        }
        let mut edge = |next: usize| {
            let args: Vec<_> = parameters[next].iter().map(|&r| names.read(r)).collect();
            graph.add_edge(blocks[next], &args)
        };
        let term = match body.last() {
            Some(Instr::Return(r)) => Terminator::Return(names.read(*r)),
            Some(Instr::Branch { cond, .. }) => Terminator::Branch {
                cond: names.read(*cond),
                yes: edge(cfg.succ[b][0].unwrap()),
                no: edge(cfg.succ[b][1].unwrap()),
            },
            _ => Terminator::Jump(edge(cfg.succ[b][0].expect("verified block has successor"))),
        };
        graph.set_terminator(block, term);
    }
    graph.verify()?;
    Ok(graph)
}
