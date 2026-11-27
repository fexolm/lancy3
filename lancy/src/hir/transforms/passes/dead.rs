//! Trace live SSA values from observable effects and control flow.
use super::facts::Facts;
use crate::hir::{Function, Inst, Value, transforms::ControlFlowGraph};
use crate::support::slotmap::DenseMap;

fn mark(value: Value, live: &mut DenseMap<Value, bool>, pending: &mut Vec<Value>) {
    if !std::mem::replace(&mut live[value], true) {
        pending.push(value);
    }
}

pub(super) fn eliminate(f: &mut Function, cfg: &ControlFlowGraph, facts: &Facts) {
    let mut live = DenseMap::filled(f.value_count(), false);
    let mut pending = Vec::new();
    let mut scanned = DenseMap::filled(f.block_capacity(), false);
    let mut instructions = Vec::new();
    for &block in cfg.reverse_postorder().iter().rev() {
        if let Some(value) = f.terminator(block).operand() {
            live[value] = true;
        }
        instructions.extend(f.instructions(block));
        for inst in instructions.drain(..).rev() {
            if let Some(result) = f.result(inst) {
                if !live[result] && facts.is_total(f, inst) {
                    continue;
                }
                live[result] = true;
            }
            // Ordinary operands are defined earlier in the block or in
            // a dominator, so they have yet to be scanned.
            f.visit_operands(inst, |v| live[v] = true);
        }
        scanned[block] = true;
        let params = f.params(block);
        if params.is_empty() {
            continue;
        }
        for (_, edge) in cfg.incoming_edges(block) {
            for (&param, &arg) in params.iter().zip(f.edge(edge).1) {
                if live[param]
                    && !std::mem::replace(&mut live[arg], true)
                    && scanned[facts.defs[arg].block]
                {
                    pending.push(arg);
                }
            }
        }
    }
    // Edge arguments can make earlier-scanned definitions live, including
    // chains of block parameters carried around a cycle.
    while let Some(value) = pending.pop() {
        let mut mark = |v| mark(v, &mut live, &mut pending);
        let def = facts.defs[value];
        if def.inst != Inst::INVALID {
            f.visit_operands(def.inst, mark);
        } else if def.block != f.entry() {
            let column = f.param_index(def.block, value).unwrap();
            for (_, edge) in cfg.incoming_edges(def.block) {
                mark(f.edge(edge).1[column]);
            }
        }
    }

    if cfg.reverse_postorder()[1..]
        .iter()
        .any(|&b| f.params(b).iter().any(|&p| !live[p]))
    {
        f.retain_block_params(|p| live[p]);
    }
    for &block in cfg.reverse_postorder() {
        f.rewrite_instructions(block, |op, result, _| {
            result.is_none_or(|v| live[v]).then_some(op)
        });
    }
}
