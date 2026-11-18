//! Evaluate pure calls with known arguments under a bounded instruction budget.

use super::super::{Function, Instr};
use std::collections::HashMap;

const STEPS_PER_CALL: usize = 1024;
const STEPS_PER_COMPILATION: usize = 16 * STEPS_PER_CALL;
const CONST_CALL_DEPTH: usize = 32;

pub(super) struct ConstEvaluator {
    remaining: usize,
    results: HashMap<(*const Function, Vec<i32>), Option<i32>>,
}

impl ConstEvaluator {
    pub(super) fn new() -> Self {
        Self {
            remaining: STEPS_PER_COMPILATION,
            results: HashMap::new(),
        }
    }

    pub(super) fn evaluate(&mut self, f: &Function, args: Vec<i32>) -> Option<i32> {
        if !f.buffer_kinds().is_empty() {
            return None;
        }
        let key = (std::ptr::from_ref(f), args);
        if let Some(result) = self.results.get(&key) {
            return *result;
        }
        if self.remaining == 0 {
            return None;
        }
        let budget = self.remaining.min(STEPS_PER_CALL);
        let mut fuel = budget;
        let result = Self::run(f, &key.1, &mut fuel, 0);
        self.remaining -= budget - fuel;
        // A cached decision must agree between dataflow and emission, even
        // when some other call subsequently exhausts the compilation budget.
        self.results.insert(key, result);
        result
    }

    fn run(f: &Function, args: &[i32], fuel: &mut usize, depth: usize) -> Option<i32> {
        if depth >= CONST_CALL_DEPTH {
            return None;
        }
        // Charge for allocating and initializing the virtual register file.
        *fuel = fuel.checked_sub(1 + f.registers.div_ceil(8))?;
        let mut regs = vec![0; f.registers];
        regs[..args.len()].copy_from_slice(args);
        let mut pc = 0;
        loop {
            *fuel = fuel.checked_sub(1)?;
            let op = &f.code[pc];
            pc += 1;
            match op {
                Instr::Const(d, n) => regs[d.0 as usize] = *n,
                Instr::Copy(d, s) => regs[d.0 as usize] = regs[s.0 as usize],
                Instr::Bin(d, op, a, b) => {
                    regs[d.0 as usize] = op.checked_eval(regs[a.0 as usize], regs[b.0 as usize])?;
                }
                Instr::Label(_) => {}
                Instr::Jump(l) => pc = f.targets[l.0 as usize],
                Instr::Branch { cond, yes, no } => {
                    let to = if regs[cond.0 as usize] != 0 { yes } else { no };
                    pc = f.targets[to.0 as usize];
                }
                Instr::Return(r) => return Some(regs[r.0 as usize]),
                Instr::Load { .. } | Instr::Store { .. } | Instr::BufferLen { .. } => return None,
                Instr::Call {
                    dst,
                    function,
                    args,
                    ..
                } => {
                    *fuel = fuel.checked_sub(args.len().div_ceil(8))?;
                    let actual: Vec<_> = args.iter().map(|r| regs[r.0 as usize]).collect();
                    regs[dst.0 as usize] = Self::run(function, &actual, fuel, depth + 1)?;
                }
            }
        }
    }
}
