//! Lower SSA operations and assign vector registers in a linear scan over
//! their last uses. XMM0..5 are volatile on both supported x64 ABIs.

use crate::hir::{Op, Program, Value};
use crate::lir::batch::{Action, Batch, Binary, Operand, Step};

struct Homes {
    regs: Vec<u8>,
    spills: Vec<u32>,
    slots: u32,
}

impl Homes {
    fn take(&mut self) -> Result<Operand, String> {
        if let Some(reg) = self.regs.pop() {
            return Ok(Operand::VectorReg(reg));
        }
        if let Some(slot) = self.spills.pop() {
            return Ok(Operand::Spill(slot));
        }
        if self.slots > i32::MAX as u32 / 16 {
            return Err("x64 spill displacement exceeds 2 GiB".into());
        }
        let slot = self.slots;
        self.slots += 1;
        Ok(Operand::Spill(slot))
    }

    fn release(&mut self, home: Operand) {
        match home {
            Operand::VectorReg(reg) => self.regs.push(reg),
            Operand::Spill(slot) => self.spills.push(slot),
            Operand::Immediate(_) => {}
        }
    }
}

pub(super) fn lower(program: &Program, vectorize: bool) -> Result<Batch, String> {
    let ops = program.ops();
    let mut last_use: Vec<usize> = (0..ops.len()).collect();
    let mut scratch_regs = 1;
    for (index, op) in ops.iter().enumerate() {
        match op {
            Op::Select(..) => scratch_regs = scratch_regs.max(2),
            Op::Mul(..) if vectorize => scratch_regs = 3,
            _ => {}
        }
        for arg in op.operands() {
            last_use[arg.index()] = index;
        }
    }
    last_use[program.result().index()] = ops.len();
    let mut homes = Homes {
        regs: (scratch_regs..6).rev().collect(),
        spills: Vec::new(),
        slots: 0,
    };
    let mut values = Vec::with_capacity(ops.len());
    let mut steps = Vec::with_capacity(ops.len());
    for (index, &op) in ops.iter().enumerate() {
        let binary =
            |kind, a: Value, b: Value| Action::Binary(kind, values[a.index()], values[b.index()]);
        let action = match op {
            Op::Const(n) => {
                values.push(Operand::Immediate(n));
                continue;
            }
            Op::Input(col) => Action::Load(col),
            Op::Add(a, b) => binary(Binary::Add, a, b),
            Op::Sub(a, b) => binary(Binary::Sub, a, b),
            Op::Mul(a, b) => binary(Binary::Mul, a, b),
            Op::And(a, b) => binary(Binary::And, a, b),
            Op::Or(a, b) => binary(Binary::Or, a, b),
            Op::Xor(a, b) => binary(Binary::Xor, a, b),
            Op::Eq(a, b) => binary(Binary::Eq, a, b),
            Op::Lt(a, b) => binary(Binary::Gt, b, a),
            Op::Select(m, a, b) => {
                Action::Select(values[m.index()], values[a.index()], values[b.index()])
            }
        };
        for arg in op.operands() {
            if last_use[arg.index()] == index {
                homes.release(values[arg.index()]);
                // An instruction can consume the same SSA value twice.
                last_use[arg.index()] = usize::MAX;
            }
        }
        if last_use[index] == index {
            // No consumer. A placeholder keeps `values` indexed by SSA id.
            values.push(Operand::Immediate(0));
            continue;
        }
        let dest = homes.take()?;
        steps.push(Step { dest, action });
        values.push(dest);
    }
    Ok(Batch {
        steps,
        result: values[program.result().index()],
        spill_slots: homes.slots as usize,
    })
}
