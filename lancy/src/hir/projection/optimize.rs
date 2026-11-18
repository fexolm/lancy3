//! Constant folding, value numbering and dead-code elimination for projection SSA.

use super::{Assumption, Op, Program, Value};
use std::collections::HashMap;

impl Op {
    fn remap(self, ids: &[Value]) -> Self {
        let get = |v: Value| ids[v.index()];
        match self {
            Self::Input(_) | Self::Const(_) => self,
            Self::Add(a, b) => Self::Add(get(a), get(b)),
            Self::Sub(a, b) => Self::Sub(get(a), get(b)),
            Self::Mul(a, b) => Self::Mul(get(a), get(b)),
            Self::And(a, b) => Self::And(get(a), get(b)),
            Self::Or(a, b) => Self::Or(get(a), get(b)),
            Self::Xor(a, b) => Self::Xor(get(a), get(b)),
            Self::Eq(a, b) => Self::Eq(get(a), get(b)),
            Self::Lt(a, b) => Self::Lt(get(a), get(b)),
            Self::Select(mask, a, b) => Self::Select(get(mask), get(a), get(b)),
        }
    }

    // Order the operands of commutative operations for value numbering.
    fn canonical(self) -> Self {
        match self {
            Self::Add(a, b) if b < a => Self::Add(b, a),
            Self::Mul(a, b) if b < a => Self::Mul(b, a),
            Self::And(a, b) if b < a => Self::And(b, a),
            Self::Or(a, b) if b < a => Self::Or(b, a),
            Self::Xor(a, b) if b < a => Self::Xor(b, a),
            Self::Eq(a, b) if b < a => Self::Eq(b, a),
            _ => self,
        }
    }
}

enum Simplified {
    Emit(Op),
    Reuse(Value),
}

fn simplify(op: Op, ops: &[Op]) -> Simplified {
    use Op::*;
    use Simplified::{Emit, Reuse};

    let constant = |v: Value| match ops[v.index()] {
        Const(n) => Some(n),
        _ => None,
    };
    // Xor(x, -1) inverts every mask bit. Swapping blend operands removes
    // the inversion even for arbitrary user-provided bit masks.
    let op = if let Select(m, yes, no) = op {
        match ops[m.index()] {
            Xor(x, y) if constant(y) == Some(-1) => Select(x, no, yes),
            Xor(x, y) if constant(x) == Some(-1) => Select(y, no, yes),
            _ => op,
        }
    } else {
        op
    };
    if matches!(op, Input(_) | Const(_)) {
        return Emit(op);
    }
    if op.operands().all(|v| constant(v).is_some()) {
        let value = op.evaluate(|v| constant(v).unwrap(), |_| unreachable!());
        return Emit(Const(value));
    }
    if let Xor(a, b) = op {
        let cancel = |nested: Value, outer| match ops[nested.index()] {
            Xor(x, y) if x == outer => Some(y),
            Xor(x, y) if y == outer => Some(x),
            _ => None,
        };
        if let Some(value) = cancel(a, b).or_else(|| cancel(b, a)) {
            return Reuse(value);
        }
    }
    match op {
        Add(a, b) | Or(a, b) | Xor(a, b) if constant(a) == Some(0) => Reuse(b),
        Add(a, b) | Sub(a, b) | Or(a, b) | Xor(a, b) if constant(b) == Some(0) => Reuse(a),
        Sub(a, b) | Xor(a, b) | Lt(a, b) if a == b => Emit(Const(0)),
        Mul(a, b) | And(a, b) if constant(a) == Some(0) || constant(b) == Some(0) => Emit(Const(0)),
        Mul(a, b) if constant(a) == Some(1) => Reuse(b),
        Mul(a, b) if constant(b) == Some(1) => Reuse(a),
        And(a, b) if constant(a) == Some(-1) => Reuse(b),
        And(a, b) if constant(b) == Some(-1) => Reuse(a),
        Or(a, b) if constant(a) == Some(-1) || constant(b) == Some(-1) => Emit(Const(-1)),
        And(a, b) | Or(a, b) if a == b => Reuse(a),
        Eq(a, b) if a == b => Emit(Const(-1)),
        Select(m, a, _) if constant(m) == Some(-1) => Reuse(a),
        Select(m, _, b) if constant(m) == Some(0) => Reuse(b),
        Select(_, a, b) if a == b => Reuse(a),
        Select(m, a, b) if constant(a) == Some(-1) && constant(b) == Some(0) => Reuse(m),
        Select(m, a, b) if constant(b) == Some(0) => Emit(And(m, a).canonical()),
        Select(m, a, b) if constant(a) == Some(-1) => Emit(Or(m, b).canonical()),
        _ => Emit(op.canonical()),
    }
}

/// Specializes input loads, folds constants, and removes common and dead
/// expressions in linear passes (expected O(n) for hash-based value numbering).
/// The caller must establish all supplied assumptions before using the result.
pub fn optimize(program: &Program, assumptions: &[Assumption]) -> Program {
    let facts: HashMap<_, _> = assumptions.iter().map(|a| (a.column, a.value)).collect();
    let mut ops = Vec::with_capacity(program.ops.len());
    let mut remap = Vec::with_capacity(program.ops.len());
    let mut table = HashMap::with_capacity(program.ops.len());
    for &op in &program.ops {
        let op = match op {
            Op::Input(col) => facts.get(&col).copied().map_or(op, Op::Const),
            _ => op.remap(&remap),
        };
        let value = match simplify(op, &ops) {
            Simplified::Reuse(v) => v,
            Simplified::Emit(op) => *table.entry(op).or_insert_with(|| {
                let v = Value::at(ops.len());
                ops.push(op);
                v
            }),
        };
        remap.push(value);
    }

    let result = remap[program.result.index()];
    let mut live = vec![false; ops.len()];
    live[result.index()] = true;
    for (i, op) in ops.iter().enumerate().rev() {
        if live[i] {
            for v in op.operands() {
                live[v.index()] = true;
            }
        }
    }
    let mut next = 0;
    for i in 0..ops.len() {
        if live[i] {
            remap[i] = Value::at(next);
            ops[next] = ops[i].remap(&remap);
            next += 1;
        }
    }
    ops.truncate(next);
    Program {
        ops,
        result: remap[result.index()],
        input_count: program.input_count,
    }
}
