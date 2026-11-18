//! Packed, topologically ordered SSA for independent rows of signed 32-bit data.

mod builder;
mod optimize;

pub use builder::{BuildError, BuildValue, Builder};
pub use optimize::optimize;

/// The position of an instruction in its program. Each instruction defines one value.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Value(u32);

impl Value {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    fn at(index: usize) -> Self {
        Self(u32::try_from(index).expect("a program cannot exceed the u32 instruction index space"))
    }
}

/// All operations work lane-wise. Arithmetic wraps, comparisons return -1 or 0,
/// and Select blends individual bits using its first operand as the mask.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Op {
    Input(u16),
    Const(i32),
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    And(Value, Value),
    Or(Value, Value),
    Xor(Value, Value),
    Eq(Value, Value),
    Lt(Value, Value),
    Select(Value, Value, Value),
}

impl Op {
    /// SSA operands in source order, without a heap allocation.
    pub fn operands(self) -> impl Iterator<Item = Value> {
        let zero = Value(0);
        let (values, n) = match self {
            Self::Input(_) | Self::Const(_) => ([zero; 3], 0),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::And(a, b)
            | Self::Or(a, b)
            | Self::Xor(a, b)
            | Self::Eq(a, b)
            | Self::Lt(a, b) => ([a, b, zero], 2),
            Self::Select(mask, a, b) => ([mask, a, b], 3),
        };
        values.into_iter().take(n)
    }

    fn evaluate(self, get: impl Fn(Value) -> i32, input: impl Fn(u16) -> i32) -> i32 {
        match self {
            Self::Input(col) => input(col),
            Self::Const(value) => value,
            Self::Add(a, b) => get(a).wrapping_add(get(b)),
            Self::Sub(a, b) => get(a).wrapping_sub(get(b)),
            Self::Mul(a, b) => get(a).wrapping_mul(get(b)),
            Self::And(a, b) => get(a) & get(b),
            Self::Or(a, b) => get(a) | get(b),
            Self::Xor(a, b) => get(a) ^ get(b),
            Self::Eq(a, b) => -i32::from(get(a) == get(b)),
            Self::Lt(a, b) => -i32::from(get(a) < get(b)),
            Self::Select(mask, a, b) => (get(mask) & get(a)) | (!get(mask) & get(b)),
        }
    }
}

/// A validated expression. Operands always refer to earlier instructions.
#[derive(Clone, Debug)]
pub struct Program {
    ops: Vec<Op>,
    result: Value,
    input_count: usize,
}

impl Program {
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    pub fn result(&self) -> Value {
        self.result
    }

    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Scalar reference interpreter. The caller supplies the declared columns
    /// and a valid row in every column read by this program.
    pub fn eval_row(&self, columns: &[&[i32]], row: usize) -> i32 {
        assert_eq!(
            columns.len(),
            self.input_count,
            "wrong number of input columns"
        );
        let mut values = Vec::with_capacity(self.ops.len());
        for &op in &self.ops {
            let value = op.evaluate(|v| values[v.index()], |col| columns[usize::from(col)][row]);
            values.push(value);
        }
        values[self.result.index()]
    }
}

/// A speculative fact for a row. A specialized program is valid for rows on
/// which all its assumptions hold. Native execution enforces this via guards.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Assumption {
    pub column: u16,
    pub value: i32,
}

#[cfg(test)]
mod tests;
