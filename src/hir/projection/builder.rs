use super::{Op, Program, Value};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Construction-time handle. The builder checks ownership before packing operands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuildValue {
    owner: u64,
    value: Value,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BuildError {
    TooManyInputs(usize),
    InputOutOfRange(u16),
    ForeignValue,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyInputs(n) => write!(f, "{n} inputs exceed the 65536-column limit"),
            Self::InputOutOfRange(i) => write!(f, "column {i} is outside the declared inputs"),
            Self::ForeignValue => f.write_str("operand or result belongs to a different builder"),
        }
    }
}

impl std::error::Error for BuildError {}

/// Appends unoptimized instructions in amortized O(1). The first construction
/// error is retained even if the affected instruction is dead at finish.
pub struct Builder {
    owner: u64,
    input_count: usize,
    ops: Vec<Op>,
    error: Option<BuildError>,
}

macro_rules! binary_builder {
    ($method:ident, $variant:ident) => {
        pub fn $method(&mut self, a: BuildValue, b: BuildValue) -> BuildValue {
            let a = self.check(a);
            let b = self.check(b);
            self.push(Op::$variant(a, b))
        }
    };
}

impl Builder {
    pub fn new(input_count: usize) -> Self {
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        #[allow(deprecated, reason = "AtomicU64::try_update is unstable on Rust 1.94")]
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("exhausted builder IDs");
        let error = (input_count > usize::from(u16::MAX) + 1)
            .then_some(BuildError::TooManyInputs(input_count));
        Self {
            owner,
            input_count,
            ops: Vec::new(),
            error,
        }
    }

    fn check(&mut self, handle: BuildValue) -> Value {
        if handle.owner != self.owner {
            self.error.get_or_insert(BuildError::ForeignValue);
        }
        handle.value
    }

    fn push(&mut self, op: Op) -> BuildValue {
        let value = Value::at(self.ops.len());
        self.ops.push(op);
        BuildValue {
            owner: self.owner,
            value,
        }
    }

    pub fn input(&mut self, column: u16) -> BuildValue {
        if usize::from(column) >= self.input_count {
            self.error
                .get_or_insert(BuildError::InputOutOfRange(column));
        }
        self.push(Op::Input(column))
    }

    pub fn constant(&mut self, value: i32) -> BuildValue {
        self.push(Op::Const(value))
    }

    binary_builder!(add, Add);
    binary_builder!(sub, Sub);
    binary_builder!(mul, Mul);
    binary_builder!(and, And);
    binary_builder!(or, Or);
    binary_builder!(xor, Xor);
    binary_builder!(eq, Eq);
    binary_builder!(lt, Lt);

    pub fn select(&mut self, mask: BuildValue, a: BuildValue, b: BuildValue) -> BuildValue {
        let mask = self.check(mask);
        let a = self.check(a);
        let b = self.check(b);
        self.push(Op::Select(mask, a, b))
    }

    pub fn finish(mut self, result: BuildValue) -> Result<Program, BuildError> {
        let result = self.check(result);
        if let Some(err) = self.error {
            return Err(err);
        }
        Ok(Program {
            ops: self.ops,
            result,
            input_count: self.input_count,
        })
    }
}
