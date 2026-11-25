//! A physical plan with two i32 columns per row. Both children of an inner
//! equijoin can themselves be arbitrary physical plans.
use splice::{BufferArg, CallError, Executable, KernelScalar};

mod kernels;
mod pipeline;
mod pricing;
pub use kernels::query;
pub use pricing::pricing_plan;

#[derive(Clone, splice::ComptimeKey)]
pub enum Expr {
    Key,
    Value,
    Literal(i32),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
}

#[derive(Clone, splice::ComptimeKey)]
pub enum Predicate {
    AtLeast(Expr, Expr),
    Equal(Expr, Expr),
    And(Box<Predicate>, Box<Predicate>),
}

#[derive(Clone, Copy, splice::ComptimeKey)]
pub enum Combine {
    Add,
    Multiply,
}

#[derive(Clone, splice::ComptimeKey)]
pub enum Plan {
    Scan(usize),
    Filter(Box<Plan>, Predicate),
    Project(Box<Plan>, Expr, Expr),
    Join(Box<Plan>, Box<Plan>, Combine),
}

/// Each table stores a contiguous key column followed by its value column.
/// Offsets are measured in i32 words. The same compiled plan can run against
/// catalogs with different table sizes. Only the table IDs are staged.
pub struct Catalog {
    offsets: Vec<i32>,
    data: Vec<i32>,
}

impl Catalog {
    pub fn offsets(&self) -> &[i32] {
        &self.offsets
    }

    pub fn data(&self) -> &[i32] {
        &self.data
    }

    pub fn new(tables: &[&[(i32, i32)]]) -> Result<Self, String> {
        let mut offsets = vec![0];
        let mut data = Vec::new();
        for table in tables {
            let words = table
                .len()
                .checked_mul(2)
                .and_then(|n| data.len().checked_add(n))
                .and_then(|n| i32::try_from(n).ok())
                .ok_or("catalog exceeds i32 address range")?;
            data.extend(table.iter().map(|&(key, _)| key));
            data.extend(table.iter().map(|&(_, value)| value));
            offsets.push(words);
        }
        Ok(Self { offsets, data })
    }

    /// Returns the packed result in the caller's workspace. Joins may reorder rows.
    pub fn run<'a>(
        &self,
        code: &(impl Executable + ?Sized),
        workspace: &'a mut [i32],
        state: &mut [i32],
    ) -> Result<&'a [i32], RunError> {
        let rows = code.call(
            &[],
            &mut [
                BufferArg::read(self.offsets()),
                BufferArg::read(self.data()),
                BufferArg::write(workspace),
                BufferArg::write(state),
            ],
        )?;
        let rows = i32::from_word(rows);
        if rows < 0 {
            return Err(RunError::WorkspaceTooSmall);
        }
        workspace
            .get(..rows as usize * 2)
            .ok_or(RunError::ResultLength)
    }
}

/// Why a query produced no rows.
#[derive(Debug, PartialEq)]
pub enum RunError {
    Call(CallError),
    WorkspaceTooSmall,
    /// The kernel reported more rows than the workspace holds.
    ResultLength,
}

impl From<CallError> for RunError {
    fn from(error: CallError) -> Self {
        Self::Call(error)
    }
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Call(error) => error.fmt(f),
            Self::WorkspaceTooSmall => f.write_str("query workspace is too small"),
            Self::ResultLength => f.write_str("invalid query result length"),
        }
    }
}

impl std::error::Error for RunError {}
