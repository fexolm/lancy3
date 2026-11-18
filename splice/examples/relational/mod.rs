//! A physical plan with two i32 columns per row. Both children of an inner
//! equijoin can themselves be arbitrary physical plans.
use splice::{BufferArg, Executable};

mod kernels;
mod pricing;
pub use kernels::query;
pub use pricing::pricing_plan;

#[derive(Clone)]
pub enum Expr {
    Key,
    Value,
    Literal(i32),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
}

#[derive(Clone)]
pub enum Predicate {
    AtLeast(Expr, Expr),
    Equal(Expr, Expr),
    And(Box<Predicate>, Box<Predicate>),
}

#[derive(Clone, Copy)]
pub enum Combine {
    Add,
    Multiply,
}

#[derive(Clone)]
pub enum Plan {
    Scan(usize),
    Filter(Box<Plan>, Predicate),
    Project(Box<Plan>, Expr, Expr),
    Join(Box<Plan>, Box<Plan>, Combine),
}

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
            for &(key, value) in *table {
                data.extend([key, value]);
            }
            offsets.push(words);
        }
        Ok(Self { offsets, data })
    }

    /// On success returns the packed result in the caller's workspace.
    pub fn run<'a>(
        &self,
        code: &(impl Executable + ?Sized),
        workspace: &'a mut [i32],
    ) -> Result<&'a [i32], String> {
        let rows = code.call(
            &[],
            &mut [
                BufferArg::Read(self.offsets()),
                BufferArg::Read(self.data()),
                BufferArg::Write(workspace),
            ],
        )?;
        if rows < 0 {
            return Err("query workspace is too small".into());
        }
        workspace
            .get(..rows as usize * 2)
            .ok_or_else(|| "invalid query result length".into())
    }
}
