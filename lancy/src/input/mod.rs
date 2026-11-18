//! Scalar functions over i32 values, with explicit control flow and calls.
//! A function owns a dense instruction stream. Parameters occupy the first
//! registers. Mutable locals and temporary results share the same index space.

mod buffer;
#[cfg(test)]
mod buffer_tests;
mod builder;
mod interpret;
mod optimize;
mod verify;

pub use buffer::{Buffer, BufferArg, BufferKind};
pub(crate) use buffer::{check_forwarding, validate_buffers};
pub use builder::FunctionBuilder;

use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reg(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Label(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Shl,
    Shr,
    Min,
    Max,
}

impl Binary {
    pub(crate) fn checked_eval(self, a: i32, b: i32) -> Option<i32> {
        Some(match self {
            Self::Add => a.wrapping_add(b),
            Self::Sub => a.wrapping_sub(b),
            Self::Mul => a.wrapping_mul(b),
            Self::Div => a.checked_div(b)?,
            Self::Rem => a.checked_rem(b)?,
            Self::And => a & b,
            Self::Or => a | b,
            Self::Xor => a ^ b,
            Self::Eq => -i32::from(a == b),
            Self::Ne => -i32::from(a != b),
            Self::Lt => -i32::from(a < b),
            Self::Le => -i32::from(a <= b),
            Self::Gt => -i32::from(a > b),
            Self::Ge => -i32::from(a >= b),
            Self::Shl => a.wrapping_shl(b as u32),
            Self::Shr => a.wrapping_shr(b as u32),
            Self::Min => a.min(b),
            Self::Max => a.max(b),
        })
    }
}

#[derive(Clone, Debug)]
pub enum Instr {
    Const(Reg, i32),
    Copy(Reg, Reg),
    Bin(Reg, Binary, Reg, Reg),
    BufferLen {
        dst: Reg,
        buffer: Buffer,
    },
    Load {
        dst: Reg,
        buffer: Buffer,
        index: Reg,
    },
    Store {
        buffer: Buffer,
        index: Reg,
        value: Reg,
    },
    Label(Label),
    Jump(Label),
    Branch {
        cond: Reg,
        yes: Label,
        no: Label,
    },
    Return(Reg),
    Call {
        dst: Reg,
        function: Arc<Function>,
        args: Vec<Reg>,
        buffers: Vec<Buffer>,
    },
}

impl Instr {
    fn may_trap(&self) -> bool {
        match self {
            Self::Bin(_, Binary::Div | Binary::Rem, ..)
            | Self::Load { .. }
            | Self::Store { .. } => true,
            Self::Call { function, .. } => function.needs_status(),
            _ => false,
        }
    }

    pub(crate) fn read(&self, mut read: impl FnMut(Reg)) {
        match self {
            Self::Copy(_, r) | Self::Return(r) | Self::Branch { cond: r, .. } => read(*r),
            Self::Bin(_, _, a, b) => {
                read(*a);
                read(*b);
            }
            Self::Load { index, .. } => read(*index),
            Self::Store { index, value, .. } => {
                read(*index);
                read(*value);
            }
            Self::Call { args, .. } => args.iter().copied().for_each(read),
            _ => {}
        }
    }

    pub(crate) fn written(&self) -> Option<Reg> {
        match self {
            Self::Const(dst, _)
            | Self::Copy(dst, _)
            | Self::Bin(dst, ..)
            | Self::BufferLen { dst, .. }
            | Self::Load { dst, .. }
            | Self::Call { dst, .. } => Some(*dst),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    pub(crate) arity: usize,
    buffer_kinds: Vec<BufferKind>,
    needs_status: bool,
    pub(crate) registers: usize,
    pub(crate) code: Vec<Instr>,
    pub(crate) targets: Vec<usize>,
}

impl Function {
    /// Import an indexed instruction stream. Applies the same control-flow,
    /// initialization and borrow checks as the incremental builder.
    pub fn from_instructions(
        arity: usize,
        buffers: Vec<BufferKind>,
        registers: usize,
        labels: usize,
        instructions: Vec<Instr>,
    ) -> Result<Self, String> {
        verify::finish(arity, buffers, registers, labels, instructions)
    }

    pub(crate) fn rebuild(&self, code: Vec<Instr>) -> Result<Self, String> {
        verify::finish(
            self.arity,
            self.buffer_kinds.clone(),
            self.registers,
            self.targets.len(),
            code,
        )
    }

    pub fn arity(&self) -> usize {
        self.arity
    }
    pub fn buffer_kinds(&self) -> &[BufferKind] {
        &self.buffer_kinds
    }

    /// True when this function or one of its callees requires a native status
    /// channel for memory operations or fallible integer arithmetic.
    pub fn needs_status(&self) -> bool {
        self.needs_status
    }

    fn refresh_status(&mut self) {
        self.needs_status = !self.buffer_kinds.is_empty() || self.code.iter().any(Instr::may_trap);
    }
    pub fn instructions(&self) -> &[Instr] {
        &self.code
    }
}
