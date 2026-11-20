//! Portable kernels over scalar, vector and mask registers, with explicit control flow and calls.
//! A function owns a dense instruction stream. Parameters occupy the first
//! registers. Mutable locals and temporary results share the same index space.

mod buffer;
mod builder;
mod simd;
mod verify;
pub use simd::{Simd, SimdOp, ValueType};

pub use buffer::{
    Buffer, BufferArg, BufferElement, BufferKind, check_forwarding, validate_buffers,
};
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
    /// Scalar semantics shared by code generators. True comparisons produce
    /// all bits set. Invalid signed division and remainder return `None`.
    pub fn checked_eval(self, a: i32, b: i32) -> Option<i32> {
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
    Simd {
        dst: Option<Reg>,
        inst: Simd,
        args: Vec<Reg>,
    },
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

    fn reads(&self) -> impl Iterator<Item = Reg> + '_ {
        use std::slice::from_ref;
        let (first, rest): (_, &[Reg]) = match self {
            Self::Copy(_, r)
            | Self::Return(r)
            | Self::Branch { cond: r, .. }
            | Self::Load { index: r, .. } => (None, from_ref(r)),
            Self::Bin(_, _, a, b)
            | Self::Store {
                index: a, value: b, ..
            } => (Some(*a), from_ref(b)),
            Self::Call { args, .. } | Self::Simd { args, .. } => (None, args),
            _ => (None, &[]),
        };
        first.into_iter().chain(rest.iter().copied())
    }

    fn written(&self) -> Option<Reg> {
        match self {
            Self::Simd { dst, .. } => *dst,
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
    source: Option<&'static crate::frontend::KernelSource>,
    arity: usize,
    buffer_kinds: Vec<BufferKind>,
    needs_status: bool,
    locals: Vec<ValueType>,
    code: Vec<Instr>,
    labels: usize,
}

impl Function {
    fn check_call(
        &self,
        caller: &[BufferKind],
        args: &[Reg],
        buffers: &[Buffer],
    ) -> crate::Result<()> {
        if args.len() != self.arity {
            return Err(format!(
                "expected {} scalar arguments, got {}",
                self.arity,
                args.len()
            ));
        }
        check_forwarding(caller, &self.buffer_kinds, buffers)
    }

    pub fn source(&self) -> Option<&'static crate::frontend::KernelSource> {
        self.source
    }

    pub fn register_count(&self) -> usize {
        self.arity + self.locals.len()
    }

    pub fn reg_type(&self, r: Reg) -> ValueType {
        self.checked_reg_type(r).expect("invalid register")
    }

    fn checked_reg_type(&self, r: Reg) -> Option<ValueType> {
        let index = r.0 as usize;
        if index < self.arity {
            Some(ValueType::Scalar)
        } else {
            self.locals.get(index - self.arity).copied()
        }
    }

    pub fn label_count(&self) -> usize {
        self.labels
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

    pub fn instructions(&self) -> &[Instr] {
        &self.code
    }
}
