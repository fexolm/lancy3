//! Portable kernels over scalar, vector and mask registers, with explicit control flow and calls.
//! A function owns a dense instruction stream. Parameters occupy the first
//! registers. Mutable locals and temporary results share the same index space.

mod buffer;
mod builder;
pub mod memory;
mod replay;
mod rust_call;
mod simd;
mod types;
mod verify;
pub use simd::{Access, Elements, Memory, OpInfo, Simd, SimdOp, Slot, ValueType};
pub use types::{Constant, ScalarType};

pub use buffer::{
    Buffer, BufferArg, BufferElement, BufferKind, check_forwarding, validate_buffers,
};
pub use builder::FunctionBuilder;
pub use replay::replay;
pub use rust_call::{PacketStore, PacketValue, RustCallPlan, RustTarget};

use crate::intrinsic::IntrinsicId;
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reg(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Label(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextSlice {
    pub buffer: Buffer,
    pub writable: bool,
}

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
    CheckedAdd,
    CheckedSub,
    CheckedMul,
    CheckedShl,
    CheckedShr,
    /// Integer `saturating_add`, `saturating_sub` and `saturating_mul`.
    SaturatingAdd,
    SaturatingSub,
    SaturatingMul,
    Min,
    Max,
}

impl Binary {
    /// Every operation, in declaration order.
    pub const ALL: [Self; 26] = [
        Self::Add,
        Self::Sub,
        Self::Mul,
        Self::Div,
        Self::Rem,
        Self::And,
        Self::Or,
        Self::Xor,
        Self::Eq,
        Self::Ne,
        Self::Lt,
        Self::Le,
        Self::Gt,
        Self::Ge,
        Self::Shl,
        Self::Shr,
        Self::CheckedAdd,
        Self::CheckedSub,
        Self::CheckedMul,
        Self::CheckedShl,
        Self::CheckedShr,
        Self::SaturatingAdd,
        Self::SaturatingSub,
        Self::SaturatingMul,
        Self::Min,
        Self::Max,
    ];

    pub fn may_trap(self) -> bool {
        matches!(
            self,
            Self::Div
                | Self::Rem
                | Self::CheckedAdd
                | Self::CheckedSub
                | Self::CheckedMul
                | Self::CheckedShl
                | Self::CheckedShr
        )
    }
}

#[derive(Clone, Debug)]
pub enum Instr {
    Debug(crate::debug::Point),
    Simd {
        dst: Option<Reg>,
        inst: Simd,
        args: Vec<Reg>,
    },
    Const(Reg, Constant),
    /// Rust's `src as T`, where `T` is the type of `dst`.
    Cast {
        dst: Reg,
        src: Reg,
    },
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
    TrapBounds,
    /// Run Rust code on scalar `args` and `slices`; `dst` is its `result`.
    RustCall {
        dst: Reg,
        target: RustTarget,
        args: Vec<Reg>,
        slices: Vec<ContextSlice>,
        result: ScalarType,
    },
    Call {
        dst: Reg,
        function: Arc<Function>,
        args: Vec<Reg>,
        buffers: Vec<Buffer>,
    },
    /// Call the intrinsic `id` on `args`; `dst` has its result type. The
    /// backend that compiles the function implements it or fails the build.
    Intrinsic {
        dst: Reg,
        id: IntrinsicId,
        args: Vec<Reg>,
    },
}

impl Instr {
    fn may_trap(&self) -> bool {
        match self {
            Self::Bin(_, op, ..) => op.may_trap(),
            Self::Load { .. } | Self::Store { .. } | Self::RustCall { .. } | Self::TrapBounds => {
                true
            }
            Self::Call { function, .. } => function.needs_status(),
            Self::Simd { inst, .. } => inst.op.may_trap(),
            _ => false,
        }
    }

    fn written(&self) -> Option<Reg> {
        match self {
            Self::Simd { dst, .. } => *dst,
            Self::Const(dst, _)
            | Self::Cast { dst, .. }
            | Self::Copy(dst, _)
            | Self::Bin(dst, ..)
            | Self::BufferLen { dst, .. }
            | Self::Load { dst, .. }
            | Self::Call { dst, .. }
            | Self::Intrinsic { dst, .. }
            | Self::RustCall { dst, .. } => Some(*dst),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    source: Option<&'static crate::frontend::KernelSource>,
    options: crate::BuildOptions,
    params: Vec<ScalarType>,
    result: ScalarType,
    buffer_kinds: Vec<BufferKind>,
    needs_status: bool,
    locals: Vec<ValueType>,
    code: Vec<Instr>,
    labels: usize,
}

/// Validate a nested kernel's scalar and buffer arguments in its caller.
pub fn check_call(
    callee: crate::frontend::Signature<'_>,
    args: impl ExactSizeIterator<Item = Option<ValueType>>,
    caller_kinds: &[BufferKind],
    buffers: &[Buffer],
) -> crate::Result<()> {
    if args.len() != callee.params.len() {
        return Err(crate::BuildError::verify(format!(
            "expected {} scalar arguments, got {}",
            callee.params.len(),
            args.len()
        )));
    }
    for (i, (ty, &param)) in args.zip(callee.params).enumerate() {
        match ty {
            Some(ValueType::Scalar(ty)) if ty == param => {}
            Some(ty) => {
                return Err(crate::BuildError::verify(format!(
                    "expected a register of type {param} for call argument {i}, got {ty:?}"
                )));
            }
            None => {
                return Err(crate::BuildError::verify(format!(
                    "invalid register for call argument {i}"
                )));
            }
        }
    }
    check_forwarding(caller_kinds, callee.buffers, buffers)
}

impl Function {
    pub fn options(&self) -> crate::BuildOptions {
        self.options
    }

    pub fn source(&self) -> Option<&'static crate::frontend::KernelSource> {
        self.source
    }

    pub fn register_count(&self) -> usize {
        self.params.len() + self.locals.len()
    }

    pub fn reg_type(&self, r: Reg) -> ValueType {
        self.checked_reg_type(r).expect("invalid register")
    }

    fn checked_reg_type(&self, r: Reg) -> Option<ValueType> {
        let index = r.0 as usize;
        match self.params.get(index) {
            Some(&ty) => Some(ValueType::Scalar(ty)),
            None => self.locals.get(index - self.params.len()).copied(),
        }
    }

    pub fn label_count(&self) -> usize {
        self.labels
    }

    pub fn arity(&self) -> usize {
        self.params.len()
    }
    pub fn params(&self) -> &[ScalarType] {
        &self.params
    }
    pub fn result(&self) -> ScalarType {
        self.result
    }
    pub fn buffer_kinds(&self) -> &[BufferKind] {
        &self.buffer_kinds
    }

    /// True when this function or one of its callees requires a native status
    /// channel for memory operations or fallible integer arithmetic.
    pub fn needs_status(&self) -> bool {
        self.needs_status
    }

    /// This function and the bodies it calls, directly or not, in first-use
    /// breadth-first order. A body shared by several calls appears once.
    pub fn reachable_functions(&self) -> Vec<&Self> {
        let mut functions = vec![self];
        let mut seen = HashSet::from([std::ptr::from_ref(self)]);
        let mut i = 0;
        while i < functions.len() {
            for inst in functions[i].instructions() {
                if let Instr::Call { function, .. } = inst
                    && seen.insert(Arc::as_ptr(function))
                {
                    functions.push(function);
                }
            }
            i += 1;
        }
        functions
    }

    pub fn instructions(&self) -> &[Instr] {
        &self.code
    }
}
