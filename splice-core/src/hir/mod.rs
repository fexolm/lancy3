//! Portable kernels over scalar, vector and mask registers, with explicit control flow and calls.
//! A function owns a dense instruction stream. Parameters occupy the first
//! registers. Mutable locals and temporary results share the same index space.

mod buffer;
mod intrinsic;
pub use intrinsic::{Intrinsic, MathOp};
mod builder;
pub mod memory;
mod replay;
mod rust_call;
mod simd;
mod types;
mod verify;
pub use simd::{Access, Elements, Memory, OpInfo, Simd, SimdOp, Slot, ValueType};
pub use types::{Constant, ScalarInfo, ScalarKind, ScalarType};

pub use buffer::{BufferArg, BufferElement, BufferKind, Types, check_forwarding, validate_buffers};
pub use builder::FunctionBuilder;
pub use replay::replay;
pub use rust_call::{PacketStore, PacketValue, RustCallPlan, RustTarget};

use std::collections::HashSet;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reg(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Label(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextSlice {
    pub slice: Slice,
    pub writable: bool,
}

/// A slice value: a pair of registers, the address of its first element
/// (of type [`ValueType::Ptr`], which names the element) and its length
/// (an `Index`). Buffer parameters are the ABI source of such pairs; any
/// pair of registers of those types is a slice, so the frontend keeps a
/// pointer with the length it was checked against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slice {
    pub ptr: Reg,
    pub len: Reg,
}

impl Slice {
    /// The registers of buffer parameter `buffer` of a function with
    /// `arity` scalar parameters: two per buffer, after the scalars.
    ///
    /// # Panics
    /// If the registers are not representable.
    pub const fn param(arity: usize, buffer: usize) -> Self {
        let ptr = arity + 2 * buffer;
        assert!(ptr < u32::MAX as usize - 1, "too many parameters");
        Self {
            ptr: Reg(ptr as u32),
            len: Reg(ptr as u32 + 1),
        }
    }
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
    /// Apply a typed intrinsic to its runtime operands.
    CallIntrinsic {
        dst: Option<Reg>,
        intrinsic: Intrinsic,
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
    /// `ptr` advanced by `offset` elements, unchecked: the frontend checks
    /// the offset against the length of the slice `ptr` addresses (see
    /// `subslice`). `dst` has the type of `ptr`.
    Offset {
        dst: Reg,
        ptr: Reg,
        offset: Reg,
    },
    /// Read element `index` of `slice`, trapping unless it is below the length.
    Load {
        dst: Reg,
        slice: Slice,
        index: Reg,
    },
    /// Write element `index` of `slice`, trapping unless it is below the length.
    Store {
        slice: Slice,
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
        buffers: Vec<Slice>,
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
            Self::CallIntrinsic {
                intrinsic: Intrinsic::Simd(inst),
                ..
            } => inst.op.may_trap(),
            _ => false,
        }
    }

    fn written(&self) -> Option<Reg> {
        match self {
            Self::CallIntrinsic { dst, .. } => *dst,
            Self::Const(dst, _)
            | Self::Cast { dst, .. }
            | Self::Copy(dst, _)
            | Self::Bin(dst, ..)
            | Self::Offset { dst, .. }
            | Self::Load { dst, .. }
            | Self::Call { dst, .. }
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
    caller_types: Types<'_>,
    buffers: &[Slice],
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
    check_forwarding(caller_types, callee.buffers, buffers)
}

impl Function {
    pub fn options(&self) -> crate::BuildOptions {
        self.options
    }

    pub fn source(&self) -> Option<&'static crate::frontend::KernelSource> {
        self.source
    }

    pub fn register_count(&self) -> usize {
        self.signature().registers() + self.locals.len()
    }

    /// The registers of buffer parameter `buffer`.
    pub fn param_slice(&self, buffer: usize) -> Slice {
        Slice::param(self.params.len(), buffer)
    }

    pub fn reg_type(&self, r: Reg) -> ValueType {
        self.checked_reg_type(r).expect("invalid register")
    }

    fn checked_reg_type(&self, r: Reg) -> Option<ValueType> {
        let index = r.0 as usize;
        let signature = self.signature();
        match signature.param_type(r) {
            Some(ty) => Some(ty),
            None => self.locals.get(index - signature.registers()).copied(),
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
