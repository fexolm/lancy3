//! Portable kernels over scalar, vector and mask registers, with explicit control flow and calls.
//! A function owns a dense instruction stream. Parameters occupy the first
//! registers. Mutable locals and temporary results share the same index space.

mod buffer;
mod builder;
mod replay;
mod simd;
mod types;
mod verify;
pub use simd::{Simd, SimdOp, ValueType};
pub use types::{Constant, ScalarType};

pub use buffer::{
    Buffer, BufferArg, BufferElement, BufferKind, check_forwarding, validate_buffers,
};
pub use builder::FunctionBuilder;
pub use replay::replay;

use std::collections::HashMap;
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

#[derive(Clone, Copy, Debug)]
pub enum RustContextTarget {
    Borrowed { context: Buffer, operation: u32 },
    Static(&'static crate::StaticRustSymbol),
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
    Min,
    Max,
}

impl Binary {
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

pub use splice_target::abi::{RustPacketSymbol, RustSymbol};

/// The operands of a Rust call are `i32` ABI words and it returns one.
const WORD: Option<ValueType> = Some(ValueType::Scalar(ScalarType::I32));

/// A scalar symbol and the number of its ABI words in use.
#[derive(Clone, Copy, Debug)]
pub struct RustCall {
    pub symbol: &'static RustSymbol,
    pub arity: u8,
}
impl RustCall {
    pub fn validate(
        &self,
        args: impl ExactSizeIterator<Item = Option<ValueType>>,
    ) -> crate::Result<()> {
        if self.arity > 2 || args.len() != usize::from(self.arity) {
            return Err(crate::BuildError::verify(format!(
                "incompatible arity for Rust import {}",
                self.symbol.name
            )));
        }
        if args.into_iter().any(|ty| ty != WORD) {
            return Err(crate::BuildError::verify(format!(
                "expected i32 operands for Rust import {}",
                self.symbol.name
            )));
        }
        Ok(())
    }
}

pub fn check_rust_packet_call(
    symbol: &RustPacketSymbol,
    args: impl ExactSizeIterator<Item = Option<ValueType>>,
) -> crate::Result<()> {
    if symbol.packet_bytes().is_none() || args.len() != symbol.arity as usize {
        return Err(crate::BuildError::verify(format!(
            "incompatible arity for Rust import {}",
            symbol.name
        )));
    }
    let args: Vec<_> = args.collect();
    if args.iter().any(|&ty| ty != WORD) {
        return Err(crate::BuildError::verify(format!(
            "expected i32 operands for Rust import {}, got {args:?}",
            symbol.name
        )));
    }
    Ok(())
}

#[doc(hidden)]
pub fn check_rust_context_call(
    kinds: &[BufferKind],
    target: RustContextTarget,
    args: impl ExactSizeIterator<Item = Option<ValueType>>,
    slices: &[ContextSlice],
) -> crate::Result<()> {
    let (arity, borrows) = match target {
        RustContextTarget::Borrowed { context, operation } => {
            let signature = kinds
                .get(context.index())
                .and_then(|kind| kind.context_signature())
                .ok_or_else(|| crate::BuildError::verify("expected a Rust context descriptor"))?;
            signature
                .shape(operation)
                .ok_or_else(|| crate::BuildError::verify("unknown Rust context operation"))?
        }
        RustContextTarget::Static(symbol) => {
            let (words, borrows) = symbol.shape();
            if crate::ContextArguments::packet_layout(words, borrows).is_none() {
                return Err(crate::BuildError::verify(
                    "invalid static Rust operation packet layout",
                ));
            }
            (words as usize, borrows)
        }
    };
    if args.len() != arity {
        return Err(crate::BuildError::verify(
            "incompatible Rust context operation arity",
        ));
    }
    if args.into_iter().any(|ty| ty != WORD) {
        return Err(crate::BuildError::verify(
            "expected i32 operands for Rust context operation",
        ));
    }
    if slices.len() != borrows as usize {
        return Err(crate::BuildError::verify(
            "incompatible Rust context operation slice arity",
        ));
    }
    let mut seen = HashMap::new();
    for &ContextSlice { buffer, writable } in slices {
        let kind = kinds
            .get(buffer.index())
            .filter(|kind| kind.context_signature().is_none())
            .ok_or_else(|| {
                crate::BuildError::verify("expected slice operands for Rust context operation")
            })?;
        if writable && !kind.writable() {
            return Err(crate::BuildError::verify(
                "Rust context operation requires a writable slice",
            ));
        }
        if slices.len() > 1
            && seen
                .insert(buffer, writable)
                .is_some_and(|prior| prior || writable)
        {
            return Err(crate::BuildError::verify(
                "writable Rust slice operand aliases another operand",
            ));
        }
    }
    Ok(())
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
    RustCall {
        dst: Reg,
        import: RustCall,
        args: Vec<Reg>,
    },
    RustPacketCall {
        dst: Reg,
        import: &'static RustPacketSymbol,
        args: Vec<Reg>,
    },
    RustContextCall {
        dst: Reg,
        target: RustContextTarget,
        args: Vec<Reg>,
        slices: Vec<ContextSlice>,
    },
    Call {
        dst: Reg,
        function: Arc<Function>,
        args: Vec<Reg>,
        buffers: Vec<Buffer>,
    },
}

impl Instr {
    pub fn terminates(&self) -> bool {
        matches!(
            self,
            Self::Return(_) | Self::TrapBounds | Self::Jump(_) | Self::Branch { .. }
        )
    }

    fn may_trap(&self) -> bool {
        match self {
            Self::Bin(_, op, ..) => op.may_trap(),
            Self::Load { .. }
            | Self::Store { .. }
            | Self::RustCall { .. }
            | Self::RustPacketCall { .. }
            | Self::RustContextCall { .. }
            | Self::TrapBounds => true,
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
            | Self::RustCall { dst, .. }
            | Self::RustPacketCall { dst, .. }
            | Self::RustContextCall { dst, .. } => Some(*dst),
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

    /// Number reachable bodies in first-use, breadth-first order, with the root at zero.
    /// Multiple calls to the same shared body receive one number.
    pub fn reachable_functions(&self) -> (Vec<&Self>, HashMap<*const Self, usize>) {
        let mut functions = vec![self];
        let mut ids = HashMap::from([(std::ptr::from_ref(self), 0)]);
        let mut i = 0;
        while i < functions.len() {
            for inst in functions[i].instructions() {
                if let Instr::Call { function, .. } = inst {
                    ids.entry(Arc::as_ptr(function)).or_insert_with(|| {
                        let id = functions.len();
                        functions.push(function);
                        id
                    });
                }
            }
            i += 1;
        }
        (functions, ids)
    }

    pub fn instructions(&self) -> &[Instr] {
        &self.code
    }
}
