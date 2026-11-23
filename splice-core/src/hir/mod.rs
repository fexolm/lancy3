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

    /// Scalar semantics shared by code generators. True comparisons produce
    /// all bits set. Trapping arithmetic with invalid operands returns its trap.
    pub fn checked_eval(self, a: i32, b: i32) -> Result<i32, crate::Trap> {
        use crate::Trap::{DivisionByZero, DivisionOverflow, Overflow};
        let divide = |quotient: Option<i32>| match (b, quotient) {
            (0, _) => Err(DivisionByZero),
            (_, None) => Err(DivisionOverflow),
            (_, Some(value)) => Ok(value),
        };
        Ok(match self {
            Self::Add => a.wrapping_add(b),
            Self::Sub => a.wrapping_sub(b),
            Self::Mul => a.wrapping_mul(b),
            Self::Div => divide(a.checked_div(b))?,
            Self::Rem => divide(a.checked_rem(b))?,
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
            Self::CheckedAdd => a.checked_add(b).ok_or(Overflow)?,
            Self::CheckedSub => a.checked_sub(b).ok_or(Overflow)?,
            Self::CheckedMul => a.checked_mul(b).ok_or(Overflow)?,
            Self::CheckedShl => a.checked_shl(b as u32).ok_or(Overflow)?,
            Self::CheckedShr => a.checked_shr(b as u32).ok_or(Overflow)?,
            Self::Min => a.min(b),
            Self::Max => a.max(b),
        })
    }
}

pub use splice_target::abi::{RustPacketSymbol, RustSymbol};

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
            return Err(format!(
                "incompatible arity for Rust import {}",
                self.symbol.name
            ));
        }
        if args.into_iter().any(|ty| ty != Some(ValueType::Scalar)) {
            return Err(format!(
                "expected scalar operands for Rust import {}",
                self.symbol.name
            ));
        }
        Ok(())
    }
}

pub fn check_rust_packet_call(
    symbol: &RustPacketSymbol,
    args: impl ExactSizeIterator<Item = Option<ValueType>>,
) -> crate::Result<()> {
    if symbol.packet_bytes().is_none() || args.len() != symbol.arity as usize {
        return Err(format!(
            "incompatible arity for Rust import {}",
            symbol.name
        ));
    }
    if args.into_iter().any(|ty| ty != Some(ValueType::Scalar)) {
        return Err(format!(
            "expected scalar operands for Rust import {}",
            symbol.name
        ));
    }
    Ok(())
}

/// Reject invalid context metadata in the ABI before generating native code.
pub fn check_context_layout(kinds: &[BufferKind]) -> crate::Result<()> {
    for signature in kinds.iter().filter_map(|kind| kind.context_signature()) {
        signature.validate()?;
    }
    Ok(())
}

/// Validate the descriptor and words of a borrowed Rust context call.
pub fn check_context_call(
    kinds: &[BufferKind],
    context: Buffer,
    operation: u32,
    args: impl ExactSizeIterator<Item = Option<ValueType>>,
    slices: &[ContextSlice],
) -> crate::Result<()> {
    check_rust_context_call(
        kinds,
        RustContextTarget::Borrowed { context, operation },
        args,
        slices,
    )
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
                .ok_or("expected a Rust context descriptor")?;
            signature.validate()?;
            signature
                .shape(operation)
                .ok_or("unknown Rust context operation")?
        }
        RustContextTarget::Static(symbol) => {
            let (words, borrows) = symbol.shape();
            if crate::ContextArguments::packet_layout(words, borrows).is_none() {
                return Err("invalid static Rust operation packet layout".into());
            }
            (words as usize, borrows)
        }
    };
    if args.len() != arity {
        return Err("incompatible Rust context operation arity".into());
    }
    if args.into_iter().any(|ty| ty != Some(ValueType::Scalar)) {
        return Err("expected scalar operands for Rust context operation".into());
    }
    if slices.len() != borrows as usize {
        return Err("incompatible Rust context operation slice arity".into());
    }
    let mut seen = HashMap::new();
    for &ContextSlice { buffer, writable } in slices {
        let kind = kinds
            .get(buffer.index())
            .filter(|kind| kind.context_signature().is_none())
            .ok_or("expected slice operands for Rust context operation")?;
        if writable && !kind.writable() {
            return Err("Rust context operation requires a writable slice".into());
        }
        if slices.len() > 1
            && seen
                .insert(buffer, writable)
                .is_some_and(|prior| prior || writable)
        {
            return Err("writable Rust slice operand aliases another operand".into());
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

    /// Registers the instruction reads, including those a source point saves.
    fn reads(&self) -> impl Iterator<Item = Reg> + '_ {
        let variables = match self {
            Self::Debug(point) => point.variables.as_ref(),
            _ => &[],
        };
        self.operands()
            .chain(variables.iter().filter_map(|var| var.value.register()))
    }

    /// Registers whose values the instruction uses. A source point only
    /// observes its variables, which need not be assigned on every path.
    fn operands(&self) -> impl Iterator<Item = Reg> + '_ {
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
            Self::Call { args, .. }
            | Self::RustCall { args, .. }
            | Self::RustPacketCall { args, .. }
            | Self::RustContextCall { args, .. }
            | Self::Simd { args, .. } => (None, args),
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
    arity: usize,
    buffer_kinds: Vec<BufferKind>,
    needs_status: bool,
    locals: Vec<ValueType>,
    code: Vec<Instr>,
    labels: usize,
}

/// Validate a nested kernel's scalar and buffer arguments in its caller.
pub fn check_call(
    arity: usize,
    kinds: &[BufferKind],
    args: impl ExactSizeIterator<Item = Option<ValueType>>,
    caller_kinds: &[BufferKind],
    buffers: &[Buffer],
) -> crate::Result<()> {
    if args.len() != arity {
        return Err(format!(
            "expected {arity} scalar arguments, got {}",
            args.len()
        ));
    }
    for (i, ty) in args.enumerate() {
        match ty {
            Some(ValueType::Scalar) => {}
            Some(_) => return Err(format!("expected a scalar register for call argument {i}")),
            None => return Err(format!("invalid register for call argument {i}")),
        }
    }
    check_forwarding(caller_kinds, kinds, buffers)
}

impl Function {
    pub fn options(&self) -> crate::BuildOptions {
        self.options
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
