//! One description of every operation: which values it reads, the type of
//! its result, its effects and its memory access. Builders, the verifier and
//! the passes ask these functions instead of matching `Op` themselves, so a
//! new operation is described here once.
use super::{Binary, Lane, Op, RustContextImport, Simd, Ternary, Type, Value};
use crate::hir::{Function, Inst, IrError, integer};

/// Why an operation may stop the program instead of producing its result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrapKind {
    /// Integer division or remainder by zero, or a signed `MIN / -1`.
    Division,
    /// A lane index at or above the vector's lane count.
    LaneIndex,
    /// `Op::BoundsCheck`.
    Bounds,
    /// `Op::TrapIf` or `Op::Trap`.
    Explicit,
    /// A memory access. The program keeps its address valid where it runs, so
    /// moving or adding the access can fault.
    Memory,
    /// A call, which can report any trap of its callee.
    Call,
}

/// What executing an operation does besides producing its result.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Effects {
    pub reads: bool,
    pub writes: bool,
    pub trap: Option<TrapKind>,
    /// Visible to a debugger: a debug store or source point.
    pub debug: bool,
}

impl Effects {
    const NONE: Self = Self {
        reads: false,
        writes: false,
        trap: None,
        debug: false,
    };
    const fn trap(kind: TrapKind) -> Self {
        Self {
            trap: Some(kind),
            ..Self::NONE
        }
    }
    const fn memory(writes: bool) -> Self {
        Self {
            reads: !writes,
            writes,
            trap: Some(TrapKind::Memory),
            debug: false,
        }
    }
    const CALL: Self = Self {
        reads: true,
        writes: true,
        trap: Some(TrapKind::Call),
        debug: false,
    };
    const DEBUG: Self = Self {
        debug: true,
        ..Self::NONE
    };

    /// No effect at all: the operation can be removed when unused, moved or
    /// evaluated speculatively.
    #[inline]
    pub fn is_none(self) -> bool {
        self == Self::NONE
    }
}

/// The memory an access touches: `ty` at `address`, or the lanes of `ty` at
/// `address + indices[k]` elements when indexed, limited to the active lanes
/// of `mask` when masked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Access {
    pub address: Value,
    pub ty: Type,
    pub indices: Option<Value>,
    pub mask: Option<Value>,
    /// The stored value of a write, or the passthrough of a masked read.
    pub value: Option<Value>,
    pub write: bool,
}

impl Op {
    /// A scalar literal in its one canonical form: `Const` for `i32`, the low
    /// bits of the lane otherwise.
    pub fn literal(lane: Lane, bits: u128) -> Self {
        if lane == Lane::I32 {
            Self::Const(bits as i32)
        } else {
            Self::Literal(lane, bits & integer::mask(lane))
        }
    }

    /// Whether the operation defines a value. Fixed by the operation alone,
    /// independently of operand types.
    #[inline]
    pub fn has_result(self) -> bool {
        !matches!(
            self,
            Self::Store(..)
                | Self::Simd(Simd::MaskedStore { .. } | Simd::Store(..))
                | Self::BoundsCheck(..)
                | Self::TrapIf(..)
                | Self::Trap(_)
                | Self::DebugStore(..)
                | Self::DebugPoint(_)
        )
    }

    /// Effects of a verified operation in `f`.
    #[inline]
    pub(crate) fn effects(self, f: &Function) -> Effects {
        match self {
            Self::Bin(Binary::Div | Binary::Rem, a, _)
            | Self::Simd(Simd::Bin(Binary::Div | Binary::Rem, a, _)) => {
                if f.value_type(a).lane().is_some_and(Lane::float) {
                    Effects::NONE
                } else {
                    Effects::trap(TrapKind::Division)
                }
            }
            Self::Simd(Simd::ExtractLane(..) | Simd::InsertLane(..)) => {
                Effects::trap(TrapKind::LaneIndex)
            }
            Self::BoundsCheck(..) => Effects::trap(TrapKind::Bounds),
            Self::TrapIf(..) | Self::Trap(_) => Effects::trap(TrapKind::Explicit),
            Self::Load(..) | Self::Simd(Simd::Load(..) | Simd::MaskedLoad { .. }) => {
                Effects::memory(false)
            }
            Self::Store(..) | Self::Simd(Simd::Store(..) | Simd::MaskedStore { .. }) => {
                Effects::memory(true)
            }
            Self::Call(_)
            | Self::RustCall(..)
            | Self::RustPacketCall(_)
            | Self::RustContextCall(..) => Effects::CALL,
            Self::DebugStore(..) | Self::DebugPoint(_) => Effects::DEBUG,
            Self::Const(_)
            | Self::Literal(..)
            | Self::Simd(_)
            | Self::Unary(..)
            | Self::Ternary(..)
            | Self::Cast(..)
            | Self::Bitcast(..)
            | Self::Copy(_)
            | Self::Bin(..)
            | Self::PtrOffset(..)
            | Self::CallScratch { .. } => Effects::NONE,
        }
    }

    /// The memory a verified load or store in `f` accesses.
    #[inline]
    pub(crate) fn access(self, f: &Function) -> Option<Access> {
        let plain = |address, ty, value, write| Access {
            address,
            ty,
            indices: None,
            mask: None,
            value,
            write,
        };
        Some(match self {
            Self::Load(ty, address) => plain(address, ty, None, false),
            Self::Simd(Simd::Load(lane, address)) => {
                plain(address, Type::Vector(lane), None, false)
            }
            Self::Store(address, value) | Self::Simd(Simd::Store(address, value)) => {
                plain(address, f.value_type(value), Some(value), true)
            }
            Self::Simd(
                Simd::MaskedLoad {
                    address,
                    mask,
                    passthrough: value,
                }
                | Simd::MaskedStore {
                    address,
                    mask,
                    value,
                },
            ) => Access {
                address: address.base,
                ty: f.value_type(value),
                indices: address.indices,
                mask: Some(mask),
                value: Some(value),
                write: matches!(self, Self::Simd(Simd::MaskedStore { .. })),
            },
            _ => return None,
        })
    }

    /// Check the operands of `inst` and return the type of its result.
    /// `operand` gives the type of each value the operation reads, so the
    /// verifier can check dominance on the same visit.
    pub(crate) fn result_type(
        self,
        f: &Function,
        inst: Inst,
        operand: impl Fn(Value) -> Result<Type, IrError>,
    ) -> Result<Option<Type>, IrError> {
        let expect = |value, expected| {
            let actual = operand(value)?;
            if actual == expected {
                Ok(())
            } else {
                Err(IrError::OperandType {
                    inst,
                    value,
                    expected,
                    actual,
                })
            }
        };
        let scalar = |value| {
            let ty = operand(value)?;
            if matches!(ty, Type::Vector(_)) {
                Err(IrError::UseSimdForm { inst, ty })
            } else {
                Ok(ty)
            }
        };
        let ty = match self {
            Self::Const(_) => Type::I32,
            Self::Literal(lane, bits) => {
                if lane == Lane::I32 || bits & !integer::mask(lane) != 0 {
                    return Err(IrError::NonCanonicalLiteral { inst, lane, bits });
                }
                lane.scalar()
            }
            Self::Simd(op) => return simd_type(op, inst, operand),
            Self::Copy(v) => operand(v)?,
            Self::Bin(op, a, b) => {
                let ty = scalar(a)?;
                expect(b, ty)?;
                if !op.supports(ty) {
                    return Err(IrError::InvalidNumeric { inst, ty });
                }
                op.result_type(ty)
            }
            Self::Unary(op, v) => {
                let ty = scalar(v)?;
                if !op.supports(ty) {
                    return Err(IrError::InvalidNumeric { inst, ty });
                }
                op.result_type(ty)
            }
            Self::Ternary(op, a, b, c) => {
                let ty = scalar(b)?;
                expect(a, select_condition(op, ty))?;
                expect(c, ty)?;
                if ty
                    .lane()
                    .is_none_or(|l| op == Ternary::MulAdd && !l.float())
                {
                    return Err(IrError::InvalidNumeric { inst, ty });
                }
                ty
            }
            Self::Cast(to, v) | Self::Bitcast(to, v) => {
                let ty = operand(v)?;
                let bitcast = matches!(self, Self::Bitcast(..));
                match (ty, to) {
                    (Type::Ptr, Type::U64) if bitcast => {}
                    (Type::Ptr, _) | (_, Type::Ptr) => {
                        return Err(IrError::PointerCast { inst, from: ty, to });
                    }
                    (Type::Vector(_), _) | (_, Type::Vector(_)) => {
                        return Err(IrError::UseSimdForm { inst, ty });
                    }
                    _ if bitcast && ty.lane().map(Lane::bytes) != to.lane().map(Lane::bytes) => {
                        return Err(IrError::InvalidNumeric { inst, ty });
                    }
                    _ => {}
                }
                to
            }
            Self::BoundsCheck(index, len) => {
                expect(index, Type::I32)?;
                expect(len, Type::I32)?;
                return Ok(None);
            }
            Self::TrapIf(cond, trap) => {
                expect(cond, Type::I32)?;
                check_trap(inst, trap)?;
                return Ok(None);
            }
            Self::Trap(trap) => {
                check_trap(inst, trap)?;
                return Ok(None);
            }
            Self::RustCall(_, a, b) => {
                expect(a, Type::I32)?;
                expect(b, Type::I32)?;
                Type::I32
            }
            Self::RustPacketCall(site) => {
                let (import, scratch, words) = f
                    .packet_site(site)
                    .ok_or(IrError::InvalidRustPacket { inst })?;
                let Some(bytes) = import.packet_bytes() else {
                    return Err(IrError::InvalidRustPacket { inst });
                };
                if words.len() != import.arity as usize || !valid_scratch(scratch, bytes) {
                    return Err(IrError::InvalidRustPacket { inst });
                }
                for &word in words {
                    expect(word, Type::I32)?;
                }
                Type::I32
            }
            Self::RustContextCall(import, [context, operation, a, b]) => {
                expect(context, Type::Ptr)?;
                expect(operation, Type::I32)?;
                expect(
                    a,
                    match import {
                        RustContextImport::Scalars(_) => Type::I32,
                        RustContextImport::Packet(_) => Type::Ptr,
                    },
                )?;
                expect(b, Type::I32)?;
                Type::I32
            }
            Self::CallScratch { offset, bytes } => {
                if !valid_scratch(offset, bytes) {
                    return Err(IrError::InvalidScratch { inst });
                }
                Type::Ptr
            }
            Self::PtrOffset(base, index, _) => {
                expect(base, Type::Ptr)?;
                expect(index, Type::I32)?;
                Type::Ptr
            }
            Self::Load(ty, address) => {
                if matches!(ty, Type::Vector(_)) {
                    return Err(IrError::UseSimdForm { inst, ty });
                }
                expect(address, Type::Ptr)?;
                ty
            }
            Self::Store(address, value) => {
                expect(address, Type::Ptr)?;
                scalar(value)?;
                return Ok(None);
            }
            Self::DebugStore(_, value) => {
                operand(value)?;
                return Ok(None);
            }
            Self::DebugPoint(_) => return Ok(None),
            Self::Call(call) => {
                let (callee, args) = f
                    .call_site(call)
                    .ok_or(IrError::UnknownCall { inst, call })?;
                if args.len() != callee.arity() {
                    return Err(IrError::CallArity {
                        call,
                        expected: callee.arity(),
                        actual: args.len(),
                    });
                }
                for (&arg, &ty) in args.iter().zip(callee.param_types()) {
                    expect(arg, ty)?;
                }
                callee.return_type()
            }
        };
        Ok(Some(ty))
    }
}

fn select_condition(op: Ternary, ty: Type) -> Type {
    if op == Ternary::Select {
        ty.mask_type()
    } else {
        ty
    }
}

/// Scratch regions start 8-byte aligned and end within the 32-bit frame.
fn valid_scratch(offset: u32, bytes: u32) -> bool {
    offset.is_multiple_of(8) && offset.checked_add(bytes).is_some()
}

fn simd_type(
    op: Simd,
    inst: Inst,
    operand: impl Fn(Value) -> Result<Type, IrError>,
) -> Result<Option<Type>, IrError> {
    use Simd::*;
    let invalid = |ty| IrError::InvalidNumeric { inst, ty };
    let vector = |v| {
        let ty = operand(v)?;
        if let Type::Vector(l) = ty {
            Ok(l)
        } else {
            Err(invalid(ty))
        }
    };
    let expect = |value, expected| {
        let actual = operand(value)?;
        if actual == expected {
            Ok(())
        } else {
            Err(IrError::OperandType {
                inst,
                value,
                expected,
                actual,
            })
        }
    };
    Ok(Some(match op {
        LaneCount(_) => Type::I32,
        Iota(l) => Type::Vector(l),
        Load(l, p) => {
            expect(p, Type::Ptr)?;
            Type::Vector(l)
        }
        Store(p, x) => {
            expect(p, Type::Ptr)?;
            vector(x)?;
            return Ok(None);
        }
        Splat(x) => {
            let ty = operand(x)?;
            match ty.lane() {
                Some(l) if l.scalar() == ty => Type::Vector(l),
                _ => return Err(invalid(ty)),
            }
        }
        Unary(op, x) => {
            let ty = Type::Vector(vector(x)?);
            if !op.supports(ty) {
                return Err(invalid(ty));
            }
            op.result_type(ty)
        }
        Bin(op, x, y) => {
            let ty = Type::Vector(vector(x)?);
            expect(y, ty)?;
            if !op.supports(ty) {
                return Err(invalid(ty));
            }
            op.result_type(ty)
        }
        Ternary(op, x, y, z) => {
            let l = vector(y)?;
            let ty = Type::Vector(l);
            expect(z, ty)?;
            expect(x, select_condition(op, ty))?;
            if op == super::Ternary::MulAdd && !l.float() {
                return Err(invalid(ty));
            }
            ty
        }
        Cast(to, x) | Widen { to, value: x, .. } | Narrow(to, x, _) => {
            let l = vector(x)?;
            let size = match op {
                Widen { .. } => l.bytes() * 2,
                Narrow(_, _, y) => {
                    expect(y, Type::Vector(l))?;
                    l.bytes() / 2
                }
                _ => l.bytes(),
            };
            if to.bytes() != size {
                return Err(invalid(Type::Vector(l)));
            }
            Type::Vector(to)
        }
        Bitcast(to, x) => {
            vector(x)?;
            Type::Vector(to)
        }
        ExtractLane(x, i) => {
            let l = vector(x)?;
            expect(i, Type::I32)?;
            l.scalar()
        }
        InsertLane(x, i, v) => {
            let l = vector(x)?;
            expect(i, Type::I32)?;
            expect(v, l.scalar())?;
            Type::Vector(l)
        }
        Shuffle(x, y, i) => {
            let l = vector(x)?;
            expect(y, Type::Vector(l))?;
            let idx = vector(i)?;
            if idx.float() || idx.bytes() != l.bytes() {
                return Err(invalid(Type::Vector(idx)));
            }
            Type::Vector(l)
        }
        Reduce(op, x) => {
            let l = vector(x)?;
            if !matches!(
                op,
                Binary::Add
                    | Binary::Mul
                    | Binary::And
                    | Binary::Or
                    | Binary::Xor
                    | Binary::Min
                    | Binary::Max
            ) {
                return Err(invalid(Type::Vector(l)));
            }
            l.scalar()
        }
        AnyTrue(x) | AllTrue(x) => {
            let l = vector(x)?;
            if l.float() {
                return Err(invalid(Type::Vector(l)));
            }
            Type::I32
        }
        Compress(x, mask) | Expand(x, mask) => {
            let ty = Type::Vector(vector(x)?);
            expect(mask, ty.mask_type())?;
            ty
        }
        MaskedLoad {
            address,
            mask,
            passthrough: x,
        }
        | MaskedStore {
            address,
            mask,
            value: x,
        } => {
            expect(address.base, Type::Ptr)?;
            let l = vector(x)?;
            expect(mask, Type::Vector(l).mask_type())?;
            if let Some(i) = address.indices {
                let idx = vector(i)?;
                if !idx.signed() || idx.bytes() != l.bytes() {
                    return Err(invalid(Type::Vector(idx)));
                }
            }
            if matches!(op, MaskedStore { .. }) {
                return Ok(None);
            }
            Type::Vector(l)
        }
    }))
}

/// A kernel stops itself only for a runtime error of its own: Rust panics
/// and ABI defects come from a Rust entry.
fn check_trap(inst: Inst, trap: crate::Trap) -> Result<(), IrError> {
    use crate::Trap as T;
    match trap {
        T::Bounds | T::DivisionByZero | T::DivisionOverflow | T::Overflow => Ok(()),
        _ => Err(IrError::InvalidTrap { inst, trap }),
    }
}
