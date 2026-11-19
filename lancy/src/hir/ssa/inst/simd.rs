use super::{Binary, Lane, Ternary, Type, Unary, Value};
use crate::hir::Inst;
use crate::hir::IrError;

/// Contiguous lanes or signed element indices relative to a base pointer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VectorAddress {
    pub base: Value,
    pub indices: Option<Value>,
}

/// Operations on scalable vectors. Each element type has a target-dependent lane count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Simd {
    LaneCount(Lane),
    /// Lane numbers starting at zero, in increasing address order.
    Iota(Lane),
    /// Concatenate two vectors and select lanes by unsigned indices. Out-of-range lanes are zero.
    Shuffle(Value, Value, Value),
    /// Fold lanes in increasing order. The verifier accepts arithmetic and bitwise reductions.
    Reduce(Binary, Value),
    AnyTrue(Value),
    AllTrue(Value),
    /// Pack active lanes to the low end, or expand low lanes into active positions. Unused lanes become zero.
    Compress(Value, Value),
    Expand(Value, Value),
    /// Convert the lower or upper half into elements twice as wide.
    Widen {
        to: Lane,
        high: bool,
        value: Value,
    },
    /// Convert and concatenate two vectors of elements twice as wide as the result.
    Narrow(Lane, Value, Value),
    Unary(Unary, Value),
    Ternary(Ternary, Value, Value, Value),
    Splat(Value),
    ExtractLane(Value, Value),
    InsertLane(Value, Value, Value),
    Bin(Binary, Value, Value),
    Cast(Lane, Value),
    Bitcast(Lane, Value),
    Load(Lane, Value),
    Store(Value, Value),
    /// Read only lanes where the mask's high bit is set. Other lanes keep passthrough.
    MaskedLoad {
        address: VectorAddress,
        mask: Value,
        passthrough: Value,
    },
    /// Write active lanes in increasing lane order, including repeated scatter indices.
    MaskedStore {
        address: VectorAddress,
        mask: Value,
        value: Value,
    },
}

impl Simd {
    pub(in crate::hir::ssa) fn result_type(self, ty: impl Fn(Value) -> Type) -> Option<Type> {
        use Simd::*;
        Some(match self {
            LaneCount(_) | AnyTrue(_) | AllTrue(_) => Type::I32,
            Iota(l)
            | Load(l, _)
            | Cast(l, _)
            | Bitcast(l, _)
            | Widen { to: l, .. }
            | Narrow(l, ..) => Type::Vector(l),
            Splat(x) => Type::Vector(ty(x).lane().unwrap_or(super::Lane::I32)),
            ExtractLane(x, _) | Reduce(_, x) => ty(x).lane().map_or(Type::I32, super::Lane::scalar),
            Unary(op, x) => op.result_type(ty(x)),
            Bin(op, x, _) => op.result_type(ty(x)),
            Ternary(_, _, x, _)
            | InsertLane(x, ..)
            | Shuffle(x, ..)
            | Compress(x, _)
            | Expand(x, _)
            | MaskedLoad { passthrough: x, .. } => ty(x),
            Store(..) | MaskedStore { .. } => return None,
        })
    }

    pub(in crate::hir::ssa) fn verify(
        self,
        inst: Inst,
        operand: impl Fn(Value) -> Result<Type, IrError>,
    ) -> Result<(), IrError> {
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
        match self {
            Iota(_) | LaneCount(_) => {}
            Load(_, p) => expect(p, Type::Ptr)?,
            Store(p, x) => {
                expect(p, Type::Ptr)?;
                vector(x)?;
            }
            Splat(x) => {
                let ty = operand(x)?;
                if ty.lane().is_none_or(|l| l.scalar() != ty) {
                    return Err(invalid(ty));
                }
            }
            Unary(op, x) => {
                let ty = Type::Vector(vector(x)?);
                if !op.supports(ty) {
                    return Err(invalid(ty));
                }
            }
            Bin(op, x, y) => {
                let ty = Type::Vector(vector(x)?);
                expect(y, ty)?;
                if !op.supports(ty) {
                    return Err(invalid(ty));
                }
            }
            Ternary(op, x, y, z) => {
                let l = vector(y)?;
                let ty = Type::Vector(l);
                expect(z, ty)?;
                expect(
                    x,
                    if op == super::Ternary::Select {
                        ty.mask_type()
                    } else {
                        ty
                    },
                )?;
                if op == super::Ternary::MulAdd && !l.float() {
                    return Err(invalid(ty));
                }
            }
            Cast(to, x) | Widen { to, value: x, .. } | Narrow(to, x, _) => {
                let l = vector(x)?;
                let size = match self {
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
            }
            Bitcast(_, x) => {
                vector(x)?;
            }
            ExtractLane(x, i) | InsertLane(x, i, _) => {
                let l = vector(x)?;
                expect(i, Type::I32)?;
                if let InsertLane(_, _, v) = self {
                    expect(v, l.scalar())?;
                }
            }
            Shuffle(x, y, i) => {
                let l = vector(x)?;
                expect(y, Type::Vector(l))?;
                let idx = vector(i)?;
                if idx.float() || idx.bytes() != l.bytes() {
                    return Err(invalid(Type::Vector(idx)));
                }
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
            }
            AnyTrue(x) | AllTrue(x) => {
                let l = vector(x)?;
                if l.float() {
                    return Err(invalid(Type::Vector(l)));
                }
            }
            Compress(x, mask) | Expand(x, mask) => {
                expect(mask, Type::Vector(vector(x)?).mask_type())?
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
            }
        }
        Ok(())
    }

    pub(super) fn map_operands(&mut self, mut map: impl FnMut(Value) -> Value) {
        match self {
            Self::MaskedLoad {
                address,
                mask,
                passthrough: value,
            }
            | Self::MaskedStore {
                address,
                mask,
                value,
            } => {
                address.base = map(address.base);
                address.indices = address.indices.map(&mut map);
                *mask = map(*mask);
                *value = map(*value);
            }
            Self::Reduce(_, v)
            | Self::AnyTrue(v)
            | Self::AllTrue(v)
            | Self::Widen { value: v, .. }
            | Self::Load(_, v)
            | Self::Unary(_, v)
            | Self::Splat(v)
            | Self::Cast(_, v)
            | Self::Bitcast(_, v) => *v = map(*v),
            Self::Shuffle(a, b, c) | Self::InsertLane(a, b, c) | Self::Ternary(_, a, b, c) => {
                *a = map(*a);
                *b = map(*b);
                *c = map(*c);
            }
            Self::Compress(a, b)
            | Self::Expand(a, b)
            | Self::Narrow(_, a, b)
            | Self::ExtractLane(a, b)
            | Self::Bin(_, a, b)
            | Self::Store(a, b) => {
                *a = map(*a);
                *b = map(*b);
            }
            Self::Iota(_) | Self::LaneCount(_) => {}
        }
    }
}
