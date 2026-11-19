use super::{Buffer, BufferKind};
use crate::simd::Element;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    Scalar,
    Vector(Element),
    Mask(Element),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdOp {
    Lanes,
    FirstN,
    Splat(u64),
    SplatScalar,
    Iota,
    Load(Buffer),
    Store(Buffer),
    LoadSelect(Buffer),
    StoreSelect(Buffer),
    Gather(Buffer),
    Scatter(Buffer),
    GatherSelect(Buffer),
    ScatterSelect(Buffer),
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Xor,
    Not,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    MaskAnd,
    MaskOr,
    MaskXor,
    MaskNot,
    Any,
    All,
    Select,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Simd {
    pub element: Element,
    pub op: SimdOp,
}

impl Simd {
    pub fn output(self) -> Option<ValueType> {
        use SimdOp::*;
        Some(match self.op {
            Lanes | Any | All => ValueType::Scalar,
            FirstN | Eq | Ne | Lt | Le | Gt | Ge | MaskAnd | MaskOr | MaskXor | MaskNot => {
                ValueType::Mask(self.element)
            }
            Store(_) | StoreSelect(_) | Scatter(_) | ScatterSelect(_) => return None,
            _ => ValueType::Vector(self.element),
        })
    }
    pub fn validate(self, args: &[ValueType], buffers: &[BufferKind]) -> crate::Result<()> {
        use SimdOp::*;
        let v = ValueType::Vector(self.element);
        let m = ValueType::Mask(self.element);
        let i = ValueType::Vector(self.element.index());
        let s = ValueType::Scalar;
        let expected: &[ValueType] = match self.op {
            Lanes | Splat(_) | Iota => &[],
            SplatScalar if self.element == Element::I32 => &[s],
            SplatScalar => return Err("runtime splat expects an i32 element".into()),
            Div if self.element.is_float() => &[v, v],
            Div => return Err("SIMD division requires floating point elements".into()),
            FirstN | Load(_) => &[s],
            Store(_) => &[s, v],
            LoadSelect(_) | StoreSelect(_) => &[s, m, v],
            Gather(_) => &[i],
            Scatter(_) => &[i, v],
            GatherSelect(_) | ScatterSelect(_) => &[i, m, v],
            Add | Sub | Mul | Eq | Ne | Lt | Le | Gt | Ge => &[v, v],
            And | Or | Xor | Not if self.element.is_float() => {
                return Err("bitwise SIMD operations require integer elements".into());
            }
            And | Or | Xor => &[v, v],
            Not => &[v],
            MaskAnd | MaskOr | MaskXor => &[m, m],
            MaskNot | Any | All => &[m],
            Select => &[m, v, v],
        };
        let compatible = |(actual, want): (&ValueType, &ValueType)| match (actual, want) {
            (ValueType::Mask(a), ValueType::Mask(b)) => a.index() == b.index(),
            _ => actual == want,
        };
        if args.len() != expected.len() || !args.iter().zip(expected).all(compatible) {
            return Err(format!(
                "invalid SIMD operands for {:?}: expected {expected:?}, got {args:?}",
                self.op
            ));
        }
        let (buffer, write) = match self.op {
            Load(b) | LoadSelect(b) | Gather(b) | GatherSelect(b) => (b, false),
            Store(b) | StoreSelect(b) | Scatter(b) | ScatterSelect(b) => (b, true),
            _ => return Ok(()),
        };
        let kind = buffers.get(buffer.index()).ok_or("invalid SIMD buffer")?;
        if kind.element() != self.element {
            return Err("SIMD buffer element type mismatch".into());
        }
        if write && !kind.writable() {
            return Err("SIMD store to read-only buffer".into());
        }
        Ok(())
    }
}
