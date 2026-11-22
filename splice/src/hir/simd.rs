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
    Shuffle(crate::simd::Shuffle),
    PrefixSum,
    AlignmentPrefix(Buffer),
    Compress,
    Expand,
    Swizzle,
    Count,
    Shr(u32),
    RotateLeft(u32),
    CompressStore(Buffer),
    Last,
    ExtractWrapping,
    ReplaceWrapping,
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
    Sqrt,
    MulAdd,
    Cast(Element),
    Widen { to: Element, high: bool },
    Narrow(Element),
    ReduceSum,
    Min,
    Max,
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
            Lanes | AlignmentPrefix(_) | Any | All | Last | ExtractWrapping | Count | ReduceSum => {
                ValueType::Scalar
            }
            FirstN | Eq | Ne | Lt | Le | Gt | Ge | MaskAnd | MaskOr | MaskXor | MaskNot => {
                ValueType::Mask(self.element)
            }
            CompressStore(_) | Store(_) | StoreSelect(_) | Scatter(_) | ScatterSelect(_) => {
                return None;
            }
            Cast(to) | Widen { to, .. } | Narrow(to) => ValueType::Vector(to),
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
            Shuffle(_) if self.element == Element::I32 => &[v, v],
            Shuffle(_) => return Err("paired shuffle requires i32 lanes".into()),
            AlignmentPrefix(_) if self.element == Element::I32 => &[s],
            Compress if self.element == Element::I32 => &[v, m],
            ExtractWrapping if self.element == Element::I32 => &[v, s],
            ReplaceWrapping if self.element == Element::I32 => &[v, s, s],
            ExtractWrapping | ReplaceWrapping => return Err("expected i32 lanes".into()),
            Swizzle if self.element == Element::I32 => &[v, v],
            Expand if self.element == Element::I32 => &[v, m, v],
            Compress | Expand | Swizzle | AlignmentPrefix(_) => {
                return Err("expected i32 lanes".into());
            }
            Count if self.element == Element::I32 => &[m],
            CompressStore(_) if self.element == Element::I32 => &[s, m, v],
            Count | CompressStore(_) => {
                return Err("mask count and compact store require i32 lanes".into());
            }
            Shr(_) | RotateLeft(_) if matches!(self.element, Element::I32 | Element::U32) => &[v],
            PrefixSum | Last | ReduceSum if self.element == Element::I32 => &[v],
            ReduceSum => return Err("SIMD reduce_sum requires i32 lanes".into()),
            Shr(_) | RotateLeft(_) | PrefixSum | Last => {
                return Err("shift, rotate, prefix_sum and last require i32 lanes".into());
            }
            SplatScalar if self.element == Element::I32 => &[s],
            SplatScalar => return Err("runtime splat expects an i32 element".into()),
            Div if self.element.is_float() => &[v, v],
            Div => return Err("SIMD division requires floating point elements".into()),
            Min | Max if matches!(self.element, Element::I32 | Element::U32) => &[v, v],
            Min | Max => return Err("SIMD min and max require i32 or u32 lanes".into()),
            Sqrt if self.element.is_float() => &[v],
            MulAdd if self.element.is_float() => &[v, v, v],
            Sqrt | MulAdd => {
                return Err("SIMD sqrt and mul_add require floating point lanes".into());
            }
            Cast(to) if self.element.bytes() == to.bytes() => &[v],
            Widen { to, .. }
                if self.element.bytes() * 2 == to.bytes()
                    && self.element.is_float() == to.is_float() =>
            {
                &[v]
            }
            Narrow(to)
                if self.element.bytes() == to.bytes() * 2
                    && self.element.is_float() == to.is_float() =>
            {
                &[v, v]
            }
            Cast(_) | Widen { .. } | Narrow(_) => return Err("invalid SIMD lane conversion".into()),
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
            AlignmentPrefix(b) | Load(b) | LoadSelect(b) | Gather(b) | GatherSelect(b) => {
                (b, false)
            }
            CompressStore(b) | Store(b) | StoreSelect(b) | Scatter(b) | ScatterSelect(b) => {
                (b, true)
            }
            _ => return Ok(()),
        };
        let kind = buffers.get(buffer.index()).ok_or("invalid SIMD buffer")?;
        if kind.context_signature().is_some() || kind.element() != self.element {
            return Err("SIMD buffer element type mismatch".into());
        }
        if write && !kind.writable() {
            return Err("SIMD store to read-only buffer".into());
        }
        Ok(())
    }
}
