use super::{Binary, Lane, Ternary, Unary, Value};

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
