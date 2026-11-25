use super::{Binary, Lane, Value};

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
    Splat(Value),
    ExtractLane(Value, Value),
    InsertLane(Value, Value, Value),
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
    /// Lanes below the `i32` count are all ones and the others zero, so a
    /// count of zero or less selects none: a predicate of integer lanes.
    FirstLanes(Lane, Value),
    /// The number of lanes of an integer vector whose high bit is set, as an `i32`.
    MaskCount(Value),
    /// Write the lanes of `value` whose `mask` lane has the high bit set to
    /// consecutive elements from `base`, lowest lane first, but at most
    /// `limit` of them (an `i32`; zero or less writes none). Nothing past
    /// the written elements is accessed.
    CompressStore {
        base: Value,
        mask: Value,
        value: Value,
        limit: Value,
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
            Self::CompressStore {
                base,
                mask,
                value,
                limit,
            } => {
                *base = map(*base);
                *mask = map(*mask);
                *value = map(*value);
                *limit = map(*limit);
            }
            Self::Reduce(_, v)
            | Self::FirstLanes(_, v)
            | Self::MaskCount(v)
            | Self::AnyTrue(v)
            | Self::AllTrue(v)
            | Self::Widen { value: v, .. }
            | Self::Splat(v) => *v = map(*v),
            Self::Shuffle(a, b, c) | Self::InsertLane(a, b, c) => {
                *a = map(*a);
                *b = map(*b);
                *c = map(*c);
            }
            Self::Compress(a, b)
            | Self::Expand(a, b)
            | Self::Narrow(_, a, b)
            | Self::ExtractLane(a, b) => {
                *a = map(*a);
                *b = map(*b);
            }
            Self::Iota(_) | Self::LaneCount(_) => {}
        }
    }
}
