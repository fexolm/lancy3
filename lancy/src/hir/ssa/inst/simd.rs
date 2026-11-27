use super::{Binary, Lane, Value};

/// Contiguous lanes or signed element indices relative to a base pointer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VectorAddress {
    /// The pointer of element zero.
    pub base: Value,
    /// A vector of signed element indices for a gather or scatter; `None`
    /// for contiguous lanes.
    pub indices: Option<Value>,
}

/// Operations on scalable vectors. Each element type has a target-dependent lane count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Simd {
    /// The number of lanes of the type at the target width, as an `i32`.
    LaneCount(Lane),
    /// Lane numbers starting at zero, in increasing address order.
    Iota(Lane),
    /// Concatenate two vectors and select lanes by unsigned indices. Out-of-range lanes are zero.
    Shuffle(Value, Value, Value),
    /// Fold lanes in increasing order. The verifier accepts arithmetic and bitwise reductions.
    Reduce(Binary, Value),
    /// `1` if any lane of an integer vector has its high bit set, else `0`.
    AnyTrue(Value),
    /// `1` if every lane of an integer vector has its high bit set, else `0`.
    AllTrue(Value),
    /// Pack active lanes to the low end, or expand low lanes into active positions. Unused lanes become zero.
    Compress(Value, Value),
    /// See `Compress`.
    Expand(Value, Value),
    /// Convert the lower or upper half into elements twice as wide.
    Widen {
        /// The wider lane type.
        to: Lane,
        /// Convert the upper half instead of the lower.
        high: bool,
        /// The vector to convert.
        value: Value,
    },
    /// Convert and concatenate two vectors of elements twice as wide as the result.
    Narrow(Lane, Value, Value),
    /// Every lane equal to the scalar.
    Splat(Value),
    /// The lane at the `i32` index; out of range traps.
    ExtractLane(Value, Value),
    /// The vector with the lane at the `i32` index replaced by the scalar; out of range traps.
    InsertLane(Value, Value, Value),
    /// Read only lanes where the mask's high bit is set. Other lanes keep passthrough.
    MaskedLoad {
        /// Where the lanes are.
        address: VectorAddress,
        /// Integer lanes; a set high bit activates the lane.
        mask: Value,
        /// The value of inactive lanes.
        passthrough: Value,
    },
    /// Write active lanes in increasing lane order, including repeated scatter indices.
    MaskedStore {
        /// Where the lanes go.
        address: VectorAddress,
        /// Integer lanes; a set high bit activates the lane.
        mask: Value,
        /// The vector to store.
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
        /// The pointer of the first written element.
        base: Value,
        /// Integer lanes; a set high bit selects the lane.
        mask: Value,
        /// The vector to store.
        value: Value,
        /// The most elements to write.
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
