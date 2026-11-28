use super::{ScalarType, Slice, Types};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    Scalar(ScalarType),
    /// A vector of lanes of a type whose [`ScalarType::is_lane`] holds.
    Vector(ScalarType),
    /// One predicate per lane of a vector of the type.
    Mask(ScalarType),
    /// The address of the first element of a slice of this kind: the
    /// pointer half of a [`Slice`].
    Ptr(super::BufferKind),
}

impl ValueType {
    /// Whether a register can have this type: vectors and masks need a
    /// lane element.
    pub const fn is_supported(self) -> bool {
        match self {
            Self::Scalar(_) | Self::Ptr(_) => true,
            Self::Vector(e) | Self::Mask(e) => e.is_lane(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdOp {
    ElementWidth,
    ByteWidth,
    FirstN,
    Splat(u64),
    SplatScalar,
    Iota,
    Shuffle(crate::simd::Shuffle),
    PrefixSum,
    AlignmentPrefix(Slice),
    Compress,
    Expand,
    Swizzle,
    Count,
    Shr(u32),
    RotateLeft(u32),
    CompressStore(Slice),
    Last,
    ExtractWrapping,
    ReplaceWrapping,
    Load(Slice),
    Store(Slice),
    LoadSelect(Slice),
    StoreSelect(Slice),
    Gather(Slice),
    Scatter(Slice),
    GatherSelect(Slice),
    ScatterSelect(Slice),
    Add,
    AddMasked,
    Sub,
    SaturatingAdd,
    SaturatingSub,
    Mul,
    Div,
    Sqrt,
    MulAdd,
    Cast(ScalarType),
    Widen { to: ScalarType, high: bool },
    Narrow(ScalarType),
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
    pub element: ScalarType,
    pub op: SimdOp,
}

/// A value an operation takes or produces, relative to its element `e`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    /// A vector of `e`.
    Vector,
    /// A mask over lanes of `e`.
    Mask,
    /// A vector of `e.index()` lane indices.
    Indices,
    /// A lane index or count: scalar `i32`.
    Lane,
    /// A memory offset or a lane count: scalar `Index`.
    Offset,
    /// A scalar `bool`.
    Bool,
    /// A scalar of the element type `e`.
    Element,
    /// A vector of the given element.
    Converted(ScalarType),
}

impl Slot {
    pub const fn value_type(self, element: ScalarType) -> ValueType {
        match self {
            Self::Vector => ValueType::Vector(element),
            Self::Mask => ValueType::Mask(element),
            Self::Indices => ValueType::Vector(element.index()),
            Self::Lane => ValueType::Scalar(ScalarType::I32),
            Self::Offset => ValueType::Scalar(ScalarType::Index),
            Self::Bool => ValueType::Scalar(ScalarType::Bool),
            Self::Element => ValueType::Scalar(element),
            Self::Converted(to) => ValueType::Vector(to),
        }
    }
}

/// The elements an operation accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Elements {
    Any,
    I32,
    I32OrU32,
    Integer,
    Float,
    /// A conversion to an element of the same width.
    SameWidth(ScalarType),
    /// A conversion to an element of twice the width and the same kind.
    Wider(ScalarType),
    /// A conversion to an element of half the width and the same kind.
    Narrower(ScalarType),
}

impl Elements {
    pub const fn accepts(self, e: ScalarType) -> bool {
        match self {
            Self::Any => true,
            Self::I32 => matches!(e, ScalarType::I32),
            Self::I32OrU32 => matches!(e, ScalarType::I32 | ScalarType::U32),
            Self::Integer => !e.is_float(),
            Self::Float => e.is_float(),
            Self::SameWidth(to) => e.bytes() == to.bytes(),
            Self::Wider(to) => e.bytes() * 2 == to.bytes() && e.is_float() == to.is_float(),
            Self::Narrower(to) => e.bytes() == to.bytes() * 2 && e.is_float() == to.is_float(),
        }
    }
}

impl std::fmt::Display for Elements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Any => f.write_str("any"),
            Self::I32 => f.write_str("i32"),
            Self::I32OrU32 => f.write_str("i32 or u32"),
            Self::Integer => f.write_str("integer"),
            Self::Float => f.write_str("floating point"),
            Self::SameWidth(to) => write!(f, "{}-byte", to.bytes()),
            Self::Wider(to) | Self::Narrower(to) => write!(
                f,
                "{} {}-byte",
                if to.is_float() {
                    "floating point"
                } else {
                    "integer"
                },
                match self {
                    Self::Wider(_) => to.bytes() / 2,
                    _ => to.bytes() * 2,
                }
            ),
        }
    }
}

/// How an operation uses its buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// Only the address of the slice: nothing is read and nothing can fail.
    Address,
    Read,
    Write,
}

/// The slice an operation accesses. The first operand addresses it: an
/// [`Slot::Offset`], or [`Slot::Indices`] when `indexed`. A `masked`
/// operation touches only the lanes of its [`Slot::Mask`] operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Memory {
    pub slice: Slice,
    pub access: Access,
    pub indexed: bool,
    pub masked: bool,
}

/// What the verifier and the backends know about an operation besides its
/// semantics: its name, operands, result, elements and memory access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpInfo {
    pub name: &'static str,
    pub operands: &'static [Slot],
    /// `None` for an operation that only writes memory.
    pub output: Option<Slot>,
    pub elements: Elements,
    pub memory: Option<Memory>,
}

impl SimdOp {
    /// The one description of the operation, from which the verifier, the
    /// result type and [`may_trap`](Self::may_trap) follow.
    pub const fn info(self) -> OpInfo {
        use Elements as E;
        use SimdOp::*;
        use Slot::*;
        let (name, operands, output, elements): (_, &'static [Slot], _, _) = match self {
            ByteWidth => ("byte_width", &[], Some(Offset), E::Any),
            ElementWidth => ("element_width", &[], Some(Offset), E::Any),
            FirstN => ("first_n", &[Offset], Some(Mask), E::Any),
            Splat(_) => ("splat", &[], Some(Vector), E::Any),
            SplatScalar => ("splat_scalar", &[Element], Some(Vector), E::Any),
            Iota => ("iota", &[], Some(Vector), E::Any),
            Shuffle(_) => ("shuffle", &[Vector, Vector], Some(Vector), E::I32),
            PrefixSum => ("prefix_sum", &[Vector], Some(Vector), E::I32),
            AlignmentPrefix(_) => ("alignment_prefix", &[Offset], Some(Offset), E::I32),
            Compress => ("compress", &[Vector, Mask], Some(Vector), E::I32),
            Expand => ("expand", &[Vector, Mask, Vector], Some(Vector), E::I32),
            Swizzle => ("swizzle", &[Vector, Vector], Some(Vector), E::I32),
            Count => ("count", &[Mask], Some(Lane), E::Any),
            Shr(_) => ("shr", &[Vector], Some(Vector), E::I32OrU32),
            RotateLeft(_) => ("rotate_left", &[Vector], Some(Vector), E::I32OrU32),
            CompressStore(_) => ("compress_store", &[Offset, Mask, Vector], None, E::I32),
            Last => ("last", &[Vector], Some(Element), E::Any),
            ExtractWrapping => ("extract_wrapping", &[Vector, Lane], Some(Element), E::Any),
            ReplaceWrapping => (
                "replace_wrapping",
                &[Vector, Lane, Element],
                Some(Vector),
                E::Any,
            ),
            Load(_) => ("load", &[Offset], Some(Vector), E::Any),
            Store(_) => ("store", &[Offset, Vector], None, E::Any),
            LoadSelect(_) => ("load_select", &[Offset, Mask, Vector], Some(Vector), E::Any),
            StoreSelect(_) => ("store_select", &[Offset, Mask, Vector], None, E::Any),
            Gather(_) => ("gather", &[Indices], Some(Vector), E::Any),
            Scatter(_) => ("scatter", &[Indices, Vector], None, E::Any),
            GatherSelect(_) => (
                "gather_select",
                &[Indices, Mask, Vector],
                Some(Vector),
                E::Any,
            ),
            ScatterSelect(_) => ("scatter_select", &[Indices, Mask, Vector], None, E::Any),
            AddMasked => (
                "add_masked",
                &[Vector, Mask, Vector, Vector],
                Some(Vector),
                E::Any,
            ),
            Add => ("add", &[Vector, Vector], Some(Vector), E::Any),
            Sub => ("sub", &[Vector, Vector], Some(Vector), E::Any),
            SaturatingAdd => (
                "saturating_add",
                &[Vector, Vector],
                Some(Vector),
                E::Integer,
            ),
            SaturatingSub => (
                "saturating_sub",
                &[Vector, Vector],
                Some(Vector),
                E::Integer,
            ),
            Mul => ("mul", &[Vector, Vector], Some(Vector), E::Any),
            Div => ("div", &[Vector, Vector], Some(Vector), E::Float),
            Sqrt => ("sqrt", &[Vector], Some(Vector), E::Float),
            MulAdd => ("mul_add", &[Vector, Vector, Vector], Some(Vector), E::Float),
            Cast(to) => ("cast", &[Vector], Some(Converted(to)), E::SameWidth(to)),
            Widen { to, .. } => ("widen", &[Vector], Some(Converted(to)), E::Wider(to)),
            Narrow(to) => (
                "narrow",
                &[Vector, Vector],
                Some(Converted(to)),
                E::Narrower(to),
            ),
            ReduceSum => ("reduce_sum", &[Vector], Some(Element), E::Integer),
            Min => ("min", &[Vector, Vector], Some(Vector), E::Integer),
            Max => ("max", &[Vector, Vector], Some(Vector), E::Integer),
            And => ("and", &[Vector, Vector], Some(Vector), E::Integer),
            Or => ("or", &[Vector, Vector], Some(Vector), E::Integer),
            Xor => ("xor", &[Vector, Vector], Some(Vector), E::Integer),
            Not => ("not", &[Vector], Some(Vector), E::Integer),
            Eq => ("eq", &[Vector, Vector], Some(Mask), E::Any),
            Ne => ("ne", &[Vector, Vector], Some(Mask), E::Any),
            Lt => ("lt", &[Vector, Vector], Some(Mask), E::Any),
            Le => ("le", &[Vector, Vector], Some(Mask), E::Any),
            Gt => ("gt", &[Vector, Vector], Some(Mask), E::Any),
            Ge => ("ge", &[Vector, Vector], Some(Mask), E::Any),
            MaskAnd => ("mask_and", &[Mask, Mask], Some(Mask), E::Any),
            MaskOr => ("mask_or", &[Mask, Mask], Some(Mask), E::Any),
            MaskXor => ("mask_xor", &[Mask, Mask], Some(Mask), E::Any),
            MaskNot => ("mask_not", &[Mask], Some(Mask), E::Any),
            Any => ("any", &[Mask], Some(Bool), E::Any),
            All => ("all", &[Mask], Some(Bool), E::Any),
            Select => ("select", &[Mask, Vector, Vector], Some(Vector), E::Any),
        };
        const fn memory(
            slice: Slice,
            access: Access,
            indexed: bool,
            masked: bool,
        ) -> Option<Memory> {
            Some(Memory {
                slice,
                access,
                indexed,
                masked,
            })
        }
        let memory = match self {
            AlignmentPrefix(b) => memory(b, Access::Address, false, false),
            Load(b) => memory(b, Access::Read, false, false),
            LoadSelect(b) => memory(b, Access::Read, false, true),
            Gather(b) => memory(b, Access::Read, true, false),
            GatherSelect(b) => memory(b, Access::Read, true, true),
            Store(b) => memory(b, Access::Write, false, false),
            StoreSelect(b) | CompressStore(b) => memory(b, Access::Write, false, true),
            Scatter(b) => memory(b, Access::Write, true, false),
            ScatterSelect(b) => memory(b, Access::Write, true, true),
            _ => None,
        };
        OpInfo {
            name,
            operands,
            output,
            elements,
            memory,
        }
    }

    /// This operation with its slice, if it has one, replaced by `f` of it.
    pub fn map_slice<E>(self, f: impl FnOnce(Slice) -> Result<Slice, E>) -> Result<Self, E> {
        use SimdOp::*;
        Ok(match self {
            AlignmentPrefix(s) => AlignmentPrefix(f(s)?),
            CompressStore(s) => CompressStore(f(s)?),
            Load(s) => Load(f(s)?),
            Store(s) => Store(f(s)?),
            LoadSelect(s) => LoadSelect(f(s)?),
            StoreSelect(s) => StoreSelect(f(s)?),
            Gather(s) => Gather(f(s)?),
            Scatter(s) => Scatter(f(s)?),
            GatherSelect(s) => GatherSelect(f(s)?),
            ScatterSelect(s) => ScatterSelect(f(s)?),
            op => op,
        })
    }

    /// Whether the operation can fail: only buffer accesses check bounds.
    pub const fn may_trap(self) -> bool {
        match self.info().memory {
            Some(memory) => !matches!(memory.access, Access::Address),
            None => false,
        }
    }
}

impl Simd {
    pub fn output(self) -> Option<ValueType> {
        self.op
            .info()
            .output
            .map(|slot| slot.value_type(self.element))
    }

    pub fn validate(self, args: &[ValueType], types: Types<'_>) -> crate::Result<()> {
        let info = self.op.info();
        let converted = match info.output {
            Some(Slot::Converted(to)) => Some(to),
            _ => None,
        };
        if let Some(e) = std::iter::once(self.element)
            .chain(converted)
            .find(|e| !e.is_lane())
        {
            return Err(crate::BuildError::unsupported(format!(
                "SIMD lanes of {} are not supported",
                e.name()
            )));
        }
        if !info.elements.accepts(self.element) {
            return Err(crate::BuildError::verify(format!(
                "SIMD {} requires {} lanes, got {}",
                info.name,
                info.elements,
                self.element.name()
            )));
        }
        // Masks of elements with the same index type are interchangeable.
        let compatible = |(actual, slot): (&ValueType, &Slot)| match (actual, slot) {
            (ValueType::Mask(a), Slot::Mask) => a.index() == self.element.index(),
            _ => *actual == slot.value_type(self.element),
        };
        if args.len() != info.operands.len() || !args.iter().zip(info.operands).all(compatible) {
            let expected: Vec<_> = info
                .operands
                .iter()
                .map(|slot| slot.value_type(self.element))
                .collect();
            return Err(crate::BuildError::verify(format!(
                "invalid SIMD operands for {}: expected {expected:?}, got {args:?}",
                info.name
            )));
        }
        let Some(memory) = info.memory else {
            return Ok(());
        };
        let kind = memory
            .slice
            .kind(types)
            .ok_or_else(|| crate::BuildError::verify("invalid SIMD buffer"))?;
        if kind.context_signature().is_some() || kind.element() != self.element {
            return Err(crate::BuildError::verify(
                "SIMD buffer element type mismatch",
            ));
        }
        if memory.access == Access::Write && !kind.writable() {
            return Err(crate::BuildError::verify("SIMD store to read-only buffer"));
        }
        Ok(())
    }
}
