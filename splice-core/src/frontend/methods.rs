//! The Rust methods of runtime kernel values that are core operations:
//! `x.saturating_add(y)` on a runtime integer stages one portable
//! [`Binary`] operation, which every backend implements. These are not
//! intrinsics (see [`mod@crate::intrinsic`]): no backend chooses them.
//!
//! The intrinsics of [`Lane`](crate::simd::Lane) and [`Mask`](crate::simd::Mask)
//! are the SIMD part ([`simd`]): each declared one stages [`SimdOp`]s on the
//! operands the kernel passes, typed by the builder.
use super::KernelBuilder;
use crate::hir::{Binary, Buffer, Constant, Reg, ScalarType, Simd, SimdOp, ValueType};
use crate::simd::{Element, Shuffle};

/// How a core method stages its operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lowering {
    /// One binary operation of the receiver and the argument.
    Binary(Binary),
    /// `0 - x` without an overflow check (`wrapping_neg`).
    WrappingNeg,
}

impl Lowering {
    fn arity(self) -> usize {
        match self {
            Self::Binary(_) => 2,
            Self::WrappingNeg => 1,
        }
    }
}

/// A core method of the integer types and its lowering.
const INTEGER_METHODS: [(&str, Lowering); 9] = [
    ("wrapping_add", Lowering::Binary(Binary::Add)),
    ("wrapping_sub", Lowering::Binary(Binary::Sub)),
    ("wrapping_mul", Lowering::Binary(Binary::Mul)),
    ("wrapping_neg", Lowering::WrappingNeg),
    ("saturating_add", Lowering::Binary(Binary::SaturatingAdd)),
    ("saturating_sub", Lowering::Binary(Binary::SaturatingSub)),
    ("saturating_mul", Lowering::Binary(Binary::SaturatingMul)),
    ("min", Lowering::Binary(Binary::Min)),
    ("max", Lowering::Binary(Binary::Max)),
];

/// The lowering of `method` on a receiver of type `ty`, if it is a core method.
fn lookup(ty: ScalarType, method: &str) -> Option<Lowering> {
    if !ty.is_integer() {
        return None;
    }
    INTEGER_METHODS
        .iter()
        .find(|(name, _)| *name == method)
        .map(|&(_, lowering)| lowering)
}

/// Whether some scalar type has the core method `method`.
#[doc(hidden)]
pub const fn is_method(method: &str) -> bool {
    let mut i = 0;
    while i < INTEGER_METHODS.len() {
        if same(INTEGER_METHODS[i].0, method) {
            return true;
        }
        i += 1;
    }
    false
}

const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Stage the Rust method `method` on runtime scalars: `args[0]` is the
/// receiver. A method that is not a core operation of the receiver's type, or
/// with operands of other types, is a staging error.
#[doc(hidden)]
pub fn method(builder: &mut dyn KernelBuilder, method: &str, args: &[Reg]) -> crate::Result<Reg> {
    let receiver = args.first().and_then(|&r| builder.reg_type(r));
    let Some(ValueType::Scalar(ty)) = receiver else {
        return Err(crate::BuildError::staging(format!(
            "`{method}` requires a runtime scalar receiver"
        )));
    };
    match lookup(ty, method) {
        Some(lowering) if operands_have_type(builder, args, ty, lowering) => {
            stage(builder, lowering, ty, args)
        }
        Some(lowering) => Err(crate::BuildError::staging(format!(
            "`{ty}::{method}` takes {} operand(s) of type {ty}",
            lowering.arity()
        ))),
        None => Err(crate::BuildError::staging(format!(
            "`{ty}::{method}` is not a kernel operation"
        ))),
    }
}

fn operands_have_type(
    builder: &dyn KernelBuilder,
    args: &[Reg],
    ty: ScalarType,
    lowering: Lowering,
) -> bool {
    args.len() == lowering.arity()
        && args
            .iter()
            .all(|&r| builder.reg_type(r) == Some(ValueType::Scalar(ty)))
}

fn stage(
    builder: &mut dyn KernelBuilder,
    lowering: Lowering,
    ty: ScalarType,
    args: &[Reg],
) -> crate::Result<Reg> {
    match lowering {
        Lowering::Binary(op) => builder.binary(op, args[0], args[1]),
        Lowering::WrappingNeg => {
            let zero = builder.constant(Constant::new(ty, 0))?;
            builder.binary(Binary::Sub, zero, args[0])
        }
    }
}

/// An operand of a SIMD method as the kernel passes it.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub enum Operand {
    /// A register of the builder.
    Reg(Reg),
    /// A value the generator knows.
    Known(Constant),
    /// A runtime slice.
    Buffer(Buffer),
}

/// The `self` of a SIMD intrinsic call: a runtime vector or mask, or the
/// `Lane<element>` of an associated function (`Lane::<T>::load`).
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub enum SimdSelf {
    Value(Reg),
    Lane(Element),
}

/// The SIMD intrinsic being staged: its receiver, its element and its operands.
struct SimdCall<'a> {
    builder: &'a mut dyn KernelBuilder,
    method: &'a str,
    receiver: Option<Reg>,
    element: Element,
    args: &'a [Operand],
    /// The element of the result, for a method generic over it (`cast::<U>()`).
    output: Option<Element>,
}

type SimdLowering = fn(&mut SimdCall<'_>) -> crate::Result<Option<Reg>>;

/// The lowering of each intrinsic that `splice::simd` declares for `Lane`,
/// by method in name order: the portable SIMD operations it stages. An operator trait's
/// method (`add`) also lowers its compound assignment (`add_assign`).
const LANE: [(&str, SimdLowering); 51] = [
    ("add", |c| c.with_args(SimdOp::Add, 1)),
    ("add_masked", |c| {
        let (mask, rhs, fallback) = (c.value(0)?, c.value(1)?, c.value(2)?);
        let sum = c.op(SimdOp::Add, &[c.this()?, rhs])?;
        let sum = sum.ok_or_else(|| crate::BuildError::staging("missing SIMD result"))?;
        c.op(SimdOp::Select, &[mask, sum, fallback])
    }),
    ("alignment_prefix", |c| {
        let (buffer, offset) = (c.buffer(0)?, c.scalar(1, ScalarType::Index)?);
        c.op(SimdOp::AlignmentPrefix(buffer), &[offset])
    }),
    ("bitand", |c| c.with_args(SimdOp::And, 1)),
    ("bitor", |c| c.with_args(SimdOp::Or, 1)),
    ("bitxor", |c| c.with_args(SimdOp::Xor, 1)),
    ("cast", |c| {
        let to = c.output()?;
        c.with_args(SimdOp::Cast(to), 0)
    }),
    ("compress", |c| c.with_args(SimdOp::Compress, 1)),
    ("compress_store", |c| c.store(SimdOp::CompressStore, true)),
    ("deinterleave_even", |c| {
        c.shuffle(Shuffle::DeinterleaveEven)
    }),
    ("deinterleave_odd", |c| c.shuffle(Shuffle::DeinterleaveOdd)),
    ("div", |c| c.with_args(SimdOp::Div, 1)),
    ("expand", |c| c.with_args(SimdOp::Expand, 2)),
    ("extract_wrapping", |c| {
        let index = c.scalar(0, ScalarType::I32)?;
        c.op(SimdOp::ExtractWrapping, &[c.this()?, index])
    }),
    ("first_n", |c| {
        let count = c.scalar(0, ScalarType::Index)?;
        c.op(SimdOp::FirstN, &[count])
    }),
    ("gather", |c| {
        let (buffer, indices) = (c.buffer(0)?, c.value(1)?);
        c.op(SimdOp::Gather(buffer), &[indices])
    }),
    ("gather_select", |c| {
        let (buffer, mask, indices, fallback) =
            (c.buffer(0)?, c.value(1)?, c.value(2)?, c.value(3)?);
        c.op(SimdOp::GatherSelect(buffer), &[indices, mask, fallback])
    }),
    ("interleave_high", |c| c.shuffle(Shuffle::InterleaveHigh)),
    ("interleave_low", |c| c.shuffle(Shuffle::InterleaveLow)),
    ("iota", |c| c.op(SimdOp::Iota, &[])),
    ("lanes", |c| c.op(SimdOp::Lanes, &[])),
    ("last", |c| c.with_args(SimdOp::Last, 0)),
    ("load", |c| {
        let (buffer, offset) = (c.buffer(0)?, c.scalar(1, ScalarType::Index)?);
        c.op(SimdOp::Load(buffer), &[offset])
    }),
    ("load_select", |c| {
        let (buffer, offset) = (c.buffer(0)?, c.scalar(1, ScalarType::Index)?);
        let (mask, fallback) = (c.value(2)?, c.value(3)?);
        c.op(SimdOp::LoadSelect(buffer), &[offset, mask, fallback])
    }),
    ("mul", |c| c.with_args(SimdOp::Mul, 1)),
    ("mul_add", |c| c.with_args(SimdOp::MulAdd, 2)),
    ("narrow", |c| {
        let to = c.output()?;
        c.with_args(SimdOp::Narrow(to), 1)
    }),
    ("not", |c| c.with_args(SimdOp::Not, 0)),
    ("prefix_sum", |c| c.with_args(SimdOp::PrefixSum, 0)),
    ("reduce_sum", |c| c.with_args(SimdOp::ReduceSum, 0)),
    ("replace_wrapping", |c| {
        let index = c.scalar(0, ScalarType::I32)?;
        let value = c.scalar(1, ScalarType::I32)?;
        c.op(SimdOp::ReplaceWrapping, &[c.this()?, index, value])
    }),
    ("rotate_left", |c| {
        let count = c.count(0)?;
        c.with_args(SimdOp::RotateLeft(count), 0)
    }),
    ("scatter", |c| c.scatter(SimdOp::Scatter, false)),
    ("scatter_select", |c| c.scatter(SimdOp::ScatterSelect, true)),
    ("shr", |c| {
        let count = c.count(0)?;
        c.with_args(SimdOp::Shr(count), 0)
    }),
    ("simd_eq", |c| c.with_args(SimdOp::Eq, 1)),
    ("simd_ge", |c| c.with_args(SimdOp::Ge, 1)),
    ("simd_gt", |c| c.with_args(SimdOp::Gt, 1)),
    ("simd_le", |c| c.with_args(SimdOp::Le, 1)),
    ("simd_lt", |c| c.with_args(SimdOp::Lt, 1)),
    ("simd_max", |c| c.with_args(SimdOp::Max, 1)),
    ("simd_min", |c| c.with_args(SimdOp::Min, 1)),
    ("simd_ne", |c| c.with_args(SimdOp::Ne, 1)),
    ("splat", |c| match c.arg(0)? {
        // `Lane::splat` of a known value carries its bit pattern.
        Operand::Known(value) => c.op(SimdOp::Splat(value.bits()), &[]),
        Operand::Reg(value) if c.element == Element::I32 => c.op(SimdOp::SplatScalar, &[value]),
        _ => Err(c.error("splats a known value or a runtime i32")),
    }),
    ("sqrt", |c| c.with_args(SimdOp::Sqrt, 0)),
    ("store", |c| c.store(SimdOp::Store, false)),
    ("store_select", |c| c.store(SimdOp::StoreSelect, true)),
    ("sub", |c| c.with_args(SimdOp::Sub, 1)),
    ("swizzle", |c| c.with_args(SimdOp::Swizzle, 1)),
    ("widen_high", |c| {
        let to = c.output()?;
        c.with_args(SimdOp::Widen { to, high: true }, 0)
    }),
    ("widen_low", |c| {
        let to = c.output()?;
        c.with_args(SimdOp::Widen { to, high: false }, 0)
    }),
];

/// The lowering of each intrinsic that `splice::simd` declares for `Mask`,
/// but `select` ([`select`]), whose element is the selected data's; in
/// name order.
const MASK: [(&str, SimdLowering); 7] = [
    ("all", |c| c.with_args(SimdOp::All, 0)),
    ("any", |c| c.with_args(SimdOp::Any, 0)),
    ("bitand", |c| c.with_args(SimdOp::MaskAnd, 1)),
    ("bitor", |c| c.with_args(SimdOp::MaskOr, 1)),
    ("bitxor", |c| c.with_args(SimdOp::MaskXor, 1)),
    ("count", |c| c.with_args(SimdOp::Count, 0)),
    ("not", |c| c.with_args(SimdOp::MaskNot, 0)),
];

/// The lowering of `method` of `Lane` (`mask` false) or `Mask`.
fn simd_lowering(mask: bool, method: &str) -> Option<SimdLowering> {
    if mask && method == "select" {
        return Some(select);
    }
    let method = method.strip_suffix("_assign").unwrap_or(method);
    let table: &[(&str, SimdLowering)] = if mask { &MASK } else { &LANE };
    // Staging looks up every SIMD call of a kernel.
    table
        .binary_search_by(|(name, _)| name.cmp(&method))
        .ok()
        .map(|index| table[index].1)
}

/// `mask.select(yes, no)`: a selection of the data's element.
fn select(c: &mut SimdCall<'_>) -> crate::Result<Option<Reg>> {
    let (yes, no) = (c.value(0)?, c.value(1)?);
    let Some(ValueType::Vector(element)) = c.builder.reg_type(yes) else {
        return Err(c.error("requires two vectors"));
    };
    let simd = Simd {
        element,
        op: SimdOp::Select,
    };
    c.builder.simd(simd, &[c.this()?, yes, no])
}

/// Stage the SIMD intrinsic `method` of `this` with `args` (the method's
/// arguments in order); `output` is the element of a result generic over
/// it (`cast::<U>()`). The result is `None` for a method that only writes
/// memory. Generated code calls it for each `Lane`/`Mask` method,
/// associated function and operator of a kernel.
#[doc(hidden)]
pub fn simd(
    builder: &mut dyn KernelBuilder,
    method: &str,
    this: SimdSelf,
    args: &[Operand],
    output: Option<Element>,
) -> crate::Result<Option<Reg>> {
    let (receiver, element, mask) = match this {
        SimdSelf::Lane(element) => (None, element, false),
        SimdSelf::Value(receiver) => match builder.reg_type(receiver) {
            Some(ValueType::Vector(element)) => (Some(receiver), element, false),
            Some(ValueType::Mask(element)) => (Some(receiver), element, true),
            _ => {
                return Err(crate::BuildError::staging(format!(
                    "`{method}` requires a runtime vector or mask receiver"
                )));
            }
        },
    };
    let lowering = simd_lowering(mask, method).ok_or_else(|| {
        crate::BuildError::staging(format!("`{method}` is not a kernel SIMD operation"))
    })?;
    lowering(&mut SimdCall {
        builder,
        method,
        receiver,
        element,
        args,
        output,
    })
}

impl SimdCall<'_> {
    fn error(&self, message: &str) -> crate::BuildError {
        crate::BuildError::staging(format!("`{}`: {message}", self.method))
    }

    fn this(&self) -> crate::Result<Reg> {
        self.receiver
            .ok_or_else(|| self.error("requires a vector or mask receiver"))
    }

    fn arg(&self, index: usize) -> crate::Result<Operand> {
        self.args
            .get(index)
            .copied()
            .ok_or_else(|| self.error("missing operand"))
    }

    /// Argument `index`, a runtime vector or mask.
    fn value(&self, index: usize) -> crate::Result<Reg> {
        match self.arg(index)? {
            Operand::Reg(reg) => Ok(reg),
            _ => Err(self.error("expected a runtime vector or mask operand")),
        }
    }

    /// Argument `index` as a register of `ty`; a known value becomes a constant.
    fn scalar(&mut self, index: usize, ty: ScalarType) -> crate::Result<Reg> {
        match self.arg(index)? {
            Operand::Reg(reg) => Ok(reg),
            Operand::Known(value) => {
                let value = value
                    .cast(ty)
                    .ok_or_else(|| self.error("expected an integer operand"))?;
                self.builder.constant(value)
            }
            Operand::Buffer(_) => Err(self.error("expected a scalar operand")),
        }
    }

    /// Argument `index`, a known shift or rotation count, modulo 32.
    fn count(&self, index: usize) -> crate::Result<u32> {
        match self.arg(index)? {
            Operand::Known(value) => Ok(value
                .cast(ScalarType::U32)
                .ok_or_else(|| self.error("expected an integer count"))?
                .bits() as u32
                & 31),
            _ => Err(self.error("the count must be known when the kernel is built")),
        }
    }

    fn buffer(&self, index: usize) -> crate::Result<Buffer> {
        match self.arg(index)? {
            Operand::Buffer(buffer) => Ok(buffer),
            _ => Err(self.error("expected a runtime slice")),
        }
    }

    fn output(&self) -> crate::Result<Element> {
        self.output
            .ok_or_else(|| self.error("specify the output lane type"))
    }

    /// `op` on the element of `self`.
    fn op(&mut self, op: SimdOp, operands: &[Reg]) -> crate::Result<Option<Reg>> {
        let simd = Simd {
            element: self.element,
            op,
        };
        self.builder.simd(simd, operands)
    }

    /// `op` on the receiver and its first `count` arguments, all vectors or masks.
    fn with_args(&mut self, op: SimdOp, count: usize) -> crate::Result<Option<Reg>> {
        // At most two arguments; no allocation on this path of every call.
        let mut operands = [self.this()?; 3];
        for index in 0..count {
            operands[1 + index] = self.value(index)?;
        }
        self.op(op, &operands[..=count])
    }

    fn shuffle(&mut self, order: Shuffle) -> crate::Result<Option<Reg>> {
        self.with_args(SimdOp::Shuffle(order), 1)
    }

    /// `receiver.store(buffer, offset)` and the forms with a mask after
    /// the offset (`store_select`, `compress_store`).
    fn store(&mut self, op: fn(Buffer) -> SimdOp, masked: bool) -> crate::Result<Option<Reg>> {
        let (buffer, offset, this) = (
            self.buffer(0)?,
            self.scalar(1, ScalarType::Index)?,
            self.this()?,
        );
        if masked {
            let mask = self.value(2)?;
            self.op(op(buffer), &[offset, mask, this])
        } else {
            self.op(op(buffer), &[offset, this])
        }
    }

    /// `receiver.scatter(buffer, indices)` and
    /// `receiver.scatter_select(buffer, mask, indices)`.
    fn scatter(&mut self, op: fn(Buffer) -> SimdOp, masked: bool) -> crate::Result<Option<Reg>> {
        let (buffer, this) = (self.buffer(0)?, self.this()?);
        if masked {
            let (mask, indices) = (self.value(1)?, self.value(2)?);
            self.op(op(buffer), &[indices, mask, this])
        } else {
            let indices = self.value(1)?;
            self.op(op(buffer), &[indices, this])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `#[intrinsic(simd)]` methods of `splice::simd`, by owner: the
    /// declarations the kernel macro reads.
    fn declared() -> Vec<(bool, String)> {
        let file = syn::parse_file(include_str!("../../../splice-macros/src/simd_api.rs"))
            .expect("simd_api.rs parses");
        let mut methods = Vec::new();
        for item in file.items {
            let syn::Item::Impl(implementation) = item else {
                continue;
            };
            let syn::Type::Path(owner) = &*implementation.self_ty else {
                continue;
            };
            let mask = owner.path.segments.last().unwrap().ident == "Mask";
            for item in implementation.items {
                if let syn::ImplItem::Verbatim(tokens) = item {
                    let method: syn::TraitItemFn = syn::parse::Parser::parse2(
                        |input: syn::parse::ParseStream| {
                            input.call(syn::Attribute::parse_outer)?;
                            input.parse::<syn::Visibility>()?;
                            input.parse()
                        },
                        tokens,
                    )
                    .expect("a body-less method");
                    methods.push((mask, method.sig.ident.to_string()));
                }
            }
        }
        methods
    }

    #[test]
    fn every_declared_simd_intrinsic_has_one_lowering() {
        let declared = declared();
        assert!(declared.len() > 60, "{declared:?}");
        for (mask, method) in &declared {
            assert!(simd_lowering(*mask, method).is_some(), "{method}");
        }
        for (mask, table) in [(false, &LANE[..]), (true, &MASK[..])] {
            for (i, (name, _)) in table.iter().enumerate() {
                assert!(
                    declared.iter().any(|(m, d)| *m == mask && d == name),
                    "{name} is not declared"
                );
                assert!(i == 0 || table[i - 1].0 < *name, "{name} out of order");
            }
        }
        assert!(declared.contains(&(true, "select".into())));
        assert!(simd_lowering(false, "saturating_add").is_none());
    }
}
