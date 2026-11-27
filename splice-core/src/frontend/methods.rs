//! The Rust methods of runtime kernel values that are core operations:
//! `x.saturating_add(y)` on a runtime integer stages one portable
//! [`Binary`] operation, which every backend implements. These are not
//! intrinsics (see [`crate::intrinsic`]): no backend chooses them.
//!
//! The methods of [`Lane`](crate::simd::Lane) and [`Mask`](crate::simd::Mask)
//! are the SIMD part ([`simd_method`]): each one stages [`SimdOp`]s on the
//! operands the kernel passes, typed by the builder.
use super::KernelBuilder;
use crate::hir::{Binary, Buffer, Constant, Reg, ScalarType, Simd, SimdOp, ValueType};
use crate::simd::Shuffle;

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

/// The SIMD method being staged: its receiver, its element and its operands.
struct SimdCall<'a> {
    builder: &'a mut dyn KernelBuilder,
    method: &'a str,
    receiver: Reg,
    element: ScalarType,
    mask: bool,
    args: &'a [Operand],
    /// The output element of a conversion (`cast::<U>()`, `widen_low`).
    output: Option<ScalarType>,
}

type SimdLowering = fn(&mut SimdCall<'_>) -> crate::Result<Option<Reg>>;

/// The SIMD methods: the Rust method of `Lane` or `Mask` and the operations
/// it stages.
const SIMD_METHODS: [(&str, SimdLowering); 35] = [
    ("interleave_low", |c| c.shuffle(Shuffle::InterleaveLow)),
    ("interleave_high", |c| c.shuffle(Shuffle::InterleaveHigh)),
    ("deinterleave_even", |c| {
        c.shuffle(Shuffle::DeinterleaveEven)
    }),
    ("deinterleave_odd", |c| c.shuffle(Shuffle::DeinterleaveOdd)),
    ("compress", |c| c.with_args(SimdOp::Compress, 1)),
    ("expand", |c| c.with_args(SimdOp::Expand, 2)),
    ("swizzle", |c| c.with_args(SimdOp::Swizzle, 1)),
    ("rotate_left", |c| {
        let count = c.count(0)?;
        c.with_args(SimdOp::RotateLeft(count), 0)
    }),
    ("saturating_add", |c| c.with_args(SimdOp::SaturatingAdd, 1)),
    ("saturating_sub", |c| c.with_args(SimdOp::SaturatingSub, 1)),
    ("simd_min", |c| c.with_args(SimdOp::Min, 1)),
    ("simd_max", |c| c.with_args(SimdOp::Max, 1)),
    ("sqrt", |c| c.with_args(SimdOp::Sqrt, 0)),
    ("mul_add", |c| c.with_args(SimdOp::MulAdd, 2)),
    ("reduce_sum", |c| c.with_args(SimdOp::ReduceSum, 0)),
    ("prefix_sum", |c| c.with_args(SimdOp::PrefixSum, 0)),
    ("last", |c| c.with_args(SimdOp::Last, 0)),
    ("cast", |c| {
        let to = c.output()?;
        c.with_args(SimdOp::Cast(to), 0)
    }),
    ("widen_low", |c| {
        let to = c.output()?;
        c.with_args(SimdOp::Widen { to, high: false }, 0)
    }),
    ("widen_high", |c| {
        let to = c.output()?;
        c.with_args(SimdOp::Widen { to, high: true }, 0)
    }),
    ("narrow", |c| {
        let to = c.output()?;
        c.with_args(SimdOp::Narrow(to), 1)
    }),
    ("extract_wrapping", |c| {
        let index = c.lane(0)?;
        c.vector_op(SimdOp::ExtractWrapping, &[c.receiver, index])
    }),
    ("replace_wrapping", |c| {
        let (index, value) = (c.lane(0)?, c.scalar(1, c.element)?);
        c.vector_op(SimdOp::ReplaceWrapping, &[c.receiver, index, value])
    }),
    ("simd_eq", |c| c.with_args(SimdOp::Eq, 1)),
    ("simd_ne", |c| c.with_args(SimdOp::Ne, 1)),
    ("simd_lt", |c| c.with_args(SimdOp::Lt, 1)),
    ("simd_le", |c| c.with_args(SimdOp::Le, 1)),
    ("simd_gt", |c| c.with_args(SimdOp::Gt, 1)),
    ("simd_ge", |c| c.with_args(SimdOp::Ge, 1)),
    ("add_masked", |c| {
        let (mask, rhs, fallback) = (c.value(0)?, c.value(1)?, c.value(2)?);
        let sum = c.vector_op(SimdOp::Add, &[c.receiver, rhs])?;
        let sum = sum.ok_or_else(|| crate::BuildError::staging("missing SIMD result"))?;
        c.vector_op(SimdOp::Select, &[mask, sum, fallback])
    }),
    ("store", |c| c.store(SimdOp::Store, false)),
    ("store_select", |c| c.store(SimdOp::StoreSelect, true)),
    ("compress_store", |c| c.store(SimdOp::CompressStore, true)),
    ("scatter", |c| c.scatter(SimdOp::Scatter, false)),
    ("scatter_select", |c| c.scatter(SimdOp::ScatterSelect, true)),
];

/// The methods of a mask receiver.
const MASK_METHODS: [(&str, SimdLowering); 4] = [
    ("any", |c| c.mask_op(SimdOp::Any)),
    ("all", |c| c.mask_op(SimdOp::All)),
    ("count", |c| c.mask_op(SimdOp::Count)),
    ("select", |c| {
        let (yes, no) = (c.value(0)?, c.value(1)?);
        // The data operands give the element of the selection.
        let Some(ValueType::Vector(element)) = c.builder.reg_type(yes) else {
            return Err(crate::BuildError::staging(
                "mask.select requires two vectors",
            ));
        };
        c.require_mask()?;
        let simd = Simd {
            element,
            op: SimdOp::Select,
        };
        c.builder.simd(simd, &[c.receiver, yes, no])
    }),
];

/// The lowering of the SIMD method `method`, if registered.
fn simd_lowering(method: &str) -> Option<SimdLowering> {
    SIMD_METHODS
        .iter()
        .chain(&MASK_METHODS)
        .find(|(name, _)| *name == method)
        .map(|&(_, lowering)| lowering)
}

/// Stage the SIMD method `method` on the vector or mask `receiver` with
/// `args` (the method's arguments in order); `output` is the element a
/// conversion produces. The result is `None` for a method that only
/// writes memory.
#[doc(hidden)]
pub fn simd_method(
    builder: &mut dyn KernelBuilder,
    method: &str,
    receiver: Reg,
    args: &[Operand],
    output: Option<ScalarType>,
) -> crate::Result<Option<Reg>> {
    let (element, mask) = match builder.reg_type(receiver) {
        Some(ValueType::Vector(element)) => (element, false),
        Some(ValueType::Mask(element)) => (element, true),
        _ => {
            return Err(crate::BuildError::staging(format!(
                "`{method}` requires a runtime vector or mask receiver"
            )));
        }
    };
    let lowering = simd_lowering(method).ok_or_else(|| {
        crate::BuildError::staging(format!("`{method}` is not a kernel SIMD operation"))
    })?;
    lowering(&mut SimdCall {
        builder,
        method,
        receiver,
        element,
        mask,
        args,
        output,
    })
}

impl SimdCall<'_> {
    fn error(&self, message: &str) -> crate::BuildError {
        crate::BuildError::staging(format!("`{}`: {message}", self.method))
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
                    .ok_or_else(|| self.error(&format!("expected a `{ty}` operand")))?;
                self.builder.constant(value)
            }
            Operand::Buffer(_) => Err(self.error("expected a scalar operand")),
        }
    }

    /// Argument `index`, a lane index (`i32`).
    fn lane(&mut self, index: usize) -> crate::Result<Reg> {
        self.scalar(index, ScalarType::I32)
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

    fn output(&self) -> crate::Result<ScalarType> {
        self.output
            .ok_or_else(|| self.error("specify the output lane type"))
    }

    fn require_mask(&self) -> crate::Result<()> {
        if self.mask {
            Ok(())
        } else {
            Err(self.error("requires a mask receiver"))
        }
    }

    /// `op` on the receiver's element; the receiver is a vector.
    fn vector_op(&mut self, op: SimdOp, operands: &[Reg]) -> crate::Result<Option<Reg>> {
        if self.mask {
            return Err(self.error("requires a vector receiver"));
        }
        let simd = Simd {
            element: self.element,
            op,
        };
        self.builder.simd(simd, operands)
    }

    /// `op` on the receiver, a mask.
    fn mask_op(&mut self, op: SimdOp) -> crate::Result<Option<Reg>> {
        self.require_mask()?;
        let simd = Simd {
            element: self.element,
            op,
        };
        self.builder.simd(simd, &[self.receiver])
    }

    /// `op` on the receiver and its first `count` arguments, all vectors or masks.
    fn with_args(&mut self, op: SimdOp, count: usize) -> crate::Result<Option<Reg>> {
        let mut operands = vec![self.receiver];
        for index in 0..count {
            operands.push(self.value(index)?);
        }
        self.vector_op(op, &operands)
    }

    fn shuffle(&mut self, order: Shuffle) -> crate::Result<Option<Reg>> {
        self.with_args(SimdOp::Shuffle(order), 1)
    }

    /// `receiver.store(buffer, offset)` and the forms with a mask after
    /// the offset (`store_select`, `compress_store`).
    fn store(&mut self, op: fn(Buffer) -> SimdOp, masked: bool) -> crate::Result<Option<Reg>> {
        let buffer = self.buffer(0)?;
        let mut operands = vec![self.scalar(1, ScalarType::Index)?];
        if masked {
            operands.push(self.value(2)?);
        }
        operands.push(self.receiver);
        self.vector_op(op(buffer), &operands)
    }

    /// `receiver.scatter(buffer, indices)` and
    /// `receiver.scatter_select(buffer, mask, indices)`.
    fn scatter(&mut self, op: fn(Buffer) -> SimdOp, masked: bool) -> crate::Result<Option<Reg>> {
        let buffer = self.buffer(0)?;
        let operands = if masked {
            vec![self.value(2)?, self.value(1)?, self.receiver]
        } else {
            vec![self.value(1)?, self.receiver]
        };
        self.vector_op(op(buffer), &operands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_simd_method_has_one_lowering() {
        let names: Vec<_> = SIMD_METHODS
            .iter()
            .chain(&MASK_METHODS)
            .map(|(name, _)| *name)
            .collect();
        for (i, name) in names.iter().enumerate() {
            assert!(simd_lowering(name).is_some(), "{name}");
            assert!(!names[..i].contains(name), "{name} is registered twice");
        }
        assert!(simd_lowering("saturating_mul").is_none());
    }
}
