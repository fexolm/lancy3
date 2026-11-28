//! CORE-10: `SimdOp::info` is the one description of every SIMD operation.
//! For every operation and element, the verifier accepts exactly what the
//! table accepts, the result has the table's type, and every backend
//! compiles and runs each accepted operation with the same effect.
use splice::frontend::{self, KernelBuilder, Signature};
use splice::hir::{Access, Buffer, Constant, ScalarType, Simd, SimdOp, Slot, ValueType};
use splice::std::intrinsics::simd::{Element, Shuffle as Pairing};
use splice::{BufferArg, BufferElement, BufferKind, BuildError, BuildOptions, Codegen, Mode};
use std::fmt::Debug;

const ELEMENTS: [Element; 6] = [
    Element::U16,
    Element::U32,
    Element::I32,
    Element::I64,
    Element::F32,
    Element::F64,
];

/// Every operation, with each value of its parameters that changes its
/// operands, result or elements.
fn operations() -> Vec<SimdOp> {
    use SimdOp::*;
    let b = Buffer(0);
    let mut ops = vec![
        ElementWidth,
        ByteWidth,
        FirstN,
        Splat(1),
        SplatScalar,
        Iota,
        PrefixSum,
        AlignmentPrefix(b),
        Compress,
        Expand,
        Swizzle,
        Count,
        Shr(3),
        RotateLeft(5),
        CompressStore(b),
        Last,
        ExtractWrapping,
        ReplaceWrapping,
        Load(b),
        Store(b),
        LoadSelect(b),
        StoreSelect(b),
        Gather(b),
        Scatter(b),
        GatherSelect(b),
        ScatterSelect(b),
        Add,
        AddMasked,
        Sub,
        Mul,
        Div,
        Sqrt,
        MulAdd,
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
    ];
    ops.extend(
        [
            Pairing::InterleaveLow,
            Pairing::InterleaveHigh,
            Pairing::DeinterleaveEven,
            Pairing::DeinterleaveOdd,
        ]
        .map(SimdOp::Shuffle),
    );
    for to in ELEMENTS {
        ops.extend([Cast(to), Narrow(to)]);
        ops.extend([false, true].map(|high| Widen { to, high }));
    }
    ops
}

/// Buffer 0 holds `e` (the operation's buffer), buffer 1 its lane indices.
fn kinds(e: Element) -> [BufferKind; 2] {
    [BufferKind::new(e, true), BufferKind::new(e.index(), false)]
}

/// Stage `inst` on operands of the table's types: vectors and indices
/// loaded at 0, masks of every lane, scalars 0.
fn stage(inst: Simd, b: &mut dyn KernelBuilder) -> splice::Result<Option<ValueType>> {
    let e = inst.element;
    let zero = b.constant(Constant::from(0usize))?;
    let load = |b: &mut dyn KernelBuilder, element, buffer| {
        let op = SimdOp::Load(Buffer(buffer));
        Ok::<_, BuildError>(b.simd(Simd { element, op }, &[zero])?.unwrap())
    };
    let mut args = Vec::new();
    for slot in inst.op.info().operands {
        args.push(match slot {
            Slot::Vector => load(b, e, 0)?,
            Slot::Indices => load(b, e.index(), 1)?,
            Slot::Mask => {
                let all = b.constant(Constant::from(usize::MAX))?;
                let op = SimdOp::FirstN;
                b.simd(Simd { element: e, op }, &[all])?.unwrap()
            }
            Slot::Lane => b.constant(Constant::from(0))?,
            Slot::Offset => zero,
            Slot::Bool | Slot::Converted(_) => unreachable!("not an operand"),
        });
    }
    let output = b.simd(inst, &args)?.map(|r| b.reg_type(r).unwrap());
    let done = b.constant(Constant::from(0))?;
    b.ret(done)?;
    Ok(output)
}

fn layout(e: Element) -> (Signature<'static>, BuildOptions) {
    let kinds: &'static [BufferKind; 2] = Box::leak(Box::new(kinds(e)));
    let options = BuildOptions::host(Mode::Release).unwrap();
    (Signature::new(&[], ScalarType::I32, kinds), options)
}

/// Run `kernel` on buffers of `T` (`1, 2, …`) and zero indices of `I`;
/// report the result and buffer 0 after the call.
fn observe<T: BufferElement + From<u8> + Debug, I: BufferElement + Default>(
    call: &dyn Fn(&mut [BufferArg<'_>]) -> Result<u64, splice::CallError>,
) -> String {
    let mut data: Vec<T> = (1..=64).map(T::from).collect();
    let indices = vec![I::default(); 64];
    let result = call(&mut [data.as_mut_slice().into(), indices.as_slice().into()]);
    format!("{result:?} {data:?}")
}

fn run(
    e: Element,
    call: &dyn Fn(&mut [BufferArg<'_>]) -> Result<u64, splice::CallError>,
) -> String {
    match e {
        Element::U16 => observe::<u16, u16>(call),
        Element::U32 => observe::<u32, u32>(call),
        Element::I32 => observe::<i32, i32>(call),
        Element::I64 => observe::<i64, i64>(call),
        Element::F32 => observe::<f32, i32>(call),
        Element::F64 => observe::<f64, i64>(call),
    }
}

#[test]
fn every_simd_operation_is_checked_and_lowered_as_its_info_says() {
    let mut accepted = 0;
    for op in operations() {
        let info = op.info();
        assert_eq!(
            op.may_trap(),
            info.memory.is_some_and(|m| m.access != Access::Address),
            "{op:?}"
        );
        for e in ELEMENTS {
            let inst = Simd { element: e, op };
            let (signature, options) = layout(e);
            let mut output = None;
            let staged = frontend::materialize(
                signature,
                Box::new(|b| {
                    output = stage(inst, b)?;
                    Ok(())
                }),
                options,
            );
            if !info.elements.accepts(e) {
                assert!(
                    matches!(staged, Err(BuildError::Verify { .. })),
                    "{inst:?} is outside {:?} but staged as {staged:?}",
                    info.elements
                );
                fn rejects<B: Codegen>(inst: Simd, e: Element) -> bool {
                    let (signature, options) = layout(e);
                    let body = Box::new(move |b: &mut dyn KernelBuilder| stage(inst, b).map(drop));
                    matches!(
                        B::build_with_options(signature, body, options),
                        Err(BuildError::Verify { .. })
                    )
                }
                splice_testkit::each_backend!(B => assert!(rejects::<B>(inst, e), "{inst:?}"));
                continue;
            }
            accepted += 1;
            let function = staged.unwrap_or_else(|error| panic!("{inst:?}: {error}"));
            assert_eq!(output, inst.output(), "{inst:?}");
            assert_eq!(
                inst.output(),
                info.output.map(|slot| slot.value_type(e)),
                "{inst:?}"
            );
            let mut seen = Vec::new();
            splice_testkit::each_backend!(B => {
                let kernel = B::compile(&function).unwrap_or_else(|error| {
                    panic!("{} rejects {inst:?}: {error}", std::any::type_name::<B>())
                });
                let call = |buffers: &mut [BufferArg<'_>]| splice::Executable::call(&kernel, &[], buffers);
                seen.push((std::any::type_name::<B>(), run(e, &call)));
            });
            for (backend, result) in &seen[1..] {
                assert_eq!(
                    result, &seen[0].1,
                    "{inst:?}: {backend} differs from {}",
                    seen[0].0
                );
            }
        }
    }
    assert!(accepted > 150, "{accepted}");
}
