use super::*;
use lancy::hir::Simd as V;
use splice_core::hir::{Constant, ScalarType};
use splice_core::{
    BufferArg, Executable,
    hir::{Simd, SimdOp},
};

fn both(
    arity: usize,
    kinds: &[BufferKind],
    stage: fn(&mut dyn KernelBuilder) -> splice_core::Result<()>,
) -> [hir::Function; 2] {
    let ir = splice_core::frontend::materialize(
        Signature::new(&vec![ScalarType::I32; arity], ScalarType::I32, kinds),
        Box::new(stage),
        options(),
    )
    .unwrap();
    [
        build(
            Signature::new(&vec![ScalarType::I32; arity], ScalarType::I32, kinds),
            Box::new(stage),
        )
        .unwrap(),
        lower(&ir, &options().target).unwrap(),
    ]
}

fn u16_op(op: SimdOp) -> Simd {
    Simd {
        element: ScalarType::U16,
        op,
    }
}

#[test]
fn signed_index_gathers_check_each_width_with_one_unsigned_comparison() {
    let kinds = [
        BufferKind::new(ScalarType::I32, false),
        BufferKind::new(ScalarType::I64, false),
    ];
    for graph in both(0, &kinds, |b| {
        for (element, buffer) in [(ScalarType::I32, Buffer(0)), (ScalarType::I64, Buffer(1))] {
            let simd = |op| Simd { element, op };
            let ids = b.simd(simd(SimdOp::Iota), &[])?.unwrap();
            b.simd(simd(SimdOp::Gather(buffer)), &[ids])?;
        }
        let zero = b.constant(Constant::from(0))?;
        b.ret(zero)?;
        Ok(())
    }) {
        let comparisons: Vec<_> = graph
            .blocks()
            .flat_map(|b| graph.instructions(b))
            .filter_map(|inst| match *graph.op(inst) {
                Op::Bin(op @ (hir::Binary::Lt | hir::Binary::Ge), left, _)
                    if matches!(graph.value_type(left), hir::Type::Vector(_)) =>
                {
                    Some((op, graph.value_type(left)))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            comparisons,
            [
                (hir::Binary::Ge, hir::Type::Vector(hir::Lane::U32)),
                (hir::Binary::Ge, hir::Type::Vector(hir::Lane::U64)),
            ]
        );
    }
}

#[test]
fn packed_load_and_store_have_an_ordered_short_store_exit() {
    let kinds = [
        BufferKind::new(ScalarType::U16, false),
        BufferKind::new(ScalarType::U16, true),
    ];
    for graph in both(1, &kinds, |b| {
        let first = b.constant(Constant::from(0usize))?;
        let a = b.simd(u16_op(SimdOp::Load(Buffer(0))), &[first])?.unwrap();
        let one = b.simd(u16_op(SimdOp::Splat(1)), &[])?.unwrap();
        let sum = b.simd(u16_op(SimdOp::Add), &[a, one])?.unwrap();
        let at = b.cast(b.arg(0)?, ScalarType::Index)?;
        b.simd(u16_op(SimdOp::Store(Buffer(1))), &[at, sum])?;
        let zero = b.constant(Constant::from(0))?;
        b.ret(zero)?;
        Ok(())
    }) {
        let ops: Vec<_> = graph
            .blocks()
            .flat_map(|block| graph.instructions(block))
            .map(|inst| graph.op(inst))
            .collect();
        let vector = |v| matches!(graph.value_type(v), hir::Type::Vector(_));
        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::Load(hir::Type::Vector(_), _)))
        );
        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::Store(_, v) if vector(*v)))
        );
        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::Simd(V::MaskedStore { .. })))
        );
        let kernel = crate::LancyKernel::compile(
            Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &kinds),
            splice_core::BuildOptions::host(splice_core::Mode::Release).unwrap(),
            |_, _| Ok((graph, None)),
        )
        .unwrap();
        let lanes = ScalarType::U16.lanes(options().target.vector_bytes());
        assert_eq!(
            lanes * 2,
            splice_target::Target::host().unwrap().vector_bytes()
        );
        let mut src: Vec<_> = (0..lanes as u16).collect();
        src[lanes - 1] = u16::MAX;
        for len in [lanes + 1, lanes, 2, 1, 0] {
            let mut dst = vec![99; lanes + 2];
            let result = kernel.call(
                &[1],
                &mut [
                    BufferArg::read::<u16>(&src),
                    BufferArg::write::<u16>(&mut dst[..len]),
                ],
            );
            assert_eq!(result.is_ok(), len > lanes, "len={len}");
            let mut expected = vec![99; lanes + 2];
            for (out, x) in expected[1..len.max(1)].iter_mut().zip(&src) {
                *out = x.wrapping_add(1);
            }
            assert_eq!(dst, expected);
        }
    }
}

#[test]
fn scatter_addresses_unsigned_u16_and_stops_at_the_first_active_error() {
    let kinds = [
        BufferKind::new(ScalarType::U16, false),
        BufferKind::new(ScalarType::U16, true),
    ];
    for graph in both(0, &kinds, |b| {
        let first = b.constant(Constant::from(0usize))?;
        let ids = b.simd(u16_op(SimdOp::Load(Buffer(0))), &[first])?.unwrap();
        let zero = b.constant(Constant::from(0))?;
        let values = b.simd(u16_op(SimdOp::Iota), &[])?.unwrap();
        b.simd(u16_op(SimdOp::Scatter(Buffer(1))), &[ids, values])?;
        b.ret(zero)?;
        Ok(())
    }) {
        let kernel = crate::LancyKernel::compile(
            Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &kinds),
            splice_core::BuildOptions::host(splice_core::Mode::Release).unwrap(),
            |_, _| Ok((graph, None)),
        )
        .unwrap();
        let mut indices = vec![0; ScalarType::U16.lanes(options().target.vector_bytes())];
        indices[..4].copy_from_slice(&[32768, 32768, 65535, 1]);
        for n in [65535, 65536] {
            let mut dst = vec![99; n];
            let result = kernel.call(
                &[],
                &mut [
                    BufferArg::read::<u16>(&indices),
                    BufferArg::write::<u16>(&mut dst),
                ],
            );
            let mut expected = vec![99; n];
            let ok = indices.iter().enumerate().all(|(i, &idx)| {
                if let Some(v) = expected.get_mut(usize::from(idx)) {
                    *v = i as u16;
                    true
                } else {
                    false
                }
            });
            assert_eq!(result.is_ok(), ok);
            assert_eq!(dst, expected);
        }
    }
}

#[test]
fn masked_contiguous_store_never_wraps_an_invalid_offset_to_zero() {
    let kinds = [BufferKind::new(ScalarType::U16, true)];
    for graph in both(1, &kinds, |b| {
        let i = b.simd(u16_op(SimdOp::Iota), &[])?.unwrap();
        let one = b.simd(u16_op(SimdOp::Splat(1)), &[])?.unwrap();
        let enabled = b.simd(u16_op(SimdOp::Eq), &[i, one])?.unwrap();
        let at = b.cast(b.arg(0)?, ScalarType::Index)?;
        b.simd(u16_op(SimdOp::StoreSelect(Buffer(0))), &[at, enabled, i])?;
        b.ret(b.arg(0)?)?;
        Ok(())
    }) {
        let kernel = crate::LancyKernel::compile(
            Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &kinds),
            splice_core::BuildOptions::host(splice_core::Mode::Release).unwrap(),
            |_, _| Ok((graph, None)),
        )
        .unwrap();
        let mut dst = [8, 9];
        let result = kernel.call(
            &[splice_core::KernelScalar::to_word(-1i32)],
            &mut [BufferArg::write::<u16>(&mut dst)],
        );
        assert_eq!(
            result.unwrap_err(),
            splice_core::CallError::Trap(splice_core::Trap::Bounds)
        );
        assert_eq!(dst, [8, 9]);
    }
}

#[test]
fn vectors_and_masks_keep_their_element_type_across_a_join() {
    let kinds = [BufferKind::new(ScalarType::F64, true)];
    for graph in both(1, &kinds, |b| {
        let inst = |op| Simd {
            element: ScalarType::F64,
            op,
        };
        let zero = b.constant(Constant::from(0))?;
        let first = b.constant(Constant::from(0usize))?;
        let v = b.local(ValueType::Vector(ScalarType::F64))?;
        let mask = b.local(ValueType::Mask(ScalarType::F64))?;
        let (left, right, join) = (b.label()?, b.label()?, b.label()?);
        let taken = b.binary(splice_core::hir::Binary::Ne, b.arg(0)?, zero)?;
        b.branch(taken, left, right)?;
        for (block, lit, cmp) in [(left, 1.25_f64, SimdOp::Eq), (right, -2.0, SimdOp::Ne)] {
            b.mark(block)?;
            let n = b.simd(inst(SimdOp::Splat(lit.to_bits())), &[])?.unwrap();
            let m = b.simd(inst(cmp), &[n, n])?.unwrap();
            b.assign(v, n)?;
            b.assign(mask, m)?;
            b.jump(join)?;
        }
        b.mark(join)?;
        b.simd(inst(SimdOp::StoreSelect(Buffer(0))), &[first, mask, v])?;
        b.ret(zero)?;
        Ok(())
    }) {
        let params: Vec<_> = graph
            .blocks()
            .flat_map(|b| graph.params(b))
            .map(|&v| graph.value_type(v))
            .collect();
        assert!(params.contains(&hir::Type::Vector(hir::Lane::F64)));
        assert!(params.contains(&hir::Type::Vector(hir::Lane::I64)));
        let kernel = crate::LancyKernel::compile(
            Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &kinds),
            splice_core::BuildOptions::host(splice_core::Mode::Release).unwrap(),
            |_, _| Ok((graph, None)),
        )
        .unwrap();
        for (input, expected) in [(1, 1.25), (0, 3.0)] {
            let mut dst = vec![3.0; ScalarType::F64.lanes(options().target.vector_bytes())];
            assert_eq!(
                kernel.call(&[input], &mut [BufferArg::write::<f64>(&mut dst)]),
                Ok(0)
            );
            assert!(dst.iter().all(|&x| x == expected));
        }
    }
}
