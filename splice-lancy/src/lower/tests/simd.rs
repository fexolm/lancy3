use super::*;
use lancy::hir::Simd as V;
use splice_core::{
    BufferArg, Executable,
    hir::{Simd, SimdOp},
};

fn both(
    arity: usize,
    kinds: &[BufferKind],
    stage: fn(&mut dyn KernelBuilder) -> splice_core::Result<()>,
) -> [hir::Function; 2] {
    let ir = splice_core::frontend::materialize(arity, kinds, Box::new(stage), options()).unwrap();
    [
        build(arity, kinds, Box::new(stage)).unwrap(),
        lower(&ir, &options().target).unwrap(),
    ]
}

fn u16_op(op: SimdOp) -> Simd {
    Simd {
        element: Element::U16,
        op,
    }
}

#[test]
fn signed_index_gathers_check_each_width_with_one_unsigned_comparison() {
    let kinds = [
        BufferKind::new(Element::I32, false),
        BufferKind::new(Element::I64, false),
    ];
    for graph in both(0, &kinds, |b| {
        for (element, buffer) in [(Element::I32, Buffer(0)), (Element::I64, Buffer(1))] {
            let simd = |op| Simd { element, op };
            let ids = b.simd(simd(SimdOp::Iota), &[])?.unwrap();
            b.simd(simd(SimdOp::Gather(buffer)), &[ids])?;
        }
        let zero = b.constant(0)?;
        b.ret(zero)?;
        Ok(())
    }) {
        let comparisons: Vec<_> = graph
            .blocks()
            .flat_map(|b| graph.instructions(b))
            .filter_map(|inst| match *graph.op(inst) {
                Op::Simd(V::Bin(op @ (hir::Binary::Lt | hir::Binary::Ge), left, _)) => {
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
        BufferKind::new(Element::U16, false),
        BufferKind::new(Element::U16, true),
    ];
    for graph in both(1, &kinds, |b| {
        let zero = b.constant(0)?;
        let a = b.simd(u16_op(SimdOp::Load(Buffer(0))), &[zero])?.unwrap();
        let one = b.simd(u16_op(SimdOp::Splat(1)), &[])?.unwrap();
        let sum = b.simd(u16_op(SimdOp::Add), &[a, one])?.unwrap();
        b.simd(u16_op(SimdOp::Store(Buffer(1))), &[b.arg(0)?, sum])?;
        b.ret(zero)?;
        Ok(())
    }) {
        let ops: Vec<_> = graph
            .blocks()
            .flat_map(|block| graph.instructions(block))
            .map(|inst| graph.op(inst))
            .collect();
        assert!(ops.iter().any(|op| matches!(op, Op::Simd(V::Load(..)))));
        assert!(ops.iter().any(|op| matches!(op, Op::Simd(V::Store(..)))));
        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::Simd(V::MaskedStore { .. })))
        );
        let kernel = crate::LancyKernel::new(graph, &options().target, 1, &kinds, None).unwrap();
        let lanes = Element::U16.lanes(options().target.vector_bytes());
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
                    BufferArg::ReadU16(&src),
                    BufferArg::WriteU16(&mut dst[..len]),
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
        BufferKind::new(Element::U16, false),
        BufferKind::new(Element::U16, true),
    ];
    for graph in both(0, &kinds, |b| {
        let zero = b.constant(0)?;
        let ids = b.simd(u16_op(SimdOp::Load(Buffer(0))), &[zero])?.unwrap();
        let values = b.simd(u16_op(SimdOp::Iota), &[])?.unwrap();
        b.simd(u16_op(SimdOp::Scatter(Buffer(1))), &[ids, values])?;
        b.ret(zero)?;
        Ok(())
    }) {
        let kernel = crate::LancyKernel::new(graph, &options().target, 0, &kinds, None).unwrap();
        let mut indices = vec![0; Element::U16.lanes(options().target.vector_bytes())];
        indices[..4].copy_from_slice(&[32768, 32768, 65535, 1]);
        for n in [65535, 65536] {
            let mut dst = vec![99; n];
            let result = kernel.call(
                &[],
                &mut [BufferArg::ReadU16(&indices), BufferArg::WriteU16(&mut dst)],
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
    let kinds = [BufferKind::new(Element::U16, true)];
    for graph in both(1, &kinds, |b| {
        let i = b.simd(u16_op(SimdOp::Iota), &[])?.unwrap();
        let one = b.simd(u16_op(SimdOp::Splat(1)), &[])?.unwrap();
        let enabled = b.simd(u16_op(SimdOp::Eq), &[i, one])?.unwrap();
        b.simd(
            u16_op(SimdOp::StoreSelect(Buffer(0))),
            &[b.arg(0)?, enabled, i],
        )?;
        b.ret(b.arg(0))?;
        Ok(())
    }) {
        let kernel = crate::LancyKernel::new(graph, &options().target, 1, &kinds, None).unwrap();
        let mut dst = [8, 9];
        let result = kernel.call(&[-1], &mut [BufferArg::WriteU16(&mut dst)]);
        assert_eq!(
            result.unwrap_err(),
            splice_core::CallError::Trap(splice_core::Trap::Bounds)
        );
        assert_eq!(dst, [8, 9]);
    }
}

#[test]
fn vectors_and_masks_keep_their_element_type_across_a_join() {
    let kinds = [BufferKind::new(Element::F64, true)];
    for graph in both(1, &kinds, |b| {
        let inst = |op| Simd {
            element: Element::F64,
            op,
        };
        let zero = b.constant(0)?;
        let v = b.local(ValueType::Vector(Element::F64))?;
        let mask = b.local(ValueType::Mask(Element::F64))?;
        let (left, right, join) = (b.label()?, b.label()?, b.label()?);
        b.branch(b.arg(0), left, right)?;
        for (block, lit, cmp) in [(left, 1.25_f64, SimdOp::Eq), (right, -2.0, SimdOp::Ne)] {
            b.mark(block)?;
            let n = b.simd(inst(SimdOp::Splat(lit.to_bits())), &[])?.unwrap();
            let m = b.simd(inst(cmp), &[n, n])?.unwrap();
            b.assign(v, n)?;
            b.assign(mask, m)?;
            b.jump(join)?;
        }
        b.mark(join)?;
        b.simd(inst(SimdOp::StoreSelect(Buffer(0))), &[zero, mask, v])?;
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
        let kernel = crate::LancyKernel::new(graph, &options().target, 1, &kinds, None).unwrap();
        for (input, expected) in [(1, 1.25), (0, 3.0)] {
            let mut dst = vec![3.0; Element::F64.lanes(options().target.vector_bytes())];
            assert_eq!(
                kernel.call(&[input], &mut [BufferArg::WriteF64(&mut dst)]),
                Ok(0)
            );
            assert!(dst.iter().all(|&x| x == expected));
        }
    }
}
