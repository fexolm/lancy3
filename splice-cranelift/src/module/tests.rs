use super::*;
use crate::Cranelift;
use cranelift_codegen::ir::{Opcode, Type};
use cranelift_jit::JITBuilder;
use cranelift_module::default_libcall_names;
use splice_core::frontend::{KernelBuilder, Signature};
use splice_core::hir::{Constant, ScalarType};
use splice_core::{
    CallError, Codegen, Executable, KernelScalar, Trap,
    hir::{Binary, FunctionBuilder},
};

/// This machine's `SPLICE_SIMD` target in the mode of this test build.
fn options() -> splice_core::BuildOptions {
    splice_core::BuildOptions::host(splice_core::Mode::for_profile(cfg!(debug_assertions))).unwrap()
}

fn increment() -> Function {
    let mut f = FunctionBuilder::new(
        options(),
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
    );
    let one = f.constant(Constant::from(1)).unwrap();
    let result = f.binary(Binary::Add, f.arg(0).unwrap(), one).unwrap();
    f.ret(result).unwrap();
    f.finish().unwrap()
}

fn caller(child: Function, times: usize) -> Function {
    let child = Arc::new(child);
    let mut f = FunctionBuilder::new(
        options(),
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
    );
    let mut value = f.arg(0).unwrap();
    for _ in 0..times {
        value = f.call_function(&child, &[value], &[]).unwrap();
    }
    f.ret(value).unwrap();
    f.finish().unwrap()
}

fn emitted_graph(f: &Function, unique_bodies: usize) -> HashMap<FuncId, Vec<FuncId>> {
    let mut jit = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let kernel = KernelModule::lower(&mut jit, f, &options().target, false).unwrap();
    assert_eq!(kernel.bodies.len(), unique_bodies);
    let mut graph = HashMap::new();
    kernel
        .emit(&mut jit.make_context(), false, |id, ctx| {
            ctx.verify(jit.isa()).unwrap();
            let calls = local_callees(&ctx.func).collect();
            assert!(graph.insert(id, calls).is_none());
            Ok(())
        })
        .unwrap();
    graph
}

#[test]
fn rotate_left_with_u32_count_verifies_for_64_bit_operands() {
    use splice_core::hir::{Intrinsic, MathOp};
    for ty in [ScalarType::I64, ScalarType::U64] {
        let mut builder =
            FunctionBuilder::new(options(), Signature::new(&[ty, ScalarType::U32], ty, &[]));
        let args = [builder.arg(0).unwrap(), builder.arg(1).unwrap()];
        let rotated = builder
            .call_intrinsic(
                Intrinsic::Math {
                    op: MathOp::RotateLeft,
                    ty,
                },
                &args,
            )
            .unwrap()
            .unwrap();
        builder.ret(rotated).unwrap();
        emitted_graph(&builder.finish().unwrap(), 1);
    }
}

/// Instructions outside cold blocks: the path that does not trap.
fn instructions_of_type(f: &ir::Function, opcode: Opcode, ty: Type) -> usize {
    f.layout
        .blocks()
        .filter(|&b| !f.layout.is_cold(b))
        .flat_map(|b| f.layout.block_insts(b))
        .filter(|&i| f.dfg.insts[i].opcode() == opcode && f.dfg.ctrl_typevar(i) == ty)
        .count()
}

#[test]
fn branches_use_comparison_bits_and_preserve_comparison_masks() {
    for overwrite in [false, true] {
        let mut f = FunctionBuilder::new(
            options(),
            Signature::new(&[ScalarType::I32; 2], ScalarType::I32, &[]),
        );
        let mask = f
            .binary(Binary::Lt, f.arg(0).unwrap(), f.arg(1).unwrap())
            .unwrap();
        if overwrite {
            let zero = f.constant(Constant::from(0)).unwrap();
            let nonzero = f.binary(Binary::Ne, f.arg(0).unwrap(), zero).unwrap();
            f.assign(mask, nonzero).unwrap();
        }
        let yes = f.label().unwrap();
        let no = f.label().unwrap();
        f.branch(mask, yes, no).unwrap();
        f.mark(yes).unwrap();
        let word = f.cast(mask, ScalarType::I32).unwrap();
        f.ret(word).unwrap();
        f.mark(no).unwrap();
        let fallback = f.constant(Constant::from(7)).unwrap();
        f.ret(fallback).unwrap();
        let f = f.finish().unwrap();

        let mut jit = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
        let module = KernelModule::lower(&mut jit, &f, &options().target, false).unwrap();
        let body = &module.bodies[&module.entry];
        let branch = body
            .layout
            .blocks()
            .flat_map(|b| body.layout.block_insts(b))
            .find(|&i| body.dfg.insts[i].opcode() == Opcode::Brif)
            .unwrap();
        let condition = body.dfg.inst_args(branch)[0];
        if !overwrite {
            let cmp = body.dfg.value_def(condition).unwrap_inst();
            assert_eq!(body.dfg.insts[cmp].opcode(), Opcode::Icmp);
        }

        let code = Cranelift::compile(&f).unwrap();
        assert_eq!(
            code.call(&[0, 1], &mut []),
            Ok(if overwrite { 7 } else { 1 })
        );
        assert_eq!(
            code.call(&[4, 2], &mut []),
            Ok(if overwrite { 1 } else { 7 })
        );
    }
}

#[test]
fn branches_use_vector_reduction_bits_and_preserve_scalar_masks() {
    use splice_core::{
        hir::ScalarType,
        hir::{Simd, SimdOp},
    };

    let mut f = FunctionBuilder::new(
        options(),
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
    );
    let op = |op| Simd {
        element: ScalarType::I32,
        op,
    };
    let count = f.cast(f.arg(0).unwrap(), ScalarType::Index).unwrap();
    let vector_mask = f.simd(op(SimdOp::FirstN), &[count]).unwrap().unwrap();
    let mask = f.simd(op(SimdOp::Any), &[vector_mask]).unwrap().unwrap();
    let yes = f.label().unwrap();
    let no = f.label().unwrap();
    f.branch(mask, yes, no).unwrap();
    f.mark(yes).unwrap();
    let word = f.cast(mask, ScalarType::I32).unwrap();
    f.ret(word).unwrap();
    f.mark(no).unwrap();
    let fallback = f.constant(Constant::from(7)).unwrap();
    f.ret(fallback).unwrap();
    let f = f.finish().unwrap();

    let mut jit = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let module = KernelModule::lower(&mut jit, &f, &options().target, false).unwrap();
    let body = &module.bodies[&module.entry];
    let branch = body
        .layout
        .blocks()
        .flat_map(|b| body.layout.block_insts(b))
        .find(|&i| body.dfg.insts[i].opcode() == Opcode::Brif)
        .unwrap();
    let condition = body.dfg.inst_args(branch)[0];
    let reduction = body.dfg.value_def(condition).unwrap_inst();
    assert!(matches!(
        body.dfg.insts[reduction].opcode(),
        Opcode::VanyTrue | Opcode::Bor
    ));

    let code = Cranelift::compile(&f).unwrap();
    assert_eq!(code.call(&[0], &mut []), Ok(7));
    assert_eq!(code.call(&[1], &mut []), Ok(1));
}

#[test]
fn nested_and_shared_calls_inline_across_function_boundaries() {
    let f = caller(caller(increment(), 2), 2);
    let graph = emitted_graph(&f, 3);
    assert_eq!(graph.len(), 1);
    assert!(graph.values().all(Vec::is_empty));
    assert_eq!(
        Cranelift::compile(&f)
            .unwrap()
            .call(&[i32::MAX.to_word()], &mut []),
        Ok((i32::MIN + 3).to_word())
    );
}

#[test]
fn shared_call_dag_keeps_reachable_out_of_line_definitions() {
    let mut f = increment();
    for _ in 0..12 {
        f = caller(f, 2);
    }
    let graph = emitted_graph(&f, 13);
    assert!(graph.values().any(|calls| !calls.is_empty()));
    assert!(graph.values().flatten().all(|id| graph.contains_key(id)));
    assert_eq!(
        Cranelift::compile(&f)
            .unwrap()
            .call(&[(-9i32).to_word()], &mut []),
        Ok((1 << 12) - 9)
    );
}

#[test]
fn packed_chunks_cover_the_vector_and_a_short_store_keeps_its_prefix() {
    use cranelift_codegen::ir::types::I16X8;
    use splice_core::{
        BufferKind,
        hir::ScalarType,
        hir::{Simd, SimdOp, Slice},
    };

    let e = ScalarType::U16;
    let mut f = FunctionBuilder::new(
        options(),
        Signature::new(
            &[ScalarType::I32; 1],
            ScalarType::I32,
            &[BufferKind::new(e, false), BufferKind::new(e, true)],
        ),
    );
    let result = f.arg(0).unwrap();
    let offset = f.cast(result, ScalarType::Index).unwrap();
    let mut simd = |op, args: &[_]| f.simd(Simd { element: e, op }, args).unwrap();
    let x = simd(SimdOp::Load(Slice::param(1, 0)), &[offset]).unwrap();
    let one = simd(SimdOp::Splat(1), &[]).unwrap();
    let sum = simd(SimdOp::Add, &[x, one]).unwrap();
    simd(SimdOp::Store(Slice::param(1, 1)), &[offset, sum]);
    f.ret(result).unwrap();
    let f = f.finish().unwrap();

    let mut jit = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let module = KernelModule::lower(&mut jit, &f, &options().target, false).unwrap();
    let body = &module.bodies[&module.entry];
    let chunks = e.lanes(options().target.vector_bytes()) / 8;
    assert_eq!(instructions_of_type(body, Opcode::Load, I16X8), chunks);
    assert_eq!(instructions_of_type(body, Opcode::Iadd, I16X8), chunks);
    assert_eq!(instructions_of_type(body, Opcode::Store, I16X8), chunks);

    let code = Cranelift::compile(&f).unwrap();
    let source: Vec<_> = (0..=e.lanes(options().target.vector_bytes()))
        .map(|x| u16::MAX - x as u16)
        .collect();
    let mut dst = vec![200u16; e.lanes(options().target.vector_bytes()) + 2];
    let mut expected = dst.clone();
    for i in 1..=e.lanes(options().target.vector_bytes()) {
        expected[i] = source[i].wrapping_add(1);
    }
    assert_eq!(
        code.call(
            &[1],
            &mut [source.as_slice().into(), dst.as_mut_slice().into()]
        ),
        Ok(1)
    );
    assert_eq!(dst, expected);

    // Cut the destination inside the second chunk when the host width permits.
    let limit = 9.min(e.lanes(options().target.vector_bytes()) - 1);
    dst.fill(200);
    let res = code.call(
        &[0],
        &mut [source.as_slice().into(), (&mut dst[..limit]).into()],
    );
    assert_eq!(res.unwrap_err(), CallError::Trap(Trap::Bounds));
    let expected: Vec<_> = (0..dst.len())
        .map(|i| {
            if i < limit {
                source[i].wrapping_add(1)
            } else {
                200
            }
        })
        .collect();
    assert_eq!(dst, expected);
}

#[test]
fn masked_contiguous_access_selects_packed_and_ordered_paths() {
    use cranelift_codegen::ir::types::I16X8;
    use splice_core::{
        BufferKind,
        hir::ScalarType,
        hir::{Simd, SimdOp, Slice},
    };

    // The only vector memory operations here have runtime masks, so a plain
    // vector load/store in CLIF must come from the masked access fast path.
    let e = ScalarType::U16;
    let mut f = FunctionBuilder::new(
        options(),
        Signature::new(
            &[ScalarType::I32; 2],
            ScalarType::I32,
            &[BufferKind::new(e, false), BufferKind::new(e, true)],
        ),
    );
    let offset = f.cast(f.arg(0).unwrap(), ScalarType::Index).unwrap();
    let result = f.arg(1).unwrap();
    let count = f.cast(result, ScalarType::Index).unwrap();
    let mut simd = |op, args: &[_]| f.simd(Simd { element: e, op }, args).unwrap();
    let mask = simd(SimdOp::FirstN, &[count]).unwrap();
    let zero = simd(SimdOp::Splat(0), &[]).unwrap();
    let v = simd(
        SimdOp::LoadSelect(Slice::param(2, 0)),
        &[offset, mask, zero],
    )
    .unwrap();
    simd(SimdOp::StoreSelect(Slice::param(2, 1)), &[offset, mask, v]);
    f.ret(result).unwrap();
    let f = f.finish().unwrap();

    let mut jit = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let module = KernelModule::lower(&mut jit, &f, &options().target, false).unwrap();
    let body = &module.bodies[&module.entry];
    assert!(instructions_of_type(body, Opcode::Load, I16X8) > 0);
    assert!(instructions_of_type(body, Opcode::Store, I16X8) > 0);

    let code = Cranelift::compile(&f).unwrap();
    let width = e.lanes(options().target.vector_bytes());
    let tail = 9.min(width - 1);
    let source: Vec<_> = (0..=width).map(|i| 100 + i as u16).collect();
    for (offset, active, src_len, dst_len) in [
        (1, width, width + 1, width + 1), // all chunks fully active
        (-1, 0, 0, 0),                    // invalid offset, no active addresses
        (1, tail, 1 + tail, 1 + tail),    // inactive lanes outside both slices
        (1, width, width + 1, 1 + tail),  // late store fault preserves earlier chunks
    ] {
        let mut actual = vec![17u16; width + 3];
        let mut expected = actual.clone();
        let mut status = Ok((active as i32).to_word());
        for lane in 0..active {
            let i = offset as usize + lane;
            if i >= dst_len {
                status = Err(CallError::Trap(Trap::Bounds));
                break;
            }
            expected[1 + i] = source[i];
        }
        let result = code.call(
            &[offset, active as i32].map(i32::to_word),
            &mut [
                (&source[..src_len]).into(),
                (&mut actual[1..1 + dst_len]).into(),
            ],
        );
        assert_eq!(result, status);
        assert_eq!(actual, expected);
    }
}

#[test]
fn a_full_gather_has_one_bounds_branch_for_all_lanes() {
    use splice::{kernel, std::intrinsics::simd::Lane};
    #[kernel]
    fn first_block(data: &[i32]) -> i32 {
        Lane::<i32>::gather(data, Lane::<i32>::iota()).last()
    }
    let mut jit = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let ir = first_block::ir().unwrap();
    let module = KernelModule::lower(&mut jit, &ir, &options().target, false).unwrap();
    let entry = &module.bodies[&module.entry];
    let branches = entry
        .layout
        .blocks()
        .flat_map(|block| entry.layout.block_insts(block))
        .filter(|&inst| entry.dfg.insts[inst].opcode() == Opcode::Brif)
        .count();
    assert_eq!(branches, 1);
}

#[test]
fn stack_shuffles_share_one_slot_per_function() {
    use splice_core::{
        hir::ScalarType,
        hir::{Simd, SimdOp},
    };

    let mut f = FunctionBuilder::new(
        options(),
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
    );
    let op = |op| Simd {
        element: ScalarType::I32,
        op,
    };
    let lane = f.arg(0).unwrap();
    let count = f.cast(lane, ScalarType::Index).unwrap();
    let iota = f.simd(op(SimdOp::Iota), &[]).unwrap().unwrap();
    let mask = f.simd(op(SimdOp::FirstN), &[count]).unwrap().unwrap();
    let packed = f
        .simd(op(SimdOp::Compress), &[iota, mask])
        .unwrap()
        .unwrap();
    let spread = f
        .simd(op(SimdOp::Expand), &[packed, mask, iota])
        .unwrap()
        .unwrap();
    let x = f
        .simd(op(SimdOp::ExtractWrapping), &[spread, lane])
        .unwrap()
        .unwrap();
    let replaced = f
        .simd(op(SimdOp::ReplaceWrapping), &[packed, lane, x])
        .unwrap()
        .unwrap();
    let y = f
        .simd(op(SimdOp::ExtractWrapping), &[replaced, lane])
        .unwrap()
        .unwrap();
    let sum = f.binary(Binary::Add, x, y).unwrap();
    f.ret(sum).unwrap();
    let f = f.finish().unwrap();

    let mut jit = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let module = KernelModule::lower(&mut jit, &f, &options().target, false).unwrap();
    let body = &module.bodies[&module.entry];
    assert_eq!(body.sized_stack_slots.len(), 1, "{body}");
    assert_eq!(
        body.sized_stack_slots.values().next().unwrap().size as usize,
        options().target.vector_bytes()
    );

    let code = Cranelift::compile(&f).unwrap();
    // Lanes below `lane` are compressed in order and expanded back, so both
    // reads see the lane's own index.
    for lane in 1..4 {
        assert_eq!(code.call(&[lane], &mut []), Ok(2 * lane));
    }
}
