use super::*;
use crate::Cranelift;
use cranelift_codegen::ir::{Opcode, Type};
use cranelift_jit::JITBuilder;
use cranelift_module::default_libcall_names;
use splice::{
    Codegen, Executable,
    hir::{Binary, FunctionBuilder},
};

fn increment() -> Function {
    let mut f = FunctionBuilder::new(1);
    let one = f.constant(1);
    let result = f.binary(Binary::Add, f.arg(0), one);
    f.ret(result);
    f.finish().unwrap()
}

fn caller(child: Function, times: usize) -> Function {
    let child = Arc::new(child);
    let mut f = FunctionBuilder::new(1);
    let mut value = f.arg(0);
    for _ in 0..times {
        value = f.call_shared(child.clone(), &[value]);
    }
    f.ret(value);
    f.finish().unwrap()
}

fn emitted_graph(f: &Function, unique_bodies: usize) -> HashMap<FuncId, Vec<FuncId>> {
    let mut jit = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let kernel = KernelModule::lower(&mut jit, f).unwrap();
    assert_eq!(kernel.bodies.len(), unique_bodies);
    let mut graph = HashMap::new();
    kernel
        .emit(&mut jit.make_context(), |id, ctx| {
            ctx.verify(jit.isa()).unwrap();
            let calls = local_callees(&ctx.func).collect();
            assert!(graph.insert(id, calls).is_none());
            Ok(())
        })
        .unwrap();
    graph
}

fn instructions_of_type(f: &ir::Function, opcode: Opcode, ty: Type) -> usize {
    f.layout
        .blocks()
        .flat_map(|b| f.layout.block_insts(b))
        .filter(|&i| f.dfg.insts[i].opcode() == opcode && f.dfg.ctrl_typevar(i) == ty)
        .count()
}

#[test]
fn nested_and_shared_calls_inline_across_function_boundaries() {
    let f = caller(caller(increment(), 2), 2);
    let graph = emitted_graph(&f, 3);
    assert_eq!(graph.len(), 1);
    assert!(graph.values().all(Vec::is_empty));
    assert_eq!(
        Cranelift::compile(&f).unwrap().call(&[i32::MAX], &mut []),
        Ok(i32::MIN + 3)
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
        Cranelift::compile(&f).unwrap().call(&[-9], &mut []),
        Ok((1 << 12) - 9)
    );
}

#[test]
fn packed_chunks_cover_the_vector_and_a_short_store_keeps_its_prefix() {
    use cranelift_codegen::ir::types::I16X8;
    use splice::{
        BufferKind,
        hir::{Buffer, Simd, SimdOp},
        simd::Element,
    };

    let e = Element::U16;
    let mut f =
        FunctionBuilder::with_buffers(1, vec![BufferKind::new(e, false), BufferKind::new(e, true)]);
    let offset = f.arg(0);
    let mut simd = |op, args: &[_]| f.simd(Simd { element: e, op }, args).unwrap();
    let x = simd(SimdOp::Load(Buffer(0)), &[offset]).unwrap();
    let one = simd(SimdOp::Splat(1), &[]).unwrap();
    let sum = simd(SimdOp::Add, &[x, one]).unwrap();
    simd(SimdOp::Store(Buffer(1)), &[offset, sum]);
    f.ret(offset);
    let f = f.finish().unwrap();

    let mut jit = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let module = KernelModule::lower(&mut jit, &f).unwrap();
    let body = &module.bodies[&module.entry];
    let chunks = e.lanes() / 8;
    assert_eq!(instructions_of_type(body, Opcode::Load, I16X8), chunks);
    assert_eq!(instructions_of_type(body, Opcode::Iadd, I16X8), chunks);
    assert_eq!(instructions_of_type(body, Opcode::Store, I16X8), chunks);

    let code = Cranelift::compile(&f).unwrap();
    let source: Vec<_> = (0..=e.lanes()).map(|x| u16::MAX - x as u16).collect();
    let mut dst = vec![200u16; e.lanes() + 2];
    let mut expected = dst.clone();
    for i in 1..=e.lanes() {
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
    let limit = 9.min(e.lanes() - 1);
    dst.fill(200);
    let res = code.call(
        &[0],
        &mut [source.as_slice().into(), (&mut dst[..limit]).into()],
    );
    assert_eq!(res.unwrap_err(), "kernel slice index out of bounds");
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
    use splice::{
        BufferKind,
        hir::{Buffer, Simd, SimdOp},
        simd::Element,
    };

    // The only vector memory operations here have runtime masks, so a plain
    // vector load/store in CLIF must come from the masked access fast path.
    let e = Element::U16;
    let mut f =
        FunctionBuilder::with_buffers(2, vec![BufferKind::new(e, false), BufferKind::new(e, true)]);
    let offset = f.arg(0);
    let count = f.arg(1);
    let mut simd = |op, args: &[_]| f.simd(Simd { element: e, op }, args).unwrap();
    let mask = simd(SimdOp::FirstN, &[count]).unwrap();
    let zero = simd(SimdOp::Splat(0), &[]).unwrap();
    let v = simd(SimdOp::LoadSelect(Buffer(0)), &[offset, mask, zero]).unwrap();
    simd(SimdOp::StoreSelect(Buffer(1)), &[offset, mask, v]);
    f.ret(count);
    let f = f.finish().unwrap();

    let mut jit = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let module = KernelModule::lower(&mut jit, &f).unwrap();
    let body = &module.bodies[&module.entry];
    assert!(instructions_of_type(body, Opcode::Load, I16X8) > 0);
    assert!(instructions_of_type(body, Opcode::Store, I16X8) > 0);

    let code = Cranelift::compile(&f).unwrap();
    let width = e.lanes();
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
        let mut status = Ok(active as i32);
        for lane in 0..active {
            let i = offset as usize + lane;
            if i >= dst_len {
                status = Err("kernel slice index out of bounds".to_owned());
                break;
            }
            expected[1 + i] = source[i];
        }
        let result = code.call(
            &[offset, active as i32],
            &mut [
                (&source[..src_len]).into(),
                (&mut actual[1..1 + dst_len]).into(),
            ],
        );
        assert_eq!(result, status);
        assert_eq!(actual, expected);
    }
}
