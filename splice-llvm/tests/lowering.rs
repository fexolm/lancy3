use splice_core::hir::{Binary, Buffer, BufferKind, FunctionBuilder};
use splice_core::{BufferArg, CallError, Codegen, Executable, Trap};
use splice_llvm::Llvm;
use std::sync::Arc;

#[test]
fn all_binary_ops_match_splice_at_signed_boundaries() {
    use Binary::*;
    let ops = [
        Add, Sub, Mul, And, Or, Xor, Eq, Ne, Lt, Le, Gt, Ge, Shl, Shr, Min, Max, Div, Rem,
    ];
    let mut b = FunctionBuilder::new(3);
    let (kind, left, right) = (b.arg(0), b.arg(1), b.arg(2));
    for (i, &op) in ops.iter().enumerate() {
        let id = b.constant(i as i32);
        let chosen = b.binary(Eq, kind, id);
        let (yes, next) = (b.label(), b.label());
        b.branch(chosen, yes, next);
        b.mark(yes);
        let result = b.binary(op, left, right);
        b.ret(result);
        b.mark(next);
    }
    b.ret(kind);
    let code = Llvm::compile(&b.finish().unwrap()).unwrap();
    let cases = [i32::MIN, -35, -1, 0, 1, 31, 32, i32::MAX];
    for (i, &op) in ops.iter().enumerate() {
        for &a in &cases {
            for &b in &cases {
                let expected = op.checked_eval(a, b).map_err(CallError::from);
                assert_eq!(
                    code.call(&[i as i32, a, b], &mut []),
                    expected,
                    "{a} {op:?} {b}"
                );
            }
        }
    }
}

#[test]
fn checked_calls_remap_buffers_and_stop_after_the_first_fault() {
    let mut child = FunctionBuilder::with_buffers(2, vec![BufferKind::Write, BufferKind::Read]);
    let zero = child.constant(0);
    let one = child.constant(1);
    let mark = child.constant(42);
    child.store(Buffer(0), zero, mark);
    let value = child.load(Buffer(1), child.arg(0));
    let quotient = child.binary(Binary::Div, value, child.arg(1));
    child.store(Buffer(0), one, quotient);
    let length = child.buffer_len(Buffer(1));
    child.ret(length);

    let mut parent = FunctionBuilder::with_buffers(2, vec![BufferKind::Read, BufferKind::Write]);
    let length = parent
        .call_with_buffers(
            child.finish().unwrap(),
            &[parent.arg(0), parent.arg(1)],
            &[Buffer(1), Buffer(0)],
        )
        .unwrap();
    let two = parent.constant(2);
    parent.store(Buffer(1), two, length);
    parent.ret(length);
    let code = Llvm::compile(&parent.finish().unwrap()).unwrap();

    for (index, divisor, expected) in [
        (-1, 1, None),
        (2, 1, None),
        (0, 0, None),
        (1, -1, None),
        (0, -2, Some(-4)),
    ] {
        let mut output = [0; 3];
        let mut buffers = [
            BufferArg::Read(&[8, i32::MIN]),
            BufferArg::Write(&mut output),
        ];
        let result = code.call(&[index, divisor], &mut buffers);
        match expected {
            Some(value) => {
                assert_eq!(result.unwrap(), 2);
                assert_eq!(output, [42, value, 2]);
            }
            None => {
                assert!(result.is_err());
                assert_eq!(output, [42, 0, 0]);
            }
        }
    }
}

#[test]
fn shared_callees_loops_and_unreachable_tails_form_valid_llvm_blocks() {
    let mut step = FunctionBuilder::new(1);
    let one = step.constant(1);
    let next = step.binary(Binary::Add, step.arg(0), one);
    step.ret(next);
    let step = Arc::new(step.finish().unwrap());

    let mut b = FunctionBuilder::new(1);
    let i = b.constant(0);
    let (head, body, done, orphan) = (b.label(), b.label(), b.label(), b.label());
    b.mark(head);
    let more = b.binary(Binary::Lt, i, b.arg(0));
    b.branch(more, body, done);
    b.mark(body);
    let a = b.call_shared(Arc::clone(&step), &[i]);
    let z = b.call_shared(step, &[a]);
    b.assign(i, z);
    b.jump(head);
    b.mark(done);
    b.ret(i);
    // A verified function may have unreachable operations after a terminator,
    // adjacent labels and a dead tail without an explicit return.
    b.constant(33);
    b.mark(orphan);
    let tail = b.label();
    b.mark(tail);
    b.constant(11);
    let f = b.finish().unwrap();
    let ir = splice_llvm::to_llvm_ir(&f).unwrap();
    assert_eq!(ir.matches("define internal fastcc i32 @kernel").count(), 2);
    let code = Llvm::compile(&f).unwrap();
    for (limit, result) in [(-2, 0), (0, 0), (1, 2), (17, 18), (72, 72)] {
        assert_eq!(code.call(&[limit], &mut []).unwrap(), result);
    }
}

#[test]
fn masked_f64_gather_checks_active_indices_at_full_i64_width() {
    use splice_core::{
        frontend::{self, KernelBuilder},
        hir::{Simd, SimdOp},
        simd::Element,
    };
    let kinds = [
        BufferKind::new(Element::I64, false),
        BufferKind::new(Element::F64, false),
        BufferKind::new(Element::F64, true),
    ];
    fn stage(b: &mut dyn KernelBuilder) -> splice_core::Result<()> {
        let zero = b.constant(0);
        let i64op = |op| Simd {
            element: Element::I64,
            op,
        };
        let f64op = |op| Simd {
            element: Element::F64,
            op,
        };
        let idx = b.simd(i64op(SimdOp::Load(Buffer(0))), &[zero])?.unwrap();
        let positions = b.simd(f64op(SimdOp::Iota), &[])?.unwrap();
        let one = b
            .simd(f64op(SimdOp::Splat(1.0_f64.to_bits())), &[])?
            .unwrap();
        let first = b.simd(f64op(SimdOp::Lt), &[positions, one])?.unwrap();
        let selected = b
            .simd(
                f64op(SimdOp::GatherSelect(Buffer(1))),
                &[idx, first, positions],
            )?
            .unwrap();
        b.simd(f64op(SimdOp::Store(Buffer(2))), &[zero, selected])?;
        b.ret(zero);
        Ok(())
    }
    let ir = frontend::materialize(0, &kinds, Box::new(stage)).unwrap();
    let n = Element::F64.lanes();
    for kernel in [
        Llvm::build(0, &kinds, Box::new(stage)).unwrap(),
        Llvm::compile(&ir).unwrap(),
    ] {
        let src = [-17.0];
        let mut indices = vec![-1_i64; n];
        for active in [0, 1_i64 << 32, -1] {
            indices[0] = active;
            let mut out = vec![777.0_f64; n];
            let result = kernel.call(
                &[],
                &mut [
                    BufferArg::from(indices.as_slice()),
                    BufferArg::from(&src[..]),
                    BufferArg::from(out.as_mut_slice()),
                ],
            );
            if active == 0 {
                assert_eq!(result, Ok(0));
                let mut expected: Vec<_> = (0..n).map(|i| i as f64).collect();
                expected[0] = src[0];
                assert_eq!(out, expected);
            } else {
                assert_eq!(result, Err(CallError::Trap(Trap::Bounds)));
                assert_eq!(out, vec![777.0; n]);
            }
        }
    }
}

#[test]
fn i32_gather_keeps_dword_indices_and_native_compaction() {
    use splice_core::{kernel, simd::Lane};
    #[kernel(crate = splice_core)]
    fn compact(src: &[i32], ids: &[i32], out: &mut [i32]) -> i32 {
        let positions = Lane::<i32>::load(ids, 0);
        let words = Lane::<i32>::gather(src, positions);
        let keep = words.simd_ge(Lane::splat(0));
        words.compress(keep).store(out, 0);
        keep.count()
    }
    let ir = splice_llvm::to_llvm_ir(&compact::ir().unwrap()).unwrap();
    let width = Lane::<i32>::lanes();
    assert!(
        ir.lines()
            .any(|line| line.contains("getelementptr i32, ptr")
                && line.contains(&format!("<{width} x i32>")))
    );
    #[cfg(target_arch = "x86_64")]
    let native = match splice_target::simd().unwrap().profile {
        splice_target::Profile::Auto => {
            is_x86_feature_detected!("avx512f")
                && (width == 16 || is_x86_feature_detected!("avx512vl"))
        }
        splice_target::Profile::Avx512 | splice_target::Profile::Avx10 => true,
        _ => false,
    };
    #[cfg(not(target_arch = "x86_64"))]
    let native = false;
    if native {
        assert!(ir.contains("@llvm.x86.avx512.mask.compress"));
    } else {
        assert!(ir.contains("@llvm.experimental.vector.compress"));
    }
}
