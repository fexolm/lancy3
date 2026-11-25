use splice_core::frontend::{KernelBuilder, Signature};
use splice_core::hir::{Binary, Buffer, BufferKind, FunctionBuilder};
use splice_core::hir::{Constant, ScalarType};
use splice_core::{BufferArg, CallError, Codegen, Executable, KernelScalar, Trap};
use splice_llvm::Llvm;
use std::sync::Arc;

/// This machine's `SPLICE_SIMD` target in the mode of this test build.
fn options() -> splice_core::BuildOptions {
    splice_core::BuildOptions::host(splice_core::Mode::for_profile(cfg!(debug_assertions))).unwrap()
}

#[test]
fn all_binary_ops_match_splice_at_signed_boundaries() {
    use Binary::*;
    let ops = [
        Add, Sub, Mul, And, Or, Xor, Eq, Ne, Lt, Le, Gt, Ge, Shl, Shr, Min, Max, Div, Rem,
    ];
    let mut b = FunctionBuilder::new(
        options(),
        Signature::new(&[ScalarType::I32; 3], ScalarType::I32, &[]),
    );
    let (kind, left, right) = (b.arg(0).unwrap(), b.arg(1).unwrap(), b.arg(2).unwrap());
    for (i, &op) in ops.iter().enumerate() {
        let id = b.constant(Constant::from(i as i32)).unwrap();
        let chosen = b.binary(Eq, kind, id).unwrap();
        let (yes, next) = (b.label().unwrap(), b.label().unwrap());
        b.branch(chosen, yes, next).unwrap();
        b.mark(yes).unwrap();
        let mut result = b.binary(op, left, right).unwrap();
        if op.is_comparison() {
            result = b.cast(result, ScalarType::I32).unwrap();
        }
        b.ret(result).unwrap();
        b.mark(next).unwrap();
    }
    b.ret(kind).unwrap();
    let code = Llvm::compile(&b.finish().unwrap()).unwrap();
    let cases = [i32::MIN, -35, -1, 0, 1, 31, 32, i32::MAX];
    for (i, &op) in ops.iter().enumerate() {
        for &a in &cases {
            for &b in &cases {
                let expected = op
                    .checked_eval(a.into(), b.into())
                    .map(|c| c.bits())
                    .map_err(CallError::from);
                assert_eq!(
                    code.call(&[i as i32, a, b].map(i32::to_word), &mut []),
                    expected,
                    "{a} {op:?} {b}"
                );
            }
        }
    }
}

#[test]
fn checked_calls_remap_buffers_and_stop_after_the_first_fault() {
    let mut child = FunctionBuilder::new(
        options(),
        Signature::new(
            &[ScalarType::I32; 2],
            ScalarType::I32,
            &[BufferKind::Write, BufferKind::Read],
        ),
    );
    let zero = child.constant(Constant::from(0usize)).unwrap();
    let one = child.constant(Constant::from(1usize)).unwrap();
    let mark = child.constant(Constant::from(42)).unwrap();
    child.store(Buffer(0), zero, mark).unwrap();
    let at = child
        .cast(child.arg(0).unwrap(), ScalarType::Index)
        .unwrap();
    let value = child.load(Buffer(1), at).unwrap();
    let quotient = child
        .binary(Binary::Div, value, child.arg(1).unwrap())
        .unwrap();
    child.store(Buffer(0), one, quotient).unwrap();
    let length = child.buffer_len(Buffer(1)).unwrap();
    let length = child.cast(length, ScalarType::I32).unwrap();
    child.ret(length).unwrap();

    let mut parent = FunctionBuilder::new(
        options(),
        Signature::new(
            &[ScalarType::I32; 2],
            ScalarType::I32,
            &[BufferKind::Read, BufferKind::Write],
        ),
    );
    let child = Arc::new(child.finish().unwrap());
    let length = parent
        .call_function(
            &child,
            &[parent.arg(0).unwrap(), parent.arg(1).unwrap()],
            &[Buffer(1), Buffer(0)],
        )
        .unwrap();
    let two = parent.constant(Constant::from(2usize)).unwrap();
    parent.store(Buffer(1), two, length).unwrap();
    parent.ret(length).unwrap();
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
            BufferArg::read::<i32>(&[8, i32::MIN]),
            BufferArg::write(&mut output),
        ];
        let result = code.call(&[index, divisor].map(i32::to_word), &mut buffers);
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
    let mut step = FunctionBuilder::new(
        options(),
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
    );
    let one = step.constant(Constant::from(1)).unwrap();
    let next = step.binary(Binary::Add, step.arg(0).unwrap(), one).unwrap();
    step.ret(next).unwrap();
    let step = Arc::new(step.finish().unwrap());

    let mut b = FunctionBuilder::new(
        options(),
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
    );
    let i = b.constant(Constant::from(0)).unwrap();
    let (head, body, done, orphan) = (
        b.label().unwrap(),
        b.label().unwrap(),
        b.label().unwrap(),
        b.label().unwrap(),
    );
    b.mark(head).unwrap();
    let more = b.binary(Binary::Lt, i, b.arg(0).unwrap()).unwrap();
    b.branch(more, body, done).unwrap();
    b.mark(body).unwrap();
    let a = b.call_function(&step, &[i], &[]).unwrap();
    let z = b.call_function(&step, &[a], &[]).unwrap();
    b.assign(i, z).unwrap();
    b.jump(head).unwrap();
    b.mark(done).unwrap();
    b.ret(i).unwrap();
    // A verified function may have unreachable operations after a terminator,
    // adjacent labels and a dead tail without an explicit return.
    b.constant(Constant::from(33)).unwrap();
    b.mark(orphan).unwrap();
    let tail = b.label().unwrap();
    b.mark(tail).unwrap();
    b.constant(Constant::from(11)).unwrap();
    let f = b.finish().unwrap();
    let ir = splice_llvm::to_llvm_ir(&f, &f.options().target).unwrap();
    assert_eq!(ir.matches("define internal fastcc i32 @kernel").count(), 2);
    let code = Llvm::compile(&f).unwrap();
    for (limit, result) in [(-2, 0), (0, 0), (1, 2), (17, 18), (72, 72)] {
        let word = code.call(&[limit.to_word()], &mut []).unwrap();
        assert_eq!(i32::from_word(word), result);
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
        let zero = b.constant(Constant::from(0usize))?;
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
        let result = b.constant(Constant::from(0))?;
        b.ret(result)?;
        Ok(())
    }
    let ir = frontend::materialize(
        Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &kinds),
        Box::new(stage),
        options(),
    )
    .unwrap();
    let n = Element::F64.lanes(options().target.vector_bytes());
    for kernel in [
        Llvm::build_with_options(
            Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &kinds),
            Box::new(stage),
            options(),
        )
        .unwrap(),
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
    // One portable function, listed for several targets in one process.
    let ir = compact::ir().unwrap();
    let targets: Vec<_> = splice_core::target::Profile::ALL
        .iter()
        .filter_map(|&profile| splice_core::Target::from_profile(profile).ok())
        .filter(|target| target.arch() == splice_core::Target::host().unwrap().arch())
        .collect();
    assert!(targets.len() > 1);
    for target in targets {
        let listing = splice_llvm::to_llvm_ir(&ir, &target).unwrap();
        let width = target.vector_bytes() / 4;
        assert!(
            listing
                .lines()
                .any(|line| line.contains("getelementptr i32, ptr")
                    && line.contains(&format!("<{width} x i32>"))),
            "{target:?}"
        );
        let native = target
            .x86()
            .is_some_and(|isa| isa.avx10 > 0 || isa.avx512f && (width == 16 || isa.vl));
        let intrinsic = if native {
            "@llvm.x86.avx512.mask.compress"
        } else {
            "@llvm.experimental.vector.compress"
        };
        assert!(listing.contains(intrinsic), "{target:?}");
    }
}
