use splice::{BufferArg, CallError, Codegen, Executable, KernelScalar, Trap, kernel};
use splice_core::frontend::{KernelBuilder, Signature};
use splice_core::hir::{Binary, FunctionBuilder};
use splice_core::hir::{Constant, ScalarType};
use splice_cranelift::Cranelift;
use std::sync::Arc;

/// This machine's `SPLICE_SIMD` target in the mode of this test build.
fn options() -> splice_core::BuildOptions {
    splice_core::BuildOptions::host(splice_core::Mode::for_profile(cfg!(debug_assertions))).unwrap()
}

#[kernel]
fn update(out: &mut [i32], row: i32, values: &[i32], divisor: i32, #[comptime] bias: i32) -> i32 {
    out[row as usize] = bias;
    let value = values[row as usize] / divisor;
    out[row as usize] = value;
    value
}

#[kernel]
fn batch(values: &[i32], out: &mut [i32], divisor: i32, #[comptime] bias: i32) -> i32 {
    let mut result = bias;
    for row in 0..out.len() {
        result = update(out, row as i32, values, divisor, bias);
    }
    result
}

#[test]
fn staged_and_portable_calls_keep_writes_up_to_the_first_fault() {
    for code in [
        batch::build::<Cranelift>(batch::BuildArgs { bias: 42 }).unwrap(),
        Cranelift::compile(&batch::ir(42).unwrap()).unwrap(),
    ] {
        let values = [9, i32::MIN, 15];
        for divisor in [1, -2] {
            let mut expected = [0; 3];
            let result = batch(&values, &mut expected, divisor, 42);
            let mut out = [0; 3];
            assert_eq!(
                code.call(batch::CallArgs {
                    values: &values,
                    out: &mut out,
                    divisor
                }),
                Ok(result)
            );
            assert_eq!(out, expected);
        }
        for (input, divisor, message, expected) in [
            (&values[..1], 3, CallError::Trap(Trap::Bounds), [3, 42, 0]),
            (
                &values[..],
                -1,
                CallError::Trap(Trap::DivisionOverflow),
                [-9, 42, 0],
            ),
            (
                &values[..],
                0,
                CallError::Trap(Trap::DivisionByZero),
                [42, 0, 0],
            ),
        ] {
            let mut out = [0; 3];
            assert_eq!(
                code.call(batch::CallArgs {
                    values: input,
                    out: &mut out,
                    divisor
                })
                .unwrap_err(),
                message
            );
            assert_eq!(out, expected);
        }
        assert_eq!(
            code.call(batch::CallArgs {
                values: &[],
                out: &mut [],
                divisor: 0
            }),
            Ok(42)
        );
        let code = code.as_inner();
        let mut out = [17];
        use splice_core::SignatureError::*;
        assert_eq!(
            code.call(
                &[],
                &mut [BufferArg::read(&values), BufferArg::write(&mut out)]
            ),
            Err(Scalars {
                expected: 1,
                actual: 0
            }
            .into())
        );
        assert_eq!(
            code.call(&[1], &mut [BufferArg::write(&mut out)]),
            Err(Buffers {
                expected: 2,
                actual: 1
            }
            .into())
        );
        assert_eq!(
            code.call(&[1], &mut [BufferArg::read(&values), BufferArg::read(&out)]),
            Err(ReadOnly { buffer: 1 }.into())
        );
        assert_eq!(
            code.call(
                &[1],
                &mut [BufferArg::from(&[1.0f32][..]), BufferArg::write(&mut out)]
            ),
            Err(splice_core::SignatureError::Element {
                buffer: 0,
                expected: splice_core::hir::ScalarType::I32
            }
            .into())
        );
        assert_eq!(out, [17]);
    }
}

#[test]
fn every_binary_operation_matches_splice_on_signed_boundaries() {
    use Binary::*;
    let cases = [i32::MIN, -35, -1, 0, 1, 31, 32, i32::MAX];
    for op in [
        Add, Sub, Mul, Div, Rem, And, Or, Xor, Eq, Ne, Lt, Le, Gt, Ge, Shl, Shr, Min, Max,
    ] {
        let output = op.output(ScalarType::I32).unwrap();
        let mut f = FunctionBuilder::new(
            options(),
            Signature::new(&[ScalarType::I32; 2], output, &[]),
        );
        let result = f.binary(op, f.arg(0).unwrap(), f.arg(1).unwrap()).unwrap();
        f.ret(result).unwrap();
        let kernel = Cranelift::compile(&f.finish().unwrap()).unwrap();
        for a in cases {
            for b in cases {
                assert_eq!(
                    kernel.call(&[a, b].map(i32::to_word), &mut []),
                    op.checked_eval(a.into(), b.into())
                        .map(|c| c.bits())
                        .map_err(CallError::from),
                    "{a} {op:?} {b}"
                );
            }
        }
    }
}

#[test]
fn negative_and_past_end_indices_stop_before_the_first_store() {
    let code = update::build::<Cranelift>(update::BuildArgs { bias: 42 }).unwrap();
    for row in [i32::MIN, -1, 1, i32::MAX] {
        let mut out = [17];
        assert_eq!(
            code.call(update::CallArgs {
                out: &mut out,
                row,
                values: &[5],
                divisor: 1
            })
            .unwrap_err(),
            CallError::Trap(Trap::Bounds)
        );
        assert_eq!(out, [17]);
    }
}

#[test]
fn unrepresentable_callee_signature_returns_a_build_error() {
    let count = usize::from(u16::MAX) + 2;
    let mut callee = FunctionBuilder::new(
        options(),
        Signature::new(&vec![ScalarType::I32; count], ScalarType::I32, &[]),
    );
    callee.ret(callee.arg(0).unwrap()).unwrap();
    let mut root = FunctionBuilder::new(
        options(),
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
    );
    let callee = Arc::new(callee.finish().unwrap());
    let value = root
        .call_function(&callee, &vec![root.arg(0).unwrap(); count], &[])
        .unwrap();
    root.ret(value).unwrap();
    let result = Cranelift::compile(&root.finish().unwrap());
    assert!(
        matches!(result.err().unwrap(), splice_core::BuildError::Unsupported(feature) if feature.contains("too many parameters"))
    );
}

#[test]
fn shared_callees_stack_arguments_and_backedges_outlive_the_ir() {
    let mut child = FunctionBuilder::new(
        options(),
        Signature::new(&[ScalarType::I32; 8], ScalarType::I32, &[]),
    );
    let next = child
        .binary(Binary::Add, child.arg(0).unwrap(), child.arg(7).unwrap())
        .unwrap();
    child.ret(next).unwrap();
    let child = Arc::new(child.finish().unwrap());

    let mut f = FunctionBuilder::new(
        options(),
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
    );
    let limit = f.arg(0).unwrap();
    let counter = f.constant(Constant::from(0)).unwrap();
    let zero = f.constant(Constant::from(0)).unwrap();
    let one = f.constant(Constant::from(1)).unwrap();
    let (head, body, done) = (f.label().unwrap(), f.label().unwrap(), f.label().unwrap());
    f.mark(head).unwrap();
    let more = f.binary(Binary::Lt, counter, limit).unwrap();
    f.branch(more, body, done).unwrap();
    f.mark(body).unwrap();
    for _ in 0..2 {
        let mut args = [zero; 8];
        args[0] = counter;
        args[7] = one;
        let step = f.call_function(&child, &args, &[]).unwrap();
        f.assign(counter, step).unwrap();
    }
    f.jump(head).unwrap();
    f.mark(done).unwrap();
    f.ret(counter).unwrap();
    let orphan = f.label().unwrap();
    f.mark(orphan).unwrap();
    f.constant(Constant::from(99)).unwrap(); // verified, unreachable fallthrough

    let code = Cranelift::compile(&f.finish().unwrap()).unwrap();
    std::thread::scope(|scope| {
        for (limit, expected) in [(-2, 0), (1, 2), (19, 20)] {
            let code = &code;
            scope.spawn(move || assert_eq!(code.call(&[limit.to_word()], &mut []), Ok(expected)));
        }
    });
}

#[test]
fn masked_f64_memory_keeps_global_lane_order_across_chunks() {
    use splice_core::{
        BufferKind,
        hir::ScalarType::{F64, I64},
        hir::{Simd, SimdOp, Slice},
    };

    let mut f = FunctionBuilder::new(
        options(),
        Signature::new(
            &[ScalarType::I32; 0],
            ScalarType::I32,
            &[
                BufferKind::new(F64, false),
                BufferKind::new(I64, false),
                BufferKind::new(F64, false),
                BufferKind::new(F64, true),
            ],
        ),
    );
    let offset = f.constant(Constant::from(0usize)).unwrap();
    let mut simd = |element, op, args: &[_]| f.simd(Simd { element, op }, args).unwrap();
    let data = simd(F64, SimdOp::Load(Slice::param(0, 0)), &[offset]).unwrap();
    let indices = simd(I64, SimdOp::Load(Slice::param(0, 1)), &[offset]).unwrap();
    let zero = simd(F64, SimdOp::Splat(0), &[]).unwrap();
    let active = simd(F64, SimdOp::Gt, &[data, zero]).unwrap();
    let gathered = simd(
        F64,
        SimdOp::GatherSelect(Slice::param(0, 2)),
        &[indices, active, zero],
    )
    .unwrap();
    let sum = simd(F64, SimdOp::Add, &[data, gathered]).unwrap();
    let selected = simd(F64, SimdOp::Select, &[active, sum, zero]).unwrap();
    simd(
        F64,
        SimdOp::ScatterSelect(Slice::param(0, 3)),
        &[indices, active, selected],
    );
    let any = simd(F64, SimdOp::Any, &[active]).unwrap();
    let all = simd(F64, SimdOp::All, &[active]).unwrap();
    let any = f.cast(any, ScalarType::I32).unwrap();
    let all = f.cast(all, ScalarType::I32).unwrap();
    let result = f.binary(Binary::Add, any, all).unwrap();
    f.ret(result).unwrap();
    let code = Cranelift::compile(&f.finish().unwrap()).unwrap();

    let mut data: Vec<_> = (0..F64.lanes(options().target.vector_bytes()))
        .map(|i| (i + 1) as f64)
        .collect();
    let mut indices = vec![1; data.len()];
    data[1] = f64::NAN;
    indices[1] = i64::MAX;
    // Leave an entire middle 128-bit chunk inactive when the width permits.
    // Its invalid indices must be skipped while later active chunks still run.
    if data.len() > 4 {
        data[2..4].fill(0.0);
        indices[2..4].fill(i64::MIN);
    }
    let table = [0.0, 0.25, 0.5];
    for last_index in [1, 2] {
        let last = (0..data.len()).rfind(|&i| data[i] > 0.0).unwrap();
        indices[last] = last_index;
        let mut expected = [17.0, 29.0];
        let mut actual = expected;
        let mut expected_result = Ok(1);
        for (&x, &i) in data.iter().zip(&indices) {
            if x > 0.0 {
                let Some(slot) = expected.get_mut(i as usize) else {
                    expected_result = Err(CallError::Trap(Trap::Bounds));
                    break;
                };
                *slot = x + table[i as usize];
            }
        }
        let result = code.call(
            &[],
            &mut [
                data.as_slice().into(),
                indices.as_slice().into(),
                table.as_slice().into(),
                actual.as_mut_slice().into(),
            ],
        );
        assert_eq!(result, expected_result);
        assert_eq!(actual, expected);
    }
}
