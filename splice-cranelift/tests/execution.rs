use splice::hir::{Binary, FunctionBuilder};
use splice::{BufferArg, Codegen, Executable, kernel};
use splice_cranelift::Cranelift;
use std::sync::Arc;

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
        batch::build::<Cranelift>(42).unwrap(),
        Cranelift::compile(&batch::ir(42).unwrap()).unwrap(),
    ] {
        let values = [9, i32::MIN, 15];
        for divisor in [1, -2] {
            let mut expected = [0; 3];
            let result = batch(&values, &mut expected, divisor, 42);
            let mut out = [0; 3];
            assert_eq!(
                code.call(
                    &[divisor],
                    &mut [BufferArg::Read(&values), BufferArg::Write(&mut out)]
                ),
                Ok(result)
            );
            assert_eq!(out, expected);
        }
        for (input, divisor, message, expected) in [
            (
                &values[..1],
                3,
                "kernel slice index out of bounds",
                [3, 42, 0],
            ),
            (
                &values[..],
                -1,
                "kernel signed division overflow",
                [-9, 42, 0],
            ),
            (
                &values[..],
                0,
                "kernel integer division by zero",
                [42, 0, 0],
            ),
        ] {
            let mut out = [0; 3];
            assert_eq!(
                code.call(
                    &[divisor],
                    &mut [BufferArg::Read(input), BufferArg::Write(&mut out)]
                )
                .unwrap_err(),
                message
            );
            assert_eq!(out, expected);
        }
        assert_eq!(
            code.call(&[0], &mut [BufferArg::Read(&[]), BufferArg::Write(&mut [])]),
            Ok(42)
        );
        let mut out = [17];
        assert!(
            code.call(
                &[],
                &mut [BufferArg::Read(&values), BufferArg::Write(&mut out)]
            )
            .is_err()
        );
        assert!(code.call(&[1], &mut [BufferArg::Write(&mut out)]).is_err());
        assert!(
            code.call(&[1], &mut [BufferArg::Read(&values), BufferArg::Read(&out)])
                .is_err()
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
        let mut f = FunctionBuilder::new(2);
        let result = f.binary(op, f.arg(0), f.arg(1));
        f.ret(result);
        let kernel = Cranelift::compile(&f.finish().unwrap()).unwrap();
        for a in cases {
            for b in cases {
                assert_eq!(
                    kernel.call(&[a, b], &mut []).ok(),
                    op.checked_eval(a, b),
                    "{a} {op:?} {b}"
                );
            }
        }
    }
}

#[test]
fn negative_and_past_end_indices_stop_before_the_first_store() {
    let code = update::build::<Cranelift>(42).unwrap();
    for row in [i32::MIN, -1, 1, i32::MAX] {
        let mut out = [17];
        assert_eq!(
            code.call(
                &[row, 1],
                &mut [BufferArg::Write(&mut out), BufferArg::Read(&[5])]
            )
            .unwrap_err(),
            "kernel slice index out of bounds"
        );
        assert_eq!(out, [17]);
    }
}

#[test]
fn unrepresentable_callee_signature_returns_a_build_error() {
    let count = usize::from(u16::MAX) + 2;
    let mut callee = FunctionBuilder::new(count);
    callee.ret(callee.arg(0));
    let mut root = FunctionBuilder::new(1);
    let value = root.call(callee.finish().unwrap(), &vec![root.arg(0); count]);
    root.ret(value);
    let result = Cranelift::compile(&root.finish().unwrap());
    assert!(result.err().unwrap().contains("too many parameters"));
}

#[test]
fn shared_callees_stack_arguments_and_backedges_outlive_the_ir() {
    let mut child = FunctionBuilder::new(8);
    let next = child.binary(Binary::Add, child.arg(0), child.arg(7));
    child.ret(next);
    let child = Arc::new(child.finish().unwrap());

    let mut f = FunctionBuilder::new(1);
    let limit = f.arg(0);
    let counter = f.constant(0);
    let zero = f.constant(0);
    let one = f.constant(1);
    let (head, body, done) = (f.label(), f.label(), f.label());
    f.mark(head);
    let more = f.binary(Binary::Lt, counter, limit);
    f.branch(more, body, done);
    f.mark(body);
    for _ in 0..2 {
        let mut args = [zero; 8];
        args[0] = counter;
        args[7] = one;
        let step = f.call_shared(child.clone(), &args);
        f.assign(counter, step);
    }
    f.jump(head);
    f.mark(done);
    f.ret(counter);
    let orphan = f.label();
    f.mark(orphan);
    f.constant(99); // verified, unreachable fallthrough

    let code = Cranelift::compile(&f.finish().unwrap()).unwrap();
    std::thread::scope(|scope| {
        for (limit, expected) in [(-2, 0), (1, 2), (19, 20)] {
            let code = &code;
            scope.spawn(move || assert_eq!(code.call(&[limit], &mut []), Ok(expected)));
        }
    });
}
