use super::frame::frame;
use super::*;
use crate::hir::{Buffer, BufferArg, BufferKind, FunctionBuilder};
use crate::jit::{Executable, ScalarKernel};
use std::sync::Arc;

#[test]
fn bit_test_branches_to_return_through_shared_join() {
    for mask in [8, 0x4000_0100, i32::MIN] {
        for comparison in [Binary::Eq, Binary::Ne] {
            let mut b = FunctionBuilder::new(2);
            let (early, slow, join) = (b.label(), b.label(), b.label());
            let result = b.local();
            let value = b.constant(mask);
            let masked = b.binary(Binary::And, value, b.arg(0));
            let zero = b.constant(0);
            let predicate = b.binary(comparison, zero, masked);
            b.branch(predicate, early, slow);
            b.mark(early);
            let choice = b.constant(57);
            b.assign(result, choice);
            b.jump(join);
            b.mark(slow);
            let sum = b.binary(Binary::Add, b.arg(0), b.arg(1));
            b.assign(result, sum);
            b.mark(join);
            let output = b.local();
            b.assign(output, result);
            b.ret(output);
            let f = b.finish().unwrap();

            let (bytes, _) = emit(&f).unwrap();
            let branch = bytes
                .windows(2)
                .position(|b| b == [0x0f, if comparison == Binary::Eq { 0x84 } else { 0x85 }])
                .unwrap();
            let disp = i32::from_le_bytes(bytes[branch + 2..branch + 6].try_into().unwrap());
            let target = (branch + 6).checked_add_signed(disp as isize).unwrap();
            // Jcc targets the out-of-line mov eax,57; ret. The computed
            // arm falls through and has its own return immediately before it.
            assert_eq!(bytes[target - 1], 0xc3);
            assert_eq!(&bytes[target..], &[0xb8, 57, 0, 0, 0, 0xc3]);
            let kernel = ScalarKernel::baseline(&f).unwrap();
            for x in [0, 8, 256, i32::MIN, i32::MAX, -1] {
                assert_eq!(kernel.call(&[x, 11]), f.interpret(&[x, 11]));
            }
        }
    }
}

#[test]
fn direct_three_abi_preserves_arguments_and_workspace_bounds() {
    type Direct = unsafe extern "C" fn(i32, i32, i32, *mut i32) -> i32;
    for pressure in [false, true] {
        let mut b = FunctionBuilder::new(3);
        let left = b.binary(Binary::Sub, b.arg(2), b.arg(0));
        let right = b.binary(Binary::Mul, b.arg(1), b.arg(2));
        let mut result = b.binary(Binary::Xor, left, right);
        if pressure {
            let mut hot = Vec::new();
            for k in 0..7 {
                let bias = b.constant(17 * k);
                hot.push(b.binary(Binary::Add, left, bias));
            }
            for _ in 0..6 {
                for &value in &hot {
                    result = b.binary(Binary::Sub, result, value);
                }
            }
        }
        b.ret(result);
        let f = b.finish().unwrap();
        if pressure {
            assert!(
                frame(&f, true).homes[..3]
                    .iter()
                    .all(|h| matches!(h, Home::Slot(_)))
            );
        }
        let (code, words) = emit3(&f).unwrap();
        let executable = Executable::new(&code).unwrap();
        // SAFETY: emit3 fixes the ABI, the mapping outlives all calls,
        // and words is the maximum workspace in i32 slots.
        let call = unsafe {
            std::mem::transmute::<*mut std::ffi::c_void, Direct>(executable.address.as_ptr())
        };
        let mut scratch = vec![0x1afe_bead; words + 2];
        for [a, b, c] in [
            [2, 7, -3],
            [2, -3, 7],
            [-3, 2, 7],
            [-3, 7, 2],
            [7, 2, -3],
            [7, -3, 2],
        ] {
            let actual = unsafe { call(a, b, c, scratch[1..].as_mut_ptr()) };
            assert_eq!(Ok(actual), f.interpret(&[a, b, c]));
            assert_eq!(scratch[0], 0x1afe_bead);
            assert_eq!(scratch[words + 1], 0x1afe_bead);
        }
    }
}

#[test]
fn signed_comparisons_and_masked_shift_counts_are_bit_exact() {
    use Binary::*;
    for op in [
        Add, Sub, Mul, And, Or, Xor, Eq, Ne, Lt, Le, Gt, Ge, Shl, Shr, Min, Max,
    ] {
        for static_rhs in [false, true] {
            for rhs in [i32::MIN, -129, -1, 0, 7, 32, 63, 0x5566_7788] {
                let mut b = FunctionBuilder::new(2);
                let value = if static_rhs {
                    b.constant(rhs)
                } else {
                    b.arg(1)
                };
                let result = b.binary(op, b.arg(0), value);
                b.ret(result);
                let function = b.finish().unwrap();
                let machine = ScalarKernel::compile(&function).unwrap();
                for lhs in [i32::MIN, -42, 0, 1, i32::MAX] {
                    let args = [lhs, rhs];
                    assert_eq!(
                        machine.call(&args),
                        function.interpret(&args),
                        "{lhs} {op:?} {rhs}"
                    );
                }
            }
        }
    }
}

#[test]
fn arithmetic_and_spilled_operands_match_the_reference() {
    let mut b = FunctionBuilder::new(40);
    let mut acc = b.arg(0);
    use Binary::*;
    let ops = [
        Add, Sub, Mul, And, Or, Xor, Shl, Shr, Eq, Ne, Lt, Le, Gt, Ge,
    ];
    for i in 1..40 {
        let x = b.binary(ops[(i - 1) % ops.len()], acc, b.arg(i));
        // Keep all parameters live and exercise displacements beyond +127.
        acc = b.binary(Xor, x, b.arg(39 - i));
    }
    b.ret(acc);
    let function = b.finish().unwrap();
    let compiled = ScalarKernel::compile(&function).unwrap();
    for seed in [1i32, -3, i32::MIN, i32::MAX, 0x1234_5678] {
        let input: Vec<_> = (0..40i32)
            .map(|i| seed.wrapping_mul(31 + i).wrapping_add(i * i))
            .collect();
        assert_eq!(compiled.call(&input), function.interpret(&input));
    }
}

#[test]
fn nested_shared_calls_return_to_the_correct_site_and_preserve_caller_values() {
    let mut b = FunctionBuilder::new(2);
    let hi = b.label();
    let lo = b.label();
    let cond = b.binary(Binary::Gt, b.arg(0), b.arg(1));
    b.branch(cond, hi, lo);
    b.mark(hi);
    let x = b.binary(Binary::Sub, b.arg(0), b.arg(1));
    b.ret(x);
    b.mark(lo);
    let y = b.binary(Binary::Add, b.arg(0), b.arg(1));
    b.ret(y);
    let leaf = Arc::new(b.finish().unwrap());

    let mut b = FunctionBuilder::new(2);
    let x = b.call_shared(leaf.clone(), &[b.arg(0), b.arg(1)]);
    let y = b.call_shared(leaf.clone(), &[b.arg(1), b.arg(0)]);
    let result = b.binary(Binary::Sub, x, y);
    b.ret(result);
    let middle = Arc::new(b.finish().unwrap());
    let mut b = FunctionBuilder::new(3);
    let x = b.call_shared(middle.clone(), &[b.arg(0), b.arg(1)]);
    let y = b.call_shared(middle, &[b.arg(2), b.arg(0)]);
    let z = b.call_shared(leaf, &[x, y]);
    let result = b.binary(Binary::Xor, z, b.arg(1));
    b.ret(result);
    let function = b.finish().unwrap();
    assert_eq!(Graph::discover(&function).functions.len(), 3);
    let compiled = ScalarKernel::compile(&function).unwrap();
    for a in [-3, 0, 11, i32::MIN, i32::MAX] {
        for c in [3, -5, i32::MAX] {
            let args = [a, -2, c];
            assert_eq!(compiled.call(&args), function.interpret(&args));
        }
    }
}

#[test]
fn wide_shared_call_graph_compiles_once_per_callee() {
    let mut leaf = FunctionBuilder::new(1);
    leaf.ret(leaf.arg(0));
    let mut callee = Arc::new(leaf.finish().unwrap());
    // Two alternative call sites at every level. Naive inlining generates
    // 2^16 copies of the leaf. Each invocation actually follows one path.
    for _ in 0..16 {
        let mut b = FunctionBuilder::new(1);
        let yes = b.label();
        let no = b.label();
        b.branch(b.arg(0), yes, no);
        b.mark(yes);
        let a = b.call_shared(callee.clone(), &[b.arg(0)]);
        b.ret(a);
        b.mark(no);
        let b_result = b.call_shared(callee, &[b.arg(0)]);
        b.ret(b_result);
        callee = Arc::new(b.finish().unwrap());
    }
    let code = ScalarKernel::compile(&callee).unwrap();
    assert!(code.code_size() < 16_384);
    assert_eq!(code.call(&[0]), Ok(0));
    assert_eq!(code.call(&[-9]), Ok(-9));
}

#[test]
fn loop_backedge_observes_mutable_arguments() {
    let mut b = FunctionBuilder::new(1);
    let acc = b.local();
    let n = b.arg(0);
    let zero = b.constant(0);
    let one = b.constant(1);
    b.assign(acc, zero);
    let header = b.label();
    let body = b.label();
    let exit = b.label();
    b.mark(header);
    let positive = b.binary(Binary::Gt, n, zero);
    b.branch(positive, body, exit);
    b.mark(body);
    let sum = b.binary(Binary::Add, acc, n);
    b.assign(acc, sum);
    let next = b.binary(Binary::Sub, n, one);
    b.assign(n, next);
    b.jump(header);
    b.mark(exit);
    b.ret(acc);
    let compiled = ScalarKernel::compile(&b.finish().unwrap()).unwrap();
    for (input, expected) in [(-2, 0), (0, 0), (1, 1), (1000, 500_500)] {
        assert_eq!(compiled.call(&[input]), Ok(expected));
    }
}

#[test]
fn nested_buffer_errors_return_to_the_host_before_later_stores() {
    use BufferKind::{Read, Write};
    let first = Buffer(0);
    let second = Buffer(1);
    let mut leaf = FunctionBuilder::with_buffers(2, vec![Read, Write]);
    let data = leaf.load(first, leaf.arg(0));
    let len = leaf.buffer_len(first);
    let value = leaf.binary(Binary::Add, data, len);
    leaf.store(second, leaf.arg(1), value);
    leaf.ret(value);

    let mut middle = FunctionBuilder::with_buffers(2, vec![Read, Write]);
    let result = middle
        .call_with_buffers(
            leaf.finish().unwrap(),
            &[middle.arg(0), middle.arg(1)],
            &[first, second],
        )
        .unwrap();
    let two = middle.constant(2);
    middle.store(second, two, result);
    middle.ret(result);

    let mut root = FunctionBuilder::with_buffers(2, vec![Write, Read]);
    let zero = root.constant(0);
    let nine = root.constant(9);
    root.store(first, zero, nine);
    let result = root
        .call_with_buffers(
            middle.finish().unwrap(),
            &[root.arg(0), root.arg(1)],
            &[second, first],
        )
        .unwrap();
    let three = root.constant(3);
    root.store(first, three, result);
    root.ret(result);
    let f = root.finish().unwrap();
    for kernel in [
        ScalarKernel::baseline(&f).unwrap(),
        ScalarKernel::compile(&f).unwrap(),
    ] {
        let mut caller = kernel.caller();
        let source = [-3, 46, 1];
        // Reuse the workspace after faults as well as after a successful call.
        for (index, target) in [(-1, 1), (1, -1), (3, 0), (0, 4), (1, 1)] {
            let mut output = [-77; 4];
            let status = caller.call_with_buffers(
                &[index, target],
                &mut [BufferArg::Write(&mut output), BufferArg::Read(&source)],
            );
            if (index, target) == (1, 1) {
                assert_eq!(status, Ok(49));
                assert_eq!(output, [9, 49, 49, 49]);
            } else {
                assert!(status.is_err());
                assert_eq!(output, [9, -77, -77, -77]);
            }
        }
    }
}

#[test]
fn unused_load_still_checks_an_empty_buffer() {
    let mut f = FunctionBuilder::with_buffers(0, vec![BufferKind::Read]);
    let zero = f.constant(0);
    f.load(Buffer(0), zero);
    f.ret(zero);
    let f = f.finish().unwrap();
    for kernel in [
        ScalarKernel::baseline(&f).unwrap(),
        ScalarKernel::compile(&f).unwrap(),
    ] {
        assert!(
            kernel
                .call_with_buffers(&[], &mut [BufferArg::Read(&[])])
                .is_err()
        );
        assert_eq!(
            kernel.call_with_buffers(&[], &mut [BufferArg::Read(&[11])]),
            Ok(0)
        );
    }
}

#[test]
fn spilled_index_and_value_address_a_late_buffer() {
    let mut kinds = vec![BufferKind::Read; 64];
    kinds.push(BufferKind::Write);
    let mut f = FunctionBuilder::with_buffers(6, kinds);
    let dst = Buffer(64);
    f.store(dst, f.arg(4), f.arg(5));
    let mut result = f.constant(0);
    for _ in 0..3 {
        for i in 0..4 {
            result = f.binary(Binary::Add, result, f.arg(i));
        }
    }
    f.ret(result);
    let f = f.finish().unwrap();
    let layout = frame(&f, true);
    assert!(matches!(layout.homes[4], Home::Slot(_)));
    assert!(matches!(layout.homes[5], Home::Slot(_)));
    let kernel = ScalarKernel::baseline(&f).unwrap();
    let mut output = [-1; 3];
    let mut buffers: Vec<_> = (0..64).map(|_| BufferArg::Read(&[])).collect();
    buffers.push(BufferArg::Write(&mut output));
    assert_eq!(
        kernel.call_with_buffers(&[1, 2, 3, 4, 2, 71], &mut buffers),
        Ok(30)
    );
    assert_eq!(output, [-1, -1, 71]);
}

#[test]
fn checked_division_preserves_spills_and_propagates_child_faults() {
    for op in [Binary::Div, Binary::Rem] {
        let mut child = FunctionBuilder::new(6);
        let quotient = child.binary(op, child.arg(4), child.arg(5));
        let mut sum = child.constant(0);
        for _ in 0..3 {
            for i in 0..4 {
                sum = child.binary(Binary::Add, sum, child.arg(i));
            }
        }
        let total = child.binary(Binary::Add, quotient, sum);
        child.ret(total);
        let child = child.finish().unwrap();
        let layout = frame(&child, false);
        for reg in [4, 5, quotient.0 as usize] {
            assert!(matches!(layout.homes[reg], Home::Slot(_)));
        }
        let mut root = FunctionBuilder::new(6);
        let args: Vec<_> = (0..6).map(|i| root.arg(i)).collect();
        let total = root.call(child, &args);
        let extra = root.binary(Binary::Add, total, root.arg(0));
        root.ret(extra);
        let root = root.finish().unwrap();
        for kernel in [
            ScalarKernel::baseline(&root).unwrap(),
            ScalarKernel::compile(&root).unwrap(),
        ] {
            let mut caller = kernel.caller();
            for (a, b) in [
                (5, 0),
                (i32::MIN, -1),
                (-7, 3),
                (7, -3),
                (-7, -3),
                (0, i32::MIN),
                (i32::MIN, 2),
                (i32::MAX, 1),
            ] {
                let actual = caller.call(&[1, 2, 3, 4, a, b]);
                match (a, b) {
                    (_, 0) => assert!(actual.unwrap_err().contains("zero")),
                    (i32::MIN, -1) => assert!(actual.unwrap_err().contains("overflow")),
                    _ => {
                        let expected = if op == Binary::Div { a / b } else { a % b };
                        assert_eq!(actual, Ok(expected.wrapping_add(31)));
                    }
                }
            }
        }
    }
}

#[test]
fn discarded_division_by_constants_still_reports_faults() {
    for op in [Binary::Div, Binary::Rem] {
        for divisor in [-1, 0, 2] {
            let mut b = FunctionBuilder::new(1);
            let rhs = b.constant(divisor);
            b.binary(op, b.arg(0), rhs);
            b.ret(rhs);
            let f = b.finish().unwrap();
            for kernel in [
                ScalarKernel::baseline(&f).unwrap(),
                ScalarKernel::compile(&f).unwrap(),
            ] {
                for lhs in [i32::MIN, 0, i32::MAX] {
                    let actual = kernel.call(&[lhs]);
                    let safe = lhs.checked_div(divisor).is_some();
                    assert_eq!(actual.ok(), safe.then_some(divisor));
                }
            }
        }
    }
}
