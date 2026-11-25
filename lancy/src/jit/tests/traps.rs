use crate::hir::{Binary, Function, Op, Scale, Terminator, Type};
use crate::{CallError, Inlining, ScalarKernel, Trap};
use std::sync::Arc;

#[test]
fn bounds_check_compares_both_i32_operands_as_unsigned() {
    let mut f = Function::scalar(2);
    let b = f.entry();
    let [index, len] = f.params(b).try_into().unwrap();
    f.append_inst(b, Op::BoundsCheck(index, len)).unwrap();
    f.set_terminator(b, Terminator::Return(index));
    let kernel = ScalarKernel::compile(&f, &crate::test_target::options()).unwrap();
    for index in [0, 1, i32::MAX, i32::MIN, -2, -1] {
        for len in [0, 1, i32::MAX, i32::MIN, -2, -1] {
            let expected = if (index as u32) < len as u32 {
                Ok(index)
            } else {
                Err(CallError::Trap(Trap::Bounds))
            };
            assert_eq!(
                kernel.call(&[index, len]),
                expected,
                "index={index}, len={len}"
            );
        }
    }
}

fn two_checked_calls(op: Binary) -> Function {
    use Type::{I32, Ptr};
    let mut leaf = Function::new(&[Ptr, I32, I32, I32], I32);
    let b = leaf.entry();
    let [dst, n, divisor, index] = leaf.params(b).try_into().unwrap();
    let one = leaf.append(b, Op::Const(1)).unwrap();
    let two = leaf.append(b, Op::Const(2)).unwrap();
    leaf.append_inst(b, Op::Store(dst, n)).unwrap();
    leaf.append_inst(b, Op::BoundsCheck(index, two)).unwrap();
    let at = leaf
        .append(b, Op::PtrOffset(dst, one, Scale::Four))
        .unwrap();
    leaf.append_inst(b, Op::Store(at, divisor)).unwrap();
    // The quotient/remainder is dead, but the check must still execute here.
    leaf.append_inst(b, Op::Bin(op, n, divisor)).unwrap();
    let at = leaf
        .append(b, Op::PtrOffset(dst, two, Scale::Four))
        .unwrap();
    leaf.append_inst(b, Op::Store(at, index)).unwrap();
    leaf.set_terminator(b, Terminator::Return(index));
    let leaf = Arc::new(leaf);

    let mut root = Function::new(&[Ptr, I32, I32, I32, I32, I32], I32);
    let b = root.entry();
    let [dst, n, d1, i1, d2, i2] = root.params(b).try_into().unwrap();
    let first = root.add_call(Arc::clone(&leaf), &[dst, n, d1, i1]);
    let a = root.append(b, Op::Call(first)).unwrap();
    let three = root.append(b, Op::Const(3)).unwrap();
    let second_dst = root
        .append(b, Op::PtrOffset(dst, three, Scale::Four))
        .unwrap();
    let second = root.add_call(leaf, &[second_dst, n, d2, i2]);
    let c = root.append(b, Op::Call(second)).unwrap();
    let six = root.append(b, Op::Const(6)).unwrap();
    let end = root
        .append(b, Op::PtrOffset(dst, six, Scale::Four))
        .unwrap();
    let marker = root.append(b, Op::Const(99)).unwrap();
    root.append_inst(b, Op::Store(end, marker)).unwrap();
    let sum = root.append(b, Op::Bin(Binary::Add, a, c)).unwrap();
    root.set_terminator(b, Terminator::Return(sum));
    root
}

#[test]
fn first_trap_stops_at_the_exact_store_in_each_callee() {
    use Trap::*;
    // Each invocation can leave a different prefix of the seven writes visible.
    // The last case also checks a successful call after earlier traps on this kernel.
    let cases = [
        ([13, 0, -1, 2, 1], Err(Bounds.into()), 1),
        ([13, 0, 1, 2, -1], Err(DivisionByZero.into()), 2),
        ([i32::MIN, -1, 0, 0, 0], Err(DivisionOverflow.into()), 2),
        ([13, 3, 1, 0, 2], Err(Bounds.into()), 4),
        ([13, 3, 0, 0, 1], Err(DivisionByZero.into()), 5),
        ([i32::MIN, 2, 1, -1, 0], Err(DivisionOverflow.into()), 5),
        ([-17, 3, 0, -5, 1], Ok(1), 7),
    ];
    for op in [Binary::Div, Binary::Rem] {
        let f = two_checked_calls(op);
        for policy in [Inlining::Always, Inlining::Never] {
            let kernel =
                ScalarKernel::compile(&f, &crate::test_target::options().with_inlining(policy))
                    .unwrap();
            for (inputs @ [n, d1, i1, d2, i2], expected_result, writes) in cases.clone() {
                let mut output = [-777; 7];
                let mut args = [0u64; 6];
                args[0] = output.as_mut_ptr() as u64;
                for (slot, input) in args[1..].iter_mut().zip(inputs) {
                    *slot = u64::from(input as u32);
                }
                // SAFETY: all stores use constant indices 0..=6 in this live,
                // exclusively borrowed i32 array, even when a check fails.
                let actual = unsafe { kernel.call_raw(&args) }.map(|v| v as i32);
                let mut expected = [-777; 7];
                expected[..writes].copy_from_slice(&[n, d1, i1, n, d2, i2, 99][..writes]);
                assert_eq!(actual, expected_result, "{op:?}, {policy:?}, {inputs:?}");
                assert_eq!(output, expected, "{op:?}, {policy:?}, {inputs:?}");
            }
        }
    }
}
