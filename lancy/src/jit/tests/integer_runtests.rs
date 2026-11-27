//! Native execution cases in the style of Cranelift's icmp-eq-imm.clif,
//! shift-right-left.clif and div-checks.clif. Expected values follow Lancy's
//! i32 semantics, including all-bits-set masks and trapping signed remainder.
//! https://github.com/bytecodealliance/wasmtime/tree/v45.0.3/cranelift/filetests/filetests/runtests
use crate::hir::{Binary, Function, Op, Terminator};
use crate::{CallError, Kernel, Trap};

// Cross the signed imm8 encoding boundary, the shift-count mask and the sign bit.
const VALUES: &[i32] = &[
    i32::MIN,
    -129,
    -128,
    -33,
    -1,
    0,
    1,
    31,
    32,
    33,
    127,
    128,
    0x55aa_33cc,
    i32::MAX,
];

fn check_immediates(op: Binary, reference: fn(i32, i32) -> Option<i32>) {
    for &constant in VALUES {
        for constant_on_left in [false, true] {
            let mut f = Function::scalar(1);
            let b = f.entry();
            let x = f.params(b)[0];
            let c = f.append(b, Op::Const(constant)).unwrap();
            let (lhs, rhs) = if constant_on_left { (c, x) } else { (x, c) };
            let value = f.append(b, Op::Bin(op, lhs, rhs)).unwrap();
            f.set_terminator(b, Terminator::Return(value));
            let kernel = Kernel::compile(&f, &crate::test_target::options()).unwrap();
            for &input in VALUES {
                let (a, b) = if constant_on_left {
                    (constant, input)
                } else {
                    (input, constant)
                };
                let expected = reference(a, b).ok_or(if b == 0 {
                    CallError::Trap(Trap::DivisionByZero)
                } else {
                    CallError::Trap(Trap::DivisionOverflow)
                });
                assert_eq!(
                    kernel.call(&[input]),
                    expected,
                    "{op:?}({a}, {b}), constant={constant}"
                );
            }
        }
    }
}

macro_rules! immediate_test {
    ($name:ident, $op:ident, $reference:expr) => {
        #[test]
        fn $name() {
            check_immediates(Binary::$op, $reference);
        }
    };
}

immediate_test!(add, Add, |a, b| Some(a.wrapping_add(b)));
immediate_test!(sub, Sub, |a, b| Some(a.wrapping_sub(b)));
immediate_test!(mul, Mul, |a, b| Some(a.wrapping_mul(b)));
immediate_test!(div, Div, i32::checked_div);
immediate_test!(rem, Rem, i32::checked_rem);
immediate_test!(and, And, |a, b| Some(a & b));
immediate_test!(or, Or, |a, b| Some(a | b));
immediate_test!(xor, Xor, |a, b| Some(a ^ b));
immediate_test!(eq, Eq, |a, b| Some(-i32::from(a == b)));
immediate_test!(ne, Ne, |a, b| Some(-i32::from(a != b)));
immediate_test!(lt, Lt, |a, b| Some(-i32::from(a < b)));
immediate_test!(le, Le, |a, b| Some(-i32::from(a <= b)));
immediate_test!(gt, Gt, |a, b| Some(-i32::from(a > b)));
immediate_test!(ge, Ge, |a, b| Some(-i32::from(a >= b)));
immediate_test!(shl, Shl, |a, b| Some(a.wrapping_shl(b as u32)));
immediate_test!(shr, Shr, |a, b| Some(a.wrapping_shr(b as u32)));
immediate_test!(min, Min, |a, b| Some(a.min(b)));
immediate_test!(max, Max, |a, b| Some(a.max(b)));

#[test]
fn right_then_left_shift_clears_low_bits() {
    let mut f = Function::scalar(2);
    let b = f.entry();
    let [x, count] = f.params(b).try_into().unwrap();
    let high = f.append(b, Op::Bin(Binary::Shr, x, count)).unwrap();
    let result = f.append(b, Op::Bin(Binary::Shl, high, count)).unwrap();
    f.set_terminator(b, Terminator::Return(result));
    let kernel = Kernel::compile(&f, &crate::test_target::options()).unwrap();
    for &x in VALUES {
        for count in -65..=65 {
            let expected = x & (-1i32).wrapping_shl(count as u32);
            assert_eq!(
                kernel.call(&[x, count]),
                Ok(expected),
                "x={x}, count={count}"
            );
        }
    }
}
