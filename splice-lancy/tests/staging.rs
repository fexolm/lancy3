use splice::kernel;
use splice_core::hir::{Binary, Constant, Function, Instr};
use splice_lancy::Lancy;

#[derive(Clone, Copy, Debug)]
enum Curve {
    Recurrence { seed: i32, steps: i32 },
    Fixed(i32),
    Forbidden,
}

#[kernel]
const fn next_coefficient(value: i32, step: i32) -> i32 {
    value * 3 - step
}

fn forbidden_coefficient() -> i32 {
    panic!("evaluated an unselected compile-time arm")
}

#[kernel]
fn offset(#[comptime] enabled: bool, #[comptime] curve: Curve, input: i32) -> i32 {
    let coefficient = if enabled {
        match curve {
            Curve::Recurrence { seed, steps } => {
                let mut i = 0;
                let mut value = seed;
                while i < steps {
                    value = next_coefficient(value, i);
                    i += 1;
                }
                value
            }
            Curve::Fixed(value) => next_coefficient(value, 1),
            Curve::Forbidden => comptime!(forbidden_coefficient()),
        }
    } else {
        1
    };
    input + coefficient
}

/// `ir()` takes the default build options, which check overflow in debug.
fn checked(release: Binary, debug: Binary) -> Binary {
    if cfg!(debug_assertions) {
        debug
    } else {
        release
    }
}

fn check_function(ir: &Function, expected_offset: i32) {
    // Apart from moves, labels and a return, the entire specialization
    // must be one embedded coefficient and a single runtime addition.
    let mut constants = Vec::new();
    let mut arithmetic = Vec::new();
    for instr in ir.instructions() {
        match instr {
            Instr::Const(_, n) => constants.push(*n),
            Instr::Bin(_, op, _, _) => arithmetic.push(*op),
            Instr::Branch { .. } | Instr::Call { .. } => {
                panic!("host control flow or host helper leaked into the IR: {instr:?}")
            }
            _ => {}
        }
    }
    assert_eq!(constants, [Constant::from(expected_offset)]);
    assert_eq!(arithmetic, [checked(Binary::Add, Binary::CheckedAdd)]);
}

#[test]
fn host_control_and_helpers_fold_during_expansion() {
    for (enabled, curve, expected_offset) in [
        (true, Curve::Recurrence { seed: 2, steps: 4 }, 144),
        (true, Curve::Recurrence { seed: -5, steps: 0 }, -5),
        (true, Curve::Fixed(7), 20),
        // This would panic if the host if evaluated its dead arm.
        (false, Curve::Forbidden, 1),
    ] {
        let raw = offset::ir(enabled, curve).unwrap();
        check_function(&raw, expected_offset);
        let native = offset::build::<Lancy>(offset::BuildArgs { enabled, curve }).unwrap();
        for x in [-11, 0, 24] {
            let ordinary = offset(enabled, curve, x);
            assert_eq!(ordinary, x + expected_offset);
            assert_eq!(
                native.call(offset::CallArgs { input: x }).unwrap(),
                ordinary
            );
        }
    }
}

#[kernel]
fn accumulate(#[comptime] bias: i32, enabled: i32, size: i32) -> i32 {
    let mut score = 7;
    // Both assigned values are host-known. The input decides which wins.
    if enabled > 0 {
        score = bias;
    }
    let mut i = 0;
    while i < size {
        score += i;
        i += 1;
    }
    score
}

#[test]
fn runtime_control_survives_and_promotes_literal_initialized_locals() {
    let stage = accumulate::ir(23).unwrap();
    assert!(
        stage
            .instructions()
            .iter()
            .any(|i| matches!(i, Instr::Branch { .. }))
    );
    assert!(
        !stage
            .instructions()
            .iter()
            .any(|i| matches!(i, Instr::Call { .. }))
    );
    let native = accumulate::build::<Lancy>(accumulate::BuildArgs { bias: 23 }).unwrap();
    for enabled in [-1, 0, 1] {
        for size in [-2, 0, 1, 6] {
            let ordinary = accumulate(23, enabled, size);
            let expected = (if enabled > 0 { 23 } else { 7 }) + (0..size).sum::<i32>();
            assert_eq!(ordinary, expected);
            assert_eq!(
                native.call(accumulate::CallArgs { enabled, size }).unwrap(),
                expected
            );
        }
    }
}

#[kernel]
fn early(#[comptime] key: i32, x: i32) -> i32 {
    if x < 0 {
        return -17;
    }
    if key == 0 {
        return x;
    }
    x + 42 / key
}

#[kernel]
#[allow(
    clippy::collapsible_if,
    reason = "exercise a host return nested under a runtime branch"
)]
fn early_inside_runtime_arm(#[comptime] bypass: bool, x: i32) -> i32 {
    if x < 0 {
        if bypass {
            return -31;
        }
    }
    x + 1
}

#[kernel]
const fn polynomial(x: i32) -> i32 {
    x * x - 1
}

const KNOWN_WEIGHT: i32 = polynomial(4);

#[kernel]
fn evaluate_both(#[comptime] known: i32, x: i32) -> i32 {
    polynomial(known) + polynomial(x)
}

#[test]
fn a_const_kernel_can_be_evaluated_and_called_in_the_same_specialization() {
    let stage = evaluate_both::ir(4).unwrap();
    let calls = stage
        .instructions()
        .iter()
        .filter_map(|i| match i {
            Instr::Call { function, .. } => Some(function),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert!(
        stage
            .instructions()
            .iter()
            .any(|i| matches!(i, Instr::Const(_, c) if *c == Constant::from(KNOWN_WEIGHT)))
    );
    assert!(calls[0].instructions().iter().any(
        |i| matches!(i, Instr::Bin(_, op, ..) if *op == checked(Binary::Mul, Binary::CheckedMul))
    ));
    let native = evaluate_both::build::<Lancy>(evaluate_both::BuildArgs { known: 4 }).unwrap();
    for x in [-7, 0, 11] {
        assert_eq!(
            native.call(evaluate_both::CallArgs { x }).unwrap(),
            evaluate_both(4, x)
        );
    }
}

#[test]
fn early_host_exit_skips_the_dead_suffix_but_preserves_sibling_runtime_paths() {
    for key in [0, 3] {
        let ir = early::ir(key).unwrap();
        if key == 0 {
            assert!(
                !ir.instructions()
                    .iter()
                    .any(|i| matches!(i, Instr::Bin(_, Binary::Add, ..)))
            );
        }
        let native = early::build::<Lancy>(early::BuildArgs { key }).unwrap();
        for x in [-6, 0, 11] {
            assert_eq!(native.call(early::CallArgs { x }).unwrap(), early(key, x));
        }
    }

    for bypass in [false, true] {
        let native =
            early_inside_runtime_arm::build::<Lancy>(early_inside_runtime_arm::BuildArgs {
                bypass,
            })
            .unwrap();
        for x in [-4, 0, 20] {
            assert_eq!(
                native
                    .call(early_inside_runtime_arm::CallArgs { x })
                    .unwrap(),
                early_inside_runtime_arm(bypass, x)
            );
        }
    }
}

#[kernel]
fn affine(#[comptime] subtract: bool, x: i32, bias: i32) -> i32 {
    if subtract { x - bias } else { x + bias }
}

// A user-defined Rust helper can use this name like any other identifier.
#[kernel]
const fn runtime(x: i32) -> i32 {
    x * 3
}

#[kernel]
fn apply_affine(#[comptime] base: i32, x: i32) -> i32 {
    let bias = runtime(base) + 1;
    affine(false, x, bias)
}

#[test]
fn a_known_i32_is_a_plain_constant_argument_of_a_runtime_call() {
    let f = apply_affine::ir(4).unwrap();
    assert_eq!(f.arity(), 1);
    let code = f.instructions();
    let (args, callee) = code
        .iter()
        .find_map(|op| match op {
            Instr::Call { args, function, .. } => Some((args, function)),
            _ => None,
        })
        .unwrap();
    assert_eq!(callee.arity(), 2);
    assert!(
        code.iter().any(
            |op| matches!(op, Instr::Const(r, c) if *r == args[1] && *c == Constant::from(13))
        )
    );
    assert!(!code.iter().any(|op| matches!(op, Instr::Bin(..))));
    let jit = apply_affine::build::<Lancy>(apply_affine::BuildArgs { base: 4 }).unwrap();
    for x in [-7, 0, 5] {
        assert_eq!(
            jit.call(apply_affine::CallArgs { x }).unwrap(),
            apply_affine(4, x)
        );
    }
}

#[derive(Clone, Copy)]
enum Chain<'a> {
    Leaf,
    Reset(i32, &'a Chain<'a>),
}

#[kernel]
fn visit(#[comptime] plan: &Chain<'_>, x: i32, bias: i32) -> i32 {
    match plan {
        Chain::Reset(value, next) => visit(next, x, *value),
        Chain::Leaf => x - bias,
    }
}

#[test]
fn recursive_specialization_accepts_a_constant_in_an_ordinary_slot() {
    let plan = Chain::Reset(7, &Chain::Leaf);
    let jit = visit::build::<Lancy>(visit::BuildArgs { plan: &plan }).unwrap();
    for x in [-8, 0, 9] {
        assert_eq!(
            jit.call(visit::CallArgs { x, bias: 900 }).unwrap(),
            visit(&plan, x, 900)
        );
    }
}

#[kernel]
fn gated_offset(#[comptime] key: i32, x: i32) -> i32 {
    if key != 0 && x > 0 { x + 42 / key } else { x }
}

#[test]
fn known_false_short_circuit_skips_expansion_of_the_outer_body() {
    let disabled = gated_offset::ir(0).unwrap();
    assert!(
        !disabled
            .instructions()
            .iter()
            .any(|op| matches!(op, Instr::Branch { .. }))
    );
    for key in [0, 3] {
        let native = gated_offset::build::<Lancy>(gated_offset::BuildArgs { key }).unwrap();
        for x in [-2, 7] {
            assert_eq!(
                native.call(gated_offset::CallArgs { x }).unwrap(),
                gated_offset(key, x)
            );
        }
    }
}

/// A crate that reaches Splice only as `splice_core` names it for both derives.
#[derive(splice::KernelType)]
struct Gain {
    factor: i32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, splice::ComptimeKey)]
enum Sign {
    Plus,
    Minus,
}

#[kernel]
fn gain(#[comptime] gain: &Gain, #[comptime] sign: Sign, x: i32) -> i32 {
    match sign {
        Sign::Plus => comptime!(gain.factor) * x,
        Sign::Minus => -comptime!(gain.factor) * x,
    }
}

#[test]
fn derives_take_the_crate_path() {
    let g = Gain { factor: 3 };
    for sign in [Sign::Plus, Sign::Minus] {
        let code = gain::build::<Lancy>(gain::BuildArgs { gain: &g, sign }).unwrap();
        assert_eq!(code.call(gain::CallArgs { x: 5 }), Ok(gain(&g, sign, 5)));
    }
}
