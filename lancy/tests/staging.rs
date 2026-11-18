use lancy::input::{Binary, Function, Instr};
use lancy::jit::ScalarFn;
use lancy::kernel;

#[derive(Clone, Copy, Debug)]
enum Curve {
    Recurrence { seed: i32, steps: i32 },
    Fixed(i32),
    Forbidden,
}

// This is an ordinary Rust function without a kernel entry point.
const fn next_coefficient(value: i32, step: i32) -> i32 {
    value * 3 - step
}

fn forbidden_coefficient() -> i32 {
    panic!("evaluated an unselected compile-time arm")
}

fn nonzero_offset(key: i32) -> i32 {
    assert_ne!(key, 0, "evaluated host code after a selected early return");
    42 / key
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
            Curve::Forbidden => forbidden_coefficient(),
        }
    } else {
        1
    };
    input + coefficient
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
    assert_eq!(constants, [expected_offset]);
    assert_eq!(arithmetic, [Binary::Add]);
}

#[test]
fn host_control_and_helpers_fold_during_expansion() {
    // All offsets are nonzero to keep the runtime addition in optimized IR.
    for (enabled, curve, expected_offset) in [
        (true, Curve::Recurrence { seed: 2, steps: 4 }, 144),
        (true, Curve::Recurrence { seed: -5, steps: 0 }, -5),
        (true, Curve::Fixed(7), 20),
        // This would panic if the host if evaluated its dead arm.
        (false, Curve::Forbidden, 1),
    ] {
        let raw = offset::build(enabled, curve).unwrap();
        let simplified = raw.simplified().unwrap();
        check_function(&raw, expected_offset);
        check_function(&simplified, expected_offset);
        let native = ScalarFn::<1>::compile(&simplified).unwrap();
        for x in [-11, 0, 24] {
            let ordinary = offset(enabled, curve, x);
            assert_eq!(ordinary, x + expected_offset);
            assert_eq!(raw.interpret(&[x]).unwrap(), ordinary);
            assert_eq!(native.invoke([x]), ordinary);
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
    let raw = accumulate::build(23).unwrap();
    let simplified = raw.simplified().unwrap();
    for stage in [&raw, &simplified] {
        // The input-dependent loop remains reachable in both representations.
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
    }
    let native = ScalarFn::<2>::compile(&simplified).unwrap();
    for enabled in [-1, 0, 1] {
        for size in [-2, 0, 1, 6] {
            let ordinary = accumulate(23, enabled, size);
            let expected = (if enabled > 0 { 23 } else { 7 }) + (0..size).sum::<i32>();
            assert_eq!(ordinary, expected);
            assert_eq!(raw.interpret(&[enabled, size]).unwrap(), expected);
            assert_eq!(native.invoke([enabled, size]), expected);
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
    x + nonzero_offset(key)
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
fn same_const_kernel_runs_as_rust_at_build_time_and_as_a_native_callee_at_runtime() {
    let raw = evaluate_both::build(4).unwrap();
    let simplified = raw.simplified().unwrap();
    for stage in [&raw, &simplified] {
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
                .any(|i| matches!(i, Instr::Const(_, KNOWN_WEIGHT)))
        );
        assert!(
            calls[0]
                .instructions()
                .iter()
                .any(|i| matches!(i, Instr::Bin(_, Binary::Mul, ..)))
        );
    }
    let native = ScalarFn::<1>::compile(&raw).unwrap();
    for x in [-7, 0, 11] {
        assert_eq!(native.invoke([x]), evaluate_both(4, x));
    }
}

#[test]
fn early_host_exit_skips_the_dead_suffix_but_preserves_sibling_runtime_paths() {
    for key in [0, 3] {
        let ir = early::build(key).unwrap();
        if key == 0 {
            assert!(
                !ir.instructions()
                    .iter()
                    .any(|i| matches!(i, Instr::Bin(_, Binary::Add, ..)))
            );
        }
        let native = ScalarFn::<1>::compile(&ir).unwrap();
        for x in [-6, 0, 11] {
            assert_eq!(native.invoke([x]), early(key, x));
        }
    }

    for bypass in [false, true] {
        let ir = early_inside_runtime_arm::build(bypass).unwrap();
        let native = ScalarFn::<1>::compile(&ir).unwrap();
        for x in [-4, 0, 20] {
            assert_eq!(native.invoke([x]), early_inside_runtime_arm(bypass, x));
        }
    }
}

#[kernel]
fn affine(#[comptime] subtract: bool, x: i32, bias: i32) -> i32 {
    if subtract { x - bias } else { x + bias }
}

// A user-defined Rust helper can use this name like any other identifier.
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
    let f = apply_affine::build(4).unwrap();
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
        code.iter()
            .any(|op| matches!(op, Instr::Const(r, 13) if *r == args[1]))
    );
    assert!(!code.iter().any(|op| matches!(op, Instr::Bin(..))));
    let jit = ScalarFn::<1>::compile(&f).unwrap();
    for x in [-7, 0, 5] {
        assert_eq!(jit.invoke([x]), apply_affine(4, x));
    }
}

#[kernel]
fn encode(mode: i32, x: i32) -> i32 {
    if mode == 0 { x + 1 } else { x * 2 }
}

#[kernel]
fn configured_encoder(#[comptime] mode: i32, x: i32) -> i32 {
    encode(mode, x)
}

#[test]
fn an_ordinary_parameter_specializes_the_control_flow_of_a_called_kernel() {
    fn callee(f: &Function) -> &Function {
        f.instructions()
            .iter()
            .find_map(|inst| match inst {
                Instr::Call { function, .. } => Some(function.as_ref()),
                _ => None,
            })
            .unwrap()
    }
    for mode in [0, 1] {
        let f = configured_encoder::build(mode).unwrap();
        assert!(
            callee(&f)
                .instructions()
                .iter()
                .any(|i| matches!(i, Instr::Branch { .. }))
        );
        let simplified = f.simplified().unwrap();
        assert!(
            !callee(&simplified)
                .instructions()
                .iter()
                .any(|i| matches!(i, Instr::Branch { .. }))
        );
        let native = ScalarFn::<1>::compile(&f).unwrap();
        for x in [-13, 0, 31] {
            assert_eq!(native.invoke([x]), encode(mode, x));
        }
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
    let ir = visit::build(&plan).unwrap();
    let jit = ScalarFn::<2>::compile(&ir).unwrap();
    for x in [-8, 0, 9] {
        assert_eq!(jit.invoke([x, 900]), visit(&plan, x, 900));
    }
}

#[kernel]
fn gated_offset(#[comptime] key: i32, x: i32) -> i32 {
    if key != 0 && x > 0 {
        x + nonzero_offset(key)
    } else {
        x
    }
}

#[test]
fn known_false_short_circuit_skips_expansion_of_the_outer_body() {
    let disabled = gated_offset::build(0).unwrap();
    assert!(
        !disabled
            .instructions()
            .iter()
            .any(|op| matches!(op, Instr::Branch { .. }))
    );
    for key in [0, 3] {
        let native = ScalarFn::<1>::compile(&gated_offset::build(key).unwrap()).unwrap();
        for x in [-2, 7] {
            assert_eq!(native.invoke([x]), gated_offset(key, x));
        }
    }
}
