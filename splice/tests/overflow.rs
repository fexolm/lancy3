use splice::{BuildError, BuildOptions, CallError, Codegen, Executable, Mode, Trap, kernel};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[derive(Clone, Copy, Debug)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Negate,
    LeftShift,
    RightShift,
    AssignLocal,
    AssignElement,
    WrappingAdd,
    WrappingNegate,
    SaturatingAdd,
    SaturatingSubtract,
    SaturatingMultiply,
}

#[kernel]
fn arithmetic(#[comptime] op: Operation, out: &mut [i32], a: i32, b: i32) -> i32 {
    out[0] = 7;
    let result = match op {
        Operation::Add => a + b,
        Operation::Subtract => a - b,
        Operation::Multiply => a * b,
        Operation::Negate => -a,
        Operation::LeftShift => a << b,
        Operation::RightShift => a >> b,
        Operation::AssignLocal => {
            let mut value = a;
            value += b;
            value
        }
        Operation::AssignElement => {
            out[1] += b;
            out[1]
        }
        Operation::WrappingAdd => a.wrapping_add(b),
        Operation::WrappingNegate => a.wrapping_neg(),
        Operation::SaturatingAdd => a.saturating_add(b),
        Operation::SaturatingSubtract => a.saturating_sub(b),
        Operation::SaturatingMultiply => a.saturating_mul(b),
    };
    out[2] = result;
    result
}

#[kernel]
fn plus_one(x: i32) -> i32 {
    x + 1
}

#[kernel]
fn nested(out: &mut [i32], x: i32) -> i32 {
    out[0] = 7;
    let value = plus_one(x);
    out[1] = value;
    value
}

#[kernel]
fn known(#[comptime] a: i32, #[comptime] b: i32, out: &mut [i32], flag: i32) -> i32 {
    out[0] = 7;
    if flag != 0 { a + b } else { 0 }
}

#[kernel]
fn required(#[comptime] a: i32, #[comptime] b: i32, flag: i32) -> i32 {
    if flag != 0 { comptime!(a + b) } else { 0 }
}

fn check_arithmetic<B: Codegen<Output: Executable>>(options: BuildOptions) {
    use Operation::*;
    for (op, overflowing) in [
        (Add, (i32::MAX, 1)),
        (Subtract, (i32::MIN, 1)),
        (Multiply, (i32::MIN, -1)),
        (Negate, (i32::MIN, 0)),
        (LeftShift, (3, 32)),
        (RightShift, (-3, -1)),
        (AssignLocal, (i32::MAX, 1)),
        (AssignElement, (i32::MAX, 1)),
        (WrappingAdd, (i32::MAX, 1)),
        (WrappingNegate, (i32::MIN, 0)),
        (SaturatingAdd, (i32::MAX, 1)),
        (SaturatingSubtract, (i32::MIN, 1)),
        (SaturatingMultiply, (i32::MIN, -1)),
    ] {
        let direct = arithmetic::build_with_options::<B>(options, arithmetic::BuildArgs { op })
            .expect("build arithmetic");
        let ir = arithmetic::ir_with_options(options, op).expect("construct arithmetic IR");
        let portable = B::compile(&ir).expect("compile arithmetic IR");
        drop(ir);

        for (a, b) in [overflowing, (6, 2)] {
            let mut expected_out = [0, a, 0];
            let expected =
                catch_unwind(AssertUnwindSafe(|| arithmetic(op, &mut expected_out, a, b)))
                    .map_err(|_| CallError::Trap(Trap::Overflow));
            for code in [&direct, &portable] {
                let mut out = [0, a, 0];
                let actual = code.call(arithmetic::CallArgs {
                    out: &mut out,
                    a,
                    b,
                });
                assert_eq!(actual, expected, "{op:?} {a}, {b} ({options:?})");
                assert_eq!(out, expected_out, "{op:?} {a}, {b} ({options:?})");
            }
        }
    }

    let direct = nested::build_with_options::<B>(options, nested::BuildArgs {}).unwrap();
    let portable = B::compile(&nested::ir_with_options(options).unwrap()).unwrap();
    let mut expected_out = [0, 0];
    let expected = catch_unwind(AssertUnwindSafe(|| nested(&mut expected_out, i32::MAX)))
        .map_err(|_| CallError::Trap(Trap::Overflow));
    for code in [&direct, &portable] {
        let mut out = [0, 0];
        let actual = code.call(nested::CallArgs {
            out: &mut out,
            x: i32::MAX,
        });
        assert_eq!(actual, expected);
        assert_eq!(out, expected_out);
    }

    let (a, b) = (i32::MAX, 1);
    let direct = known::build_with_options::<B>(options, known::BuildArgs { a, b }).unwrap();
    let portable = B::compile(&known::ir_with_options(options, a, b).unwrap()).unwrap();
    for flag in [0, 1] {
        let mut expected_out = [0];
        let expected = catch_unwind(AssertUnwindSafe(|| known(a, b, &mut expected_out, flag)))
            .map_err(|_| CallError::Trap(Trap::Overflow));
        for code in [&direct, &portable] {
            let mut out = [0];
            let actual = code.call(known::CallArgs {
                out: &mut out,
                flag,
            });
            assert_eq!(actual, expected);
            assert_eq!(out, expected_out);
        }
    }

    let checks = catch_unwind(|| required(a, b, 1)).is_err();
    let overflow = |error: BuildError| matches!(error, BuildError::Comptime(Trap::Overflow));
    assert_eq!(
        required::build_with_options::<B>(options, required::BuildArgs { a, b })
            .err()
            .is_some_and(overflow),
        checks
    );
    assert_eq!(
        required::ir_with_options(options, a, b)
            .err()
            .is_some_and(overflow),
        checks
    );
}

#[test]
fn operator_overflow_respects_the_rust_profile_in_both_compilation_paths() {
    for options in [
        BuildOptions::host(Mode::Release).unwrap(),
        BuildOptions::host(Mode::Debug).unwrap(),
    ] {
        splice_testkit::each_backend!(B => check_arithmetic::<B>(options));
    }
}

// MAC-03: faulting host arithmetic on any integer type is deferred to the
// point where the kernel reaches it, and `comptime!` reports it as an error.
#[kernel]
fn host_div(#[comptime] d: usize, flag: i32) -> i32 {
    if flag != 0 { (12 / d) as i32 } else { 0 }
}

#[kernel]
fn host_wide(#[comptime] a: i64, #[comptime] shift: u32, flag: i32) -> i32 {
    if flag == 1 {
        (a + 1) as i32
    } else if flag == 2 {
        (1u64 << shift) as i32
    } else if flag == 3 {
        (-a) as i32
    } else if flag == 4 {
        (a / -1) as i32
    } else {
        0
    }
}

#[kernel]
fn comptime_wide(#[comptime] a: i64, flag: i32) -> i32 {
    if flag != 0 {
        comptime!((a * 2) as i32)
    } else {
        0
    }
}

thread_local! {
    static PANICS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Run `f` and report whether it panicked on this thread, even if the panic
/// was caught inside `f`. A build must not panic: under `panic = "abort"`
/// that would end the process.
fn panics<T>(f: impl FnOnce() -> T) -> (T, bool) {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            PANICS.with(|n| n.set(n.get() + 1));
            previous(info);
        }));
    });
    let before = PANICS.with(|n| n.get());
    let value = f();
    (value, PANICS.with(|n| n.get()) != before)
}

fn native(f: impl FnOnce() -> i32) -> Result<i32, CallError> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|e| {
        let message = e
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| e.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        if message.contains("divide by zero") {
            CallError::Trap(Trap::DivisionByZero)
        } else if message.contains("divide with overflow") {
            CallError::Trap(Trap::DivisionOverflow)
        } else {
            CallError::Trap(Trap::Overflow)
        }
    })
}

fn check_host_faults<B: Codegen<Output: Executable>>(options: BuildOptions) {
    let (codes, panicked) = panics(|| {
        [
            host_div::build_with_options::<B>(options, host_div::BuildArgs { d: 0 }).unwrap(),
            B::compile(&host_div::ir_with_options(options, 0).unwrap()).unwrap(),
        ]
    });
    assert!(!panicked, "building host_div panicked");
    for code in &codes {
        for flag in [0, 1] {
            let actual = code.call(host_div::CallArgs { flag });
            assert_eq!(actual, native(|| host_div(0, flag)), "host_div flag {flag}");
        }
    }

    for (a, shift) in [(i64::MAX, 70), (i64::MIN, 64), (5, 3)] {
        let (codes, panicked) = panics(|| {
            [
                host_wide::build_with_options::<B>(options, host_wide::BuildArgs { a, shift })
                    .unwrap(),
                B::compile(&host_wide::ir_with_options(options, a, shift).unwrap()).unwrap(),
            ]
        });
        assert!(!panicked, "building host_wide({a}, {shift}) panicked");
        for code in &codes {
            for flag in 0..=4 {
                let actual = code.call(host_wide::CallArgs { flag });
                let expected = native(|| host_wide(a, shift, flag));
                assert_eq!(actual, expected, "host_wide({a}, {shift}) flag {flag}");
            }
        }
    }

    for a in [i64::MAX, 7] {
        let checks = native(|| comptime_wide(a, 1)).is_err();
        let (direct, panicked) = panics(|| {
            comptime_wide::build_with_options::<B>(options, comptime_wide::BuildArgs { a })
        });
        assert!(!panicked, "comptime! panicked for {a}");
        assert_eq!(direct.is_err(), checks, "comptime_wide({a})");
        let (ir, panicked) = panics(|| comptime_wide::ir_with_options(options, a));
        assert!(!panicked, "comptime! panicked for {a}");
        assert_eq!(ir.is_err(), checks, "comptime_wide({a}) ir");
    }
}

#[test]
fn host_arithmetic_faults_are_deferred_for_every_integer_type() {
    for options in [
        BuildOptions::host(Mode::Release).unwrap(),
        BuildOptions::host(Mode::Debug).unwrap(),
    ] {
        splice_testkit::each_backend!(B => check_host_faults::<B>(options));
    }
}

// MAC-15: overflow checks follow the crate that compiles a kernel, as they do
// for its Rust function, and `BuildOptions::with_overflow_checks` overrides
// them for kernel code. The fixture crate checks overflow exactly when this
// crate does not (see the workspace profiles).
#[kernel]
fn across_crates(a: i32, b: i32, c: i32) -> i32 {
    splice_test_overflow::add(a, b) - c
}

fn check_crate_overflow<B: Codegen<Output: Executable>>(mode: Mode) {
    use splice_test_overflow::{add, add_known};
    let here = cfg!(debug_assertions);
    let fixture = !here;
    let wraps = |checks: bool, value: i32| {
        if checks {
            Err(CallError::Trap(Trap::Overflow))
        } else {
            Ok(value)
        }
    };
    let options = BuildOptions::host(mode).unwrap();
    for options in [
        options,
        options.with_overflow_checks(false),
        options.with_overflow_checks(true),
    ] {
        let kernel = options.overflow_checks;
        let codes = [
            add::build_with_options::<B>(options, add::BuildArgs {}).unwrap(),
            B::compile(&add::ir_with_options(options).unwrap()).unwrap(),
        ];
        for code in &codes {
            let actual = code.call(add::CallArgs { a: i32::MAX, b: 1 });
            assert_eq!(
                actual,
                wraps(kernel.unwrap_or(fixture), i32::MIN),
                "{options:?}"
            );
        }
        // Staging arithmetic is native Rust of the fixture crate.
        let codes = [
            add_known::build_with_options::<B>(options, add_known::BuildArgs { a: i32::MAX })
                .unwrap(),
            B::compile(&add_known::ir_with_options(options, i32::MAX).unwrap()).unwrap(),
        ];
        for code in &codes {
            let actual = code.call(add_known::CallArgs { flag: true });
            assert_eq!(actual, wraps(fixture, i32::MIN), "{options:?}");
            assert_eq!(code.call(add_known::CallArgs { flag: false }), Ok(0));
        }
        let codes = [
            across_crates::build_with_options::<B>(options, across_crates::BuildArgs {}).unwrap(),
            B::compile(&across_crates::ir_with_options(options).unwrap()).unwrap(),
        ];
        for code in &codes {
            let call = |a, b, c| code.call(across_crates::CallArgs { a, b, c });
            assert_eq!(
                call(i32::MAX, 1, 0),
                wraps(kernel.unwrap_or(fixture), i32::MIN),
                "{options:?}"
            );
            assert_eq!(
                call(0, 0, i32::MIN),
                wraps(kernel.unwrap_or(here), i32::MIN),
                "{options:?}"
            );
        }
    }
}

#[test]
fn kernels_follow_the_overflow_checks_of_their_crate() {
    for mode in [Mode::Release, Mode::Debug] {
        splice_testkit::each_backend!(B => check_crate_overflow::<B>(mode));
    }
}
