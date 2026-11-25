use splice::simd::Lane;
use splice::{BuildError, Codegen, Executable, kernel};
use splice_lancy::Lancy;
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

mod formulas {
    use splice::kernel;

    #[kernel]
    pub(super) fn affine(x: i32, #[comptime] scale: i32, y: i32, #[comptime] bias: i32) -> i32 {
        x * scale + y + bias
    }
}

#[test]
fn named_arguments_preserve_field_values_and_rust_evaluation_order() {
    use formulas::affine;

    fn check<B: Codegen<Output: Executable>>() {
        let mut order = Vec::new();
        let direct = affine::build::<B>(affine::BuildArgs {
            bias: {
                order.push("bias");
                1
            },
            scale: {
                order.push("scale");
                3
            },
        })
        .unwrap();
        assert_eq!(order, ["bias", "scale"]);

        let ir = affine::ir(3, 1).unwrap();
        let portable = B::compile(&ir).unwrap();
        for code in [direct, portable] {
            order.clear();
            let result = code.call(affine::CallArgs {
                y: {
                    order.push("y");
                    4
                },
                x: {
                    order.push("x");
                    2
                },
            });
            assert_eq!(result.unwrap(), affine(2, 3, 4, 1));
            assert_eq!(order, ["y", "x"]);
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
#[expect(
    clippy::redundant_pattern,
    reason = "exercise the name of an @ parameter pattern"
)]
fn patterns(#[comptime] (a, b): (i32, i32), _: i32, mut arg1: i32, x @ _: i32) -> i32 {
    arg1 += a;
    arg1 * b + x
}

#[test]
fn argument_patterns_keep_signature_indices_and_explicit_names() {
    fn check<B: Codegen<Output: Executable>>() {
        let direct = patterns::build::<B>(patterns::BuildArgs { arg1_: (2, 3) }).unwrap();
        let ir = patterns::ir((2, 3)).unwrap();
        let portable = B::compile(&ir).unwrap();
        for code in [direct, portable] {
            let actual = code.call(patterns::CallArgs {
                arg1: 4,
                arg2: 99,
                x: 1,
            });
            assert_eq!(actual.unwrap(), patterns((2, 3), 99, 4, 1));
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

struct Plan {
    min: i32,
}

#[derive(splice::KernelType)]
struct ProjectedPair {
    first: i32,
    second: i32,
}

#[kernel]
fn project(#[comptime] require_known: bool, #[comptime] first: i32, second: i32) -> i32 {
    let pair = ProjectedPair { first, second };
    if require_known {
        comptime!(pair.first)
    } else {
        pair.first - pair.second
    }
}

#[kernel]
fn project_known() -> i32 {
    comptime!(project(true, 9, 4))
}

#[test]
fn fields_inherit_the_stage_of_the_constructed_aggregate() {
    fn check<B: Codegen<Output: Executable>>() {
        assert!(project::ir(true, 9).is_err());
        let known = B::compile(&project_known::ir().unwrap()).unwrap();
        assert_eq!(known.call(project_known::CallArgs {}).unwrap(), 9);
        assert!(
            project::build::<B>(project::BuildArgs {
                require_known: true,
                first: 9
            })
            .is_err()
        );
        let portable = B::compile(&project::ir(false, 9).unwrap()).unwrap();
        let direct = project::build::<B>(project::BuildArgs {
            require_known: false,
            first: 9,
        })
        .unwrap();
        for code in [direct, portable] {
            assert_eq!(
                code.call(project::CallArgs { second: 4 }).unwrap(),
                project(false, 9, 4)
            );
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
fn floor(#[comptime] plan: &Plan, value: i32) -> i32 {
    let threshold = comptime!({
        let Plan { min } = plan;
        *min
    });
    if value < threshold { threshold } else { value }
}

#[test]
fn portable_and_direct_specializations_outlive_the_plan_and_its_changes() {
    fn check<B: Codegen<Output: Executable>>() {
        let (older, newer, ir) = {
            let mut plan = Plan { min: 7 };
            let older = floor::build::<B>(floor::BuildArgs { plan: &plan }).unwrap();
            plan.min = 9;
            let newer = floor::build::<B>(floor::BuildArgs { plan: &plan }).unwrap();
            let ir = floor::ir(&plan).unwrap();
            (older, newer, ir)
        };
        let portable = B::compile(&ir).unwrap();
        drop(ir);
        for (code, min) in [(older, 7), (newer, 9), (portable, 9)] {
            for value in [3, 11] {
                let actual = code.call(floor::CallArgs { value }).unwrap();
                assert_eq!(actual, floor(&Plan { min }, value));
            }
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

static BUILD_EFFECTS: AtomicUsize = AtomicUsize::new(0);

struct DropProbe(i32);

impl Drop for DropProbe {
    fn drop(&mut self) {
        BUILD_EFFECTS.fetch_add(1, Ordering::Relaxed);
    }
}

const OWNED_PROBE: DropProbe = DropProbe(10);

#[kernel]
fn destructure_owner(runtime: i32) -> i32 {
    comptime!({
        let DropProbe(value) = OWNED_PROBE;
        value + runtime
    })
}

fn record(value: i32) -> i32 {
    BUILD_EFFECTS.fetch_add(1, Ordering::Relaxed);
    value
}

#[kernel]
fn opaque_requirement(#[comptime] value: i32) -> i32 {
    comptime!(record(value))
}

#[kernel]
fn ordinary_call(#[comptime] value: i32) -> i32 {
    record(value)
}

fn assert_build_error<T>(result: Result<T, BuildError>) {
    let error = result.err().expect("expected specialization to fail");
    let _: &dyn std::error::Error = &error;
}

#[test]
fn opaque_rust_effects_run_only_during_call() {
    assert_build_error(destructure_owner::ir());
    assert_build_error(destructure_owner::build::<Lancy>(
        destructure_owner::BuildArgs {},
    ));
    assert_build_error(opaque_requirement::ir(7));
    splice_testkit::each_backend!(B => assert_build_error(opaque_requirement::build::<B>(
        opaque_requirement::BuildArgs { value: 7 },
    )));
    assert_eq!(BUILD_EFFECTS.load(Ordering::Relaxed), 0);

    fn check<B: Codegen<Output: Executable>>() {
        let before = BUILD_EFFECTS.load(Ordering::Relaxed);
        let direct = ordinary_call::build::<B>(ordinary_call::BuildArgs { value: 7 }).unwrap();
        let portable = B::compile(&ordinary_call::ir(7).unwrap()).unwrap();
        assert_eq!(BUILD_EFFECTS.load(Ordering::Relaxed), before);
        for code in [direct, portable] {
            for _ in 0..2 {
                assert_eq!(code.call(ordinary_call::CallArgs {}).unwrap(), 7);
            }
        }
        assert_eq!(BUILD_EFFECTS.load(Ordering::Relaxed), before + 4);
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
fn require_division(#[comptime] divisor: i32, flag: i32) -> i32 {
    if flag != 0 {
        comptime!(12 / divisor)
    } else {
        0
    }
}

#[kernel]
fn guarded_division(#[comptime] divisor: i32, flag: i32, output: &mut [i32]) -> i32 {
    output[0] = 7;
    if flag != 0 { 12 / divisor } else { 0 }
}

#[kernel]
fn require_lane_count() -> i32 {
    comptime!(Lane::<i32>::lanes() as i32)
}

#[kernel]
fn select(#[comptime] read: bool, value: i32) -> i32 {
    if read {
        comptime!(value.wrapping_mul(0))
    } else {
        7
    }
}

#[test]
fn specialization_checks_requirements_in_reachable_branches() {
    fn check<B: Codegen<Output: Executable>>() {
        assert_build_error(require_division::ir(0));
        assert_build_error(require_division::build::<B>(require_division::BuildArgs {
            divisor: 0,
        }));
        let ir = require_division::ir(3).unwrap();
        let direct =
            require_division::build::<B>(require_division::BuildArgs { divisor: 3 }).unwrap();
        for code in [direct, B::compile(&ir).unwrap()] {
            assert_eq!(
                code.call(require_division::CallArgs { flag: 1 }).unwrap(),
                4
            );
            assert_eq!(
                code.call(require_division::CallArgs { flag: 0 }).unwrap(),
                0
            );
        }

        assert_build_error(select::ir(true));
        assert_build_error(select::build::<B>(select::BuildArgs { read: true }));
        let direct = select::build::<B>(select::BuildArgs { read: false }).unwrap();
        let portable = B::compile(&select::ir(false).unwrap()).unwrap();
        for code in [direct, portable] {
            assert_eq!(code.call(select::CallArgs { value: 99 }).unwrap(), 7);
        }

        assert_build_error(require_lane_count::ir());
        assert_build_error(require_lane_count::build::<B>(
            require_lane_count::BuildArgs {},
        ));
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[test]
fn ordinary_division_fails_only_when_reached_and_keeps_prior_writes() {
    fn check<B: Codegen<Output: Executable>>() {
        let ir = guarded_division::ir(0).unwrap();
        let direct =
            guarded_division::build::<B>(guarded_division::BuildArgs { divisor: 0 }).unwrap();
        for code in [direct, B::compile(&ir).unwrap()] {
            let mut output = [0, 11];
            let error: splice::CallError = code
                .call(guarded_division::CallArgs {
                    flag: 1,
                    output: &mut output,
                })
                .unwrap_err();
            let _: &dyn std::error::Error = &error;
            assert_eq!(output, [7, 11]);

            output[0] = 0;
            let value = code
                .call(guarded_division::CallArgs {
                    flag: 0,
                    output: &mut output,
                })
                .unwrap();
            assert_eq!(value, 0);
            assert_eq!(output, [7, 11]);
            assert!(
                code.call(guarded_division::CallArgs {
                    flag: 0,
                    output: &mut []
                })
                .is_err()
            );
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

mod effects {
    use super::{AtomicUsize, Ordering};

    pub(super) static TICKS: AtomicUsize = AtomicUsize::new(0);

    pub(super) fn tick(value: i32) -> i32 {
        TICKS.fetch_add(1, Ordering::Relaxed);
        assert!(value >= 0, "negative tick");
        value
    }
}

#[kernel]
fn guarded_tick(enabled: i32, value: i32, output: &mut [i32]) -> i32 {
    output[0] = value;
    if enabled != 0 {
        effects::tick(value);
        output[1] = value;
    }
    7
}

#[test]
fn a_runtime_effect_follows_control_and_stores_even_if_its_result_is_unused() {
    use effects::TICKS;

    fn check<B: Codegen<Output: Executable>>() {
        let before = TICKS.load(Ordering::Relaxed);
        let ir = guarded_tick::ir().unwrap();
        let direct = guarded_tick::build::<B>(guarded_tick::BuildArgs {}).unwrap();
        let portable = B::compile(&ir).unwrap();
        assert_eq!(TICKS.load(Ordering::Relaxed), before);

        for code in [direct, portable] {
            let before = TICKS.load(Ordering::Relaxed);
            let call = |enabled, value, output: &mut [i32]| {
                code.call(guarded_tick::CallArgs {
                    enabled,
                    value,
                    output,
                })
            };
            assert!(call(1, 1, &mut []).is_err());
            let mut output = [0, 19];
            assert_eq!(call(0, -2, &mut output).unwrap(), 7);
            assert_eq!(output, [-2, 19]);
            assert_eq!(TICKS.load(Ordering::Relaxed), before);

            assert!(call(1, -3, &mut output).is_err());
            assert_eq!(output, [-3, 19]);
            assert_eq!(TICKS.load(Ordering::Relaxed), before + 1);

            assert_eq!(call(1, 4, &mut output).unwrap(), 7);
            assert_eq!(output, [4, 4]);
            assert_eq!(TICKS.load(Ordering::Relaxed), before + 2);
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
fn first_value(value: i32, _: i32) -> i32 {
    value
}

#[kernel]
fn discard_bound_runtime_value(value: i32) -> i32 {
    comptime!(first_value(3, value))
}

static NESTED_ARGUMENT_EVALS: AtomicUsize = AtomicUsize::new(0);

fn nested_effect(value: i32) -> i32 {
    NESTED_ARGUMENT_EVALS.fetch_add(1, Ordering::Relaxed);
    value
}

#[kernel]
fn discard_effectful_runtime_value(value: i32) -> i32 {
    comptime!(first_value(3, nested_effect(value)))
}

#[test]
fn mandatory_nested_calls_distinguish_a_bound_input_from_an_effectful_argument() {
    fn check<B: Codegen<Output: Executable>>() {
        assert_build_error(discard_effectful_runtime_value::ir());
        assert_build_error(discard_effectful_runtime_value::build::<B>(
            discard_effectful_runtime_value::BuildArgs {},
        ));

        let ir = discard_bound_runtime_value::ir().unwrap();
        let direct =
            discard_bound_runtime_value::build::<B>(discard_bound_runtime_value::BuildArgs {})
                .unwrap();
        for code in [direct, B::compile(&ir).unwrap()] {
            for value in [-5, 12] {
                assert_eq!(
                    code.call(discard_bound_runtime_value::CallArgs { value })
                        .unwrap(),
                    3
                );
            }
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
    assert_eq!(NESTED_ARGUMENT_EVALS.load(Ordering::Relaxed), 0);
}

#[kernel]
fn known_inside(value: i32) -> i32 {
    comptime!(value)
}

#[kernel]
fn choose_input(#[comptime] use_runtime: bool, value: i32) -> i32 {
    if use_runtime {
        known_inside(value)
    } else {
        known_inside(3) + value
    }
}

#[test]
fn a_nested_callee_inherits_the_stage_of_its_input() {
    fn check<B: Codegen<Output: Executable>>() {
        assert_build_error(known_inside::ir());
        assert_build_error(known_inside::build::<B>(known_inside::BuildArgs {}));
        assert_build_error(choose_input::ir(true));
        assert_build_error(choose_input::build::<B>(choose_input::BuildArgs {
            use_runtime: true,
        }));

        let ir = choose_input::ir(false).unwrap();
        let direct =
            choose_input::build::<B>(choose_input::BuildArgs { use_runtime: false }).unwrap();
        for code in [direct, B::compile(&ir).unwrap()] {
            assert_eq!(
                code.call(choose_input::CallArgs { value: 9 }).unwrap(),
                choose_input(false, 9)
            );
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
fn recursive_argument(#[comptime] depth: i32, values: &[i32]) -> i32 {
    if depth == 0 {
        1
    } else {
        recursive_argument(values[0], values)
    }
}

#[test]
fn runtime_argument_to_recursive_comptime_parameter_fails_only_if_reached() {
    fn check<B: Codegen<Output: Executable>>() {
        assert_build_error(recursive_argument::ir(1));
        assert_build_error(recursive_argument::build::<B>(
            recursive_argument::BuildArgs { depth: 1 },
        ));

        let ir = recursive_argument::ir(0).unwrap();
        let direct =
            recursive_argument::build::<B>(recursive_argument::BuildArgs { depth: 0 }).unwrap();
        for code in [direct, B::compile(&ir).unwrap()] {
            assert_eq!(
                code.call(recursive_argument::CallArgs { values: &[] })
                    .unwrap(),
                1
            );
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel(rust)]
fn first_byte(bytes: &[u8]) -> Option<&u8> {
    bytes.first()
}

#[test]
fn a_rust_result_can_borrow_the_callers_buffer_after_callargs_is_consumed() {
    macro_rules! check {
        ($backend:ty) => {{
            let direct = first_byte::build::<$backend>(first_byte::BuildArgs {}).unwrap();
            let ir = first_byte::ir().unwrap();
            let portable = <$backend>::compile(&ir).unwrap();
            drop(ir);
            for code in [direct, portable] {
                let bytes = [14, 27];
                let first = code
                    .call(first_byte::CallArgs { bytes: &bytes })
                    .unwrap()
                    .unwrap();
                assert!(std::ptr::eq(first, &bytes[0]));
                assert_eq!(
                    code.call(first_byte::CallArgs { bytes: &[] }).unwrap(),
                    None
                );
            }
        }};
    }
    splice_testkit::each_backend!(B => check!(B));
}

struct Capture {
    value: i32,
    drops: Arc<AtomicUsize>,
}

impl Capture {
    fn value(&self) -> i32 {
        self.value
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[kernel(rust)]
fn captured_value(#[comptime] plan: Capture) -> i32 {
    plan.value()
}

#[test]
fn an_owned_rust_capture_lives_until_the_last_code_or_ir_owner_is_dropped() {
    macro_rules! check {
        ($backend:ty) => {{
            let drops = Arc::new(AtomicUsize::new(0));
            let capture = |value| Capture {
                value,
                drops: Arc::clone(&drops),
            };
            let direct =
                captured_value::build::<$backend>(captured_value::BuildArgs { plan: capture(11) })
                    .unwrap();
            let ir = captured_value::ir(capture(43)).unwrap();
            let portable = <$backend>::compile(&ir).unwrap();
            let other = <$backend>::compile(&ir).unwrap();
            assert_eq!(drops.load(Ordering::Relaxed), 0);
            drop(ir);

            assert_eq!(direct.call(captured_value::CallArgs {}).unwrap(), 11);
            drop(direct);
            assert_eq!(drops.load(Ordering::Relaxed), 1);
            assert_eq!(portable.call(captured_value::CallArgs {}).unwrap(), 43);
            drop(portable);
            assert_eq!(drops.load(Ordering::Relaxed), 1);
            assert_eq!(other.call(captured_value::CallArgs {}).unwrap(), 43);
            drop(other);
            assert_eq!(drops.load(Ordering::Relaxed), 2);
        }};
    }
    splice_testkit::each_backend!(B => check!(B));
}

#[derive(splice::KernelType)]
struct Threshold {
    min: i32,
}

#[kernel]
impl Threshold {
    #[intrinsic("splice.tests.unknown")]
    fn max(a: i32, b: i32) -> i32 {
        if a > b { a } else { b }
    }
}

#[kernel]
fn apply_threshold(#[comptime] threshold: &Threshold, value: i32) -> i32 {
    let min = comptime!(Threshold::max(threshold.min, 0));
    Threshold::max(min, value)
}

#[test]
fn a_published_method_uses_its_body_when_the_intrinsic_is_unavailable() {
    fn check<B: Codegen<Output: Executable>>() {
        for min in [-3, 7] {
            let threshold = Threshold { min };
            let direct = apply_threshold::build::<B>(apply_threshold::BuildArgs {
                threshold: &threshold,
            })
            .unwrap();
            let ir = apply_threshold::ir(&threshold).unwrap();
            for code in [direct, B::compile(&ir).unwrap()] {
                for value in [-5, 6, 10] {
                    assert_eq!(
                        code.call(apply_threshold::CallArgs { value }).unwrap(),
                        apply_threshold(&threshold, value)
                    );
                }
            }
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel(rust)]
fn total(n: usize, value: i32) -> i32 {
    use std::vec::Vec as Values;

    let mut values: Values<i32> = Values::new();
    values.resize(n, value);
    values.iter().copied().fold(0, i32::wrapping_add)
}

#[test]
fn a_runtime_collection_uses_rust_allocation_and_arithmetic() {
    macro_rules! check {
        ($backend:ty) => {{
            let ir = total::ir().unwrap();
            for code in [
                total::build::<$backend>(total::BuildArgs {}).unwrap(),
                <$backend>::compile(&ir).unwrap(),
            ] {
                for (n, value) in [(0, -3), (4, -7), (3, i32::MAX)] {
                    assert_eq!(
                        code.call(total::CallArgs { n, value }).unwrap(),
                        (n as i32).wrapping_mul(value)
                    );
                }
            }
        }};
    }
    splice_testkit::each_backend!(B => check!(B));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Pair(u64, bool);

#[kernel(rust)]
fn constant<T: Copy + 'static>(#[comptime] value: T) -> T {
    value
}

#[test]
fn an_owned_copy_capture_can_be_returned_on_every_call() {
    macro_rules! check {
        ($backend:ty) => {{
            let value = Pair(1 << 45, true);
            let direct = constant::build::<$backend, Pair>(constant::BuildArgs { value }).unwrap();
            let portable = {
                let ir = constant::ir(value).unwrap();
                <$backend>::compile(&ir).unwrap()
            };
            for code in [direct, portable] {
                for _ in 0..2 {
                    assert_eq!(code.call(constant::CallArgs {}).unwrap(), value);
                }
            }
        }};
    }
    splice_testkit::each_backend!(B => check!(B));
}

#[kernel(rust)]
fn borrowed_iter(xs: &[u8]) -> impl Iterator<Item = &u8> {
    xs.iter()
}

#[test]
fn a_borrowing_opaque_result_outlives_the_kernel() {
    macro_rules! check {
        ($backend:ty) => {{
            let bytes = [3, 4, 9];
            let direct = {
                let code = borrowed_iter::build::<$backend>(borrowed_iter::BuildArgs {}).unwrap();
                code.call(borrowed_iter::CallArgs { xs: &bytes }).unwrap()
            };
            assert!(direct.eq(&bytes));

            let portable = {
                let ir = borrowed_iter::ir().unwrap();
                let code = <$backend>::compile(&ir).unwrap();
                code.call(borrowed_iter::CallArgs { xs: &bytes }).unwrap()
            };
            assert!(portable.eq(&bytes));
        }};
    }
    splice_testkit::each_backend!(B => check!(B));
}

#[kernel]
fn sum_items(xs: &[i32]) -> i32 {
    let mut sum = 0_i32;
    for &x in xs {
        sum = sum.wrapping_add(x);
    }
    sum
}

#[test]
fn iterating_a_runtime_slice_lowers_to_a_loop_and_handles_an_empty_input() {
    use splice::hir::Instr;

    let ir = sum_items::ir().unwrap();
    let mut labels = std::collections::HashSet::new();
    let backedge = ir.instructions().iter().any(|op| match op {
        Instr::Label(label) => {
            labels.insert(*label);
            false
        }
        Instr::Jump(label) => labels.contains(label),
        _ => false,
    });
    assert!(backedge);
    assert!(
        !ir.instructions()
            .iter()
            .any(|op| matches!(op, Instr::RustContextCall { .. }))
    );
    assert!(
        ir.instructions()
            .iter()
            .any(|op| matches!(op, Instr::Load { .. }))
    );
    let printed = sum_items::print().unwrap();
    assert!(
        printed.contains(" in xs") && printed.contains("wrapping_add"),
        "{printed}"
    );

    fn check<B: Codegen<Output: Executable>>(ir: &splice::TypedIr<sum_items::Signature>) {
        for code in [
            sum_items::build::<B>(sum_items::BuildArgs {}).unwrap(),
            B::compile(ir).unwrap(),
        ] {
            for xs in [&[][..], &[5], &[i32::MAX, 1, -3, 4]] {
                assert_eq!(
                    code.call(sum_items::CallArgs { xs }).unwrap(),
                    sum_items(xs)
                );
            }
        }
    }

    splice_testkit::each_backend!(B => check::<B>(&ir));
}

static ADD_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
struct AddOperand(i32);

impl std::ops::Add for AddOperand {
    type Output = i32;

    fn add(self, other: Self) -> i32 {
        ADD_CALLS.fetch_add(1, Ordering::Relaxed);
        self.0 + other.0
    }
}

#[kernel]
fn require_overloaded_add(#[comptime] arg: AddOperand) -> i32 {
    comptime!(arg + arg)
}

#[kernel(rust)]
fn overloaded_add(#[comptime] arg: AddOperand) -> i32 {
    arg + arg
}

#[test]
fn overload_resolution_does_not_make_custom_add_pure() {
    macro_rules! check {
        ($backend:ty) => {{
            let before = ADD_CALLS.load(Ordering::Relaxed);
            let arg = AddOperand(9);
            assert_build_error(require_overloaded_add::ir(arg));
            assert_build_error(require_overloaded_add::build::<$backend>(
                require_overloaded_add::BuildArgs { arg },
            ));
            let direct =
                overloaded_add::build::<$backend>(overloaded_add::BuildArgs { arg }).unwrap();
            let portable = <$backend>::compile(&overloaded_add::ir(arg).unwrap()).unwrap();
            assert_eq!(ADD_CALLS.load(Ordering::Relaxed), before);

            for code in [direct, portable] {
                let args = overloaded_add::CallArgs {};
                assert_eq!(code.call(args).unwrap(), 18);
            }
            assert_eq!(ADD_CALLS.load(Ordering::Relaxed), before + 2);
        }};
    }

    splice_testkit::each_backend!(B => check!(B));
}

struct Counter(Cell<i32>);

impl Counter {
    fn tick(&self) -> i32 {
        let next = self.0.get() + 1;
        self.0.set(next);
        next
    }
}

#[kernel]
fn short_borrow_for_effect(#[comptime] counter: &Counter, x: i32) -> i32 {
    x.wrapping_add(counter.tick())
}

#[kernel(rust)]
fn runtime_counter(counter: &Counter) -> i32 {
    counter.tick()
}

#[test]
fn a_method_with_an_interior_write_cannot_use_a_short_build_borrow() {
    macro_rules! check {
        ($backend:ty) => {{
            let counter = Counter(Cell::new(0));
            assert_build_error(short_borrow_for_effect::ir(&counter));
            assert_build_error(short_borrow_for_effect::build::<$backend>(
                short_borrow_for_effect::BuildArgs { counter: &counter },
            ));
            let direct = runtime_counter::build::<$backend>(runtime_counter::BuildArgs {}).unwrap();
            let portable = <$backend>::compile(&runtime_counter::ir().unwrap()).unwrap();
            assert_eq!(counter.0.get(), 0);
            for (index, code) in [direct, portable].into_iter().enumerate() {
                assert_eq!(
                    code.call(runtime_counter::CallArgs { counter: &counter })
                        .unwrap(),
                    1 + index as i32
                );
            }
            assert_eq!(counter.0.get(), 2);
        }};
    }

    splice_testkit::each_backend!(B => check!(B));
}

#[kernel]
trait Map {
    fn apply(value: i32, #[comptime] offset: i32) -> i32;
}

struct PublishedMap;

#[kernel]
impl Map for PublishedMap {
    fn apply(value: i32, #[comptime] offset: i32) -> i32 {
        value.wrapping_add(offset)
    }
}

impl PublishedMap {
    fn apply(value: i32, offset: i32) -> i32 {
        value.wrapping_sub(offset)
    }
}

#[kernel]
fn call_shadowing_method(value: i32) -> i32 {
    PublishedMap::apply(value, 3)
}

#[kernel]
fn require_shadowing_method() -> i32 {
    comptime!(PublishedMap::apply(1, 3))
}

static ORDINARY_MAP_CALLS: AtomicUsize = AtomicUsize::new(0);

struct OrdinaryMap;

impl Map for OrdinaryMap {
    fn apply(value: i32, offset: i32) -> i32 {
        ORDINARY_MAP_CALLS.fetch_add(1, Ordering::Relaxed);
        value.wrapping_mul(offset)
    }
}

#[kernel]
fn apply_maps(#[comptime] offset: i32, value: i32) -> i32 {
    let delta = comptime!(<PublishedMap as Map>::apply(1, offset));
    <OrdinaryMap as Map>::apply(<PublishedMap as Map>::apply(value, offset), delta)
}

#[test]
fn trait_dispatch_requires_publication_on_the_selected_implementation() {
    fn check<B: Codegen<Output: Executable>>() {
        assert!(require_shadowing_method::ir().is_err());
        let shadow =
            call_shadowing_method::build::<B>(call_shadowing_method::BuildArgs {}).unwrap();
        let from_ir = B::compile(&call_shadowing_method::ir().unwrap()).unwrap();
        for code in [shadow, from_ir] {
            assert_eq!(
                code.call(call_shadowing_method::CallArgs { value: 11 })
                    .unwrap(),
                8
            );
        }
        let before = ORDINARY_MAP_CALLS.load(Ordering::Relaxed);
        let direct = apply_maps::build::<B>(apply_maps::BuildArgs { offset: 3 }).unwrap();
        let portable = B::compile(&apply_maps::ir(3).unwrap()).unwrap();
        assert_eq!(ORDINARY_MAP_CALLS.load(Ordering::Relaxed), before);
        for code in [direct, portable] {
            assert_eq!(code.call(apply_maps::CallArgs { value: 2 }).unwrap(), 20);
        }
        assert_eq!(ORDINARY_MAP_CALLS.load(Ordering::Relaxed), before + 2);
    }

    splice_testkit::each_backend!(B => check::<B>());
}

fn add_three(value: i32) -> i32 {
    value.wrapping_add(3)
}

fn multiply_by_five(value: i32) -> i32 {
    value.wrapping_mul(5)
}

fn fail_callback(_: i32) -> i32 {
    panic!("callback failed")
}

#[kernel(rust)]
fn use_callback(#[comptime] callback: fn(i32) -> i32, x: i32) -> i32 {
    callback(x)
}

#[test]
fn captured_function_pointers_keep_their_identity_and_handle_unwinds() {
    macro_rules! check {
        ($backend:ty) => {{
            for (callback, expected) in [
                (add_three as fn(i32) -> i32, Some(10)),
                (fail_callback, None),
                (multiply_by_five, Some(35)),
            ] {
                let direct =
                    use_callback::build::<$backend>(use_callback::BuildArgs { callback }).unwrap();
                let ir = use_callback::ir(callback).unwrap();
                for code in [direct, <$backend>::compile(&ir).unwrap()] {
                    assert_eq!(code.call(use_callback::CallArgs { x: 7 }).ok(), expected);
                }
            }
        }};
    }

    splice_testkit::each_backend!(B => check!(B));
}

#[derive(splice::KernelType)]
struct PublishedTotals {
    count: i32,
}

#[kernel]
impl PublishedTotals {
    fn max(a: i32, b: i32) -> i32 {
        if a > b { a } else { b }
    }

    fn limit(&self, value: i32) -> i32 {
        if value < self.count {
            self.count
        } else {
            value
        }
    }
}

#[kernel]
fn apply_published_limit(#[comptime] plan: &PublishedTotals, value: i32) -> i32 {
    let minimum = comptime!(PublishedTotals::limit(plan, 3));
    PublishedTotals::limit(plan, value) + minimum + comptime!(PublishedTotals::max(3, 7))
}

#[test]
fn inherent_methods_in_a_marked_impl_can_specialize() {
    fn check<B: Codegen<Output: Executable>>() {
        let (direct, ir) = {
            let plan = PublishedTotals { count: 5 };
            let direct =
                apply_published_limit::build::<B>(apply_published_limit::BuildArgs { plan: &plan })
                    .unwrap();
            (direct, apply_published_limit::ir(&plan).unwrap())
        };
        let portable = B::compile(&ir).unwrap();
        for code in [direct, portable] {
            for (value, expected) in [(2, 17), (7, 19)] {
                assert_eq!(
                    code.call(apply_published_limit::CallArgs { value })
                        .unwrap(),
                    expected,
                );
            }
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

mod trait_stage {
    use super::*;

    mod api {
        use splice::kernel;

        #[kernel]
        pub(super) trait Step {
            fn advance(value: i32, #[comptime] offset: i32) -> i32;
        }

        pub(super) trait Earlier {
            fn previous(value: i32) -> i32;
        }
    }

    mod implementation {
        use super::api::{Earlier, Step as StepAlias};
        use splice::kernel;

        pub(super) struct Open;

        #[kernel]
        impl StepAlias for Open {
            fn advance(value: i32, #[comptime] offset: i32) -> i32 {
                value.wrapping_add(offset)
            }
        }

        impl Earlier for Open {
            fn previous(value: i32) -> i32 {
                super::ORDINARY_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                value.wrapping_sub(1)
            }
        }
    }

    use self::api::{Earlier, Step, Step as StepAlias};
    use self::implementation::Open;

    struct Late;

    static ORDINARY_CALLS: AtomicUsize = AtomicUsize::new(0);

    impl Step for Late {
        fn advance(value: i32, offset: i32) -> i32 {
            ORDINARY_CALLS.fetch_add(1, Ordering::Relaxed);
            value.wrapping_mul(offset)
        }
    }

    #[kernel]
    fn combine<T: StepAlias>(#[comptime] offset: i32, value: i32) -> i32 {
        let initial = comptime!(<Open as StepAlias>::advance(2, offset));
        initial + <T as StepAlias>::advance(value, offset) + <Open as Earlier>::previous(value)
    }

    #[kernel]
    fn takes_runtime_offset(value: i32) -> i32 {
        <Late as Step>::advance(1, value)
    }

    #[test]
    fn a_marked_trait_preserves_stage_rules_for_an_unmarked_impl() {
        fn check<B: Codegen<Output: Executable>>() {
            assert_build_error(takes_runtime_offset::ir());
            let before = ORDINARY_CALLS.load(Ordering::Relaxed);
            let direct = combine::build::<B, Late>(combine::BuildArgs { offset: 3 }).unwrap();
            let portable = B::compile(&combine::ir::<Late>(3).unwrap()).unwrap();
            assert_eq!(ORDINARY_CALLS.load(Ordering::Relaxed), before);
            for code in [direct, portable] {
                assert_eq!(code.call(combine::CallArgs { value: 4 }).unwrap(), 20);
            }
            assert_eq!(ORDINARY_CALLS.load(Ordering::Relaxed), before + 4);
        }

        splice_testkit::each_backend!(B => check::<B>());
    }
}

mod staged_conditions {
    use splice::kernel;

    #[kernel]
    pub(super) fn identity(a: i32) -> i32 {
        a
    }

    // CORE-04: whether `identity(k) > 0` is known is the frontend's Word,
    // not a peephole over the backend's last emitted instruction.
    #[kernel]
    pub(super) fn known(#[comptime] k: i32, x: i32) -> i32 {
        if identity(k) > 0 { x + 1 } else { x - 1 }
    }

    #[kernel]
    pub(super) fn unknown(x: i32) -> i32 {
        if identity(x) > 0 { x + 1 } else { x - 1 }
    }
}

#[test]
fn mixed_stage_conditions_fold_in_the_frontend_on_every_path() {
    use splice::hir::Instr;
    use staged_conditions::{known, unknown};

    fn check<B: Codegen<Output: Executable>>() {
        for k in [-3, 0, 4] {
            let direct = known::build::<B>(known::BuildArgs { k }).unwrap();
            let portable = B::compile(&known::ir(k).unwrap()).unwrap();
            for code in [&direct, &portable] {
                for x in [-5, 0, 9] {
                    assert_eq!(code.call(known::CallArgs { x }), Ok(known(k, x)));
                }
            }
        }
        let direct = unknown::build::<B>(unknown::BuildArgs {}).unwrap();
        let portable = B::compile(&unknown::ir().unwrap()).unwrap();
        for code in [&direct, &portable] {
            for x in [-5, 0, 9] {
                assert_eq!(code.call(unknown::CallArgs { x }), Ok(unknown(x)));
            }
        }
    }
    splice_testkit::each_backend!(B => check::<B>());

    // The portable IR shows the structure that the backends compile: a known
    // condition keeps one arm and no branch, an unknown one branches.
    let branches = |ir: &splice::Function| {
        ir.instructions()
            .iter()
            .any(|op| matches!(op, Instr::Branch { .. }))
    };
    for k in [-3, 4] {
        let ir = known::ir(k).unwrap();
        assert!(!branches(&ir));
    }
    let ir = unknown::ir().unwrap();
    assert!(branches(&ir));
}

mod index_arithmetic {
    use splice::kernel;

    // MAC-02: `usize` locals follow Rust `usize` arithmetic in 64 bits.
    #[kernel]
    pub(super) fn decrement_len(xs: &[i32]) -> i32 {
        let mut i = xs.len();
        i -= 1;
        if i > 5 { 1 } else { 0 }
    }

    #[kernel]
    pub(super) fn double_len(xs: &[i32], extra: i32) -> i32 {
        let mut i = xs.len();
        i += extra as usize;
        i *= 2;
        if i > 5 { 1 } else { 0 }
    }

    // A `usize` parameter and result cross the entry as 64-bit words.
    #[kernel]
    pub(super) fn shift_index(base: usize, xs: &[i32], past: bool) -> usize {
        if past { base + xs.len() } else { base }
    }
}

#[test]
fn compound_index_assignment_matches_rust_usize() {
    use index_arithmetic::{decrement_len, double_len};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    fn expect<B: Codegen<Output: Executable>>(
        native: impl FnOnce() -> i32,
        jit: Result<i32, String>,
        case: &str,
    ) {
        let backend = std::any::type_name::<B>();
        match catch_unwind(AssertUnwindSafe(native)) {
            Ok(value) => assert_eq!(jit, Ok(value), "{backend} {case}"),
            Err(_) => assert!(
                jit.is_err(),
                "{backend} {case}: Rust panicked, JIT gave {jit:?}"
            ),
        }
    }

    // Both kernels stage natively, not as a call into their Rust function.
    for ir in [
        decrement_len::ir().unwrap().into_inner(),
        double_len::ir().unwrap().into_inner(),
    ] {
        assert!(!ir.instructions().iter().any(|i| matches!(
            i,
            splice::hir::Instr::RustCall { .. } | splice::hir::Instr::RustPacketCall { .. }
        )));
    }

    fn check<B: Codegen<Output: Executable>>() {
        let empty: [i32; 0] = [];
        let decrement = decrement_len::build::<B>(decrement_len::BuildArgs {}).unwrap();
        let jit = decrement
            .call(decrement_len::CallArgs { xs: &empty })
            .map_err(|e| e.to_string());
        expect::<B>(|| decrement_len(&empty), jit, "0 - 1");

        // 2 * (1 + i32::MAX) is 2^32: exact in usize, but not in 32 bits.
        let one = [0];
        let double = double_len::build::<B>(double_len::BuildArgs {}).unwrap();
        let jit = double
            .call(double_len::CallArgs {
                xs: &one,
                extra: i32::MAX,
            })
            .map_err(|e| e.to_string());
        expect::<B>(|| double_len(&one, i32::MAX), jit, "2^31 * 2");
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[test]
fn usize_parameters_and_results_keep_all_64_bits() {
    use index_arithmetic::shift_index;
    fn check<B: Codegen<Output: Executable>>() {
        let kernel = shift_index::build::<B>(shift_index::BuildArgs {}).unwrap();
        let xs = [1, 2, 3];
        for (base, past) in [
            (0, true),
            (1 << 40, true),
            (usize::MAX, false),
            (1 << 63, true),
        ] {
            let jit = kernel.call(shift_index::CallArgs {
                base,
                xs: &xs,
                past,
            });
            let backend = std::any::type_name::<B>();
            assert_eq!(
                jit,
                Ok(shift_index(base, &xs, past)),
                "{backend} {base} {past}"
            );
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
}
