use splice::simd::Lane;
use splice::{BuildError, Codegen, Executable, kernel};
use splice_lancy::Lancy;
use std::cell::Cell;
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

#[derive(splice::KernelType)]
struct Wrapped(i32, i32);

/// A tuple struct's constructor builds an aggregate like a struct literal.
#[kernel]
fn wrapped(#[comptime] known: i32, value: i32) -> i32 {
    let pair = Wrapped(known, value);
    pair.0 * 10 + pair.1
}

#[test]
fn tuple_struct_constructors_take_runtime_fields() {
    fn check<B: Codegen<Output: Executable>>() {
        let portable = B::compile(&wrapped::ir(3).unwrap()).unwrap();
        let direct = wrapped::build::<B>(wrapped::BuildArgs { known: 3 }).unwrap();
        for code in [direct, portable] {
            assert_eq!(
                code.call(wrapped::CallArgs { value: 4 }).unwrap(),
                wrapped(3, 4)
            );
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
}

/// `comptime!` in the elements of a host `vec!` is evaluated by the
/// generator, inside a loop or not.
#[kernel]
fn host_vectors(#[comptime] n: i32, value: i32) -> i32 {
    let first = vec![comptime!(n)];
    let second = vec![comptime!(n + 1), 2];
    let third = vec![comptime!(n * 2); 2];
    value + first[0] + second[0] + third[1]
}

#[test]
fn comptime_in_host_vec_elements_is_evaluated() {
    fn check<B: Codegen<Output: Executable>>() {
        let portable = B::compile(&host_vectors::ir(3).unwrap()).unwrap();
        let direct = host_vectors::build::<B>(host_vectors::BuildArgs { n: 3 }).unwrap();
        for code in [direct, portable] {
            assert_eq!(
                code.call(host_vectors::CallArgs { value: 4 }).unwrap(),
                host_vectors(3, 4)
            );
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
}

/// MAC-23: `#[cfg]` and `#[cfg_attr]` on parameters apply to every
/// generated parameter list, as rustc applies them to the function.
#[kernel]
fn configured(
    #[cfg(not(test))] absent: i32,
    #[cfg(test)] value: i32,
    #[cfg_attr(test, comptime)] scale: i32,
    #[cfg_attr(not(test), comptime)] offset: i32,
) -> i32 {
    value * scale + offset
}

struct Configured;

#[kernel]
impl Configured {
    fn scaled(
        #[cfg(not(test))] absent: bool,
        #[cfg_attr(test, comptime)] scale: i32,
        value: i32,
    ) -> i32 {
        value * scale
    }
}

#[kernel]
fn configured_method(value: i32) -> i32 {
    Configured::scaled(3, value)
}

#[test]
fn configured_parameters_follow_their_conditions() {
    fn check<B: Codegen<Output: Executable>>() {
        let portable = B::compile(&configured::ir(3).unwrap()).unwrap();
        let direct = configured::build::<B>(configured::BuildArgs { scale: 3 }).unwrap();
        for code in [direct, portable] {
            let args = configured::CallArgs {
                value: 4,
                offset: 5,
            };
            assert_eq!(code.call(args).unwrap(), configured(4, 3, 5));
        }
        let method = B::compile(&configured_method::ir().unwrap()).unwrap();
        assert_eq!(
            method
                .call(configured_method::CallArgs { value: 4 })
                .unwrap(),
            configured_method(4)
        );
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

#[kernel(print)]
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
            .any(|op| matches!(op, Instr::RustCall { .. }))
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

#[test]
fn a_method_with_an_interior_write_cannot_use_a_short_build_borrow() {
    macro_rules! check {
        ($backend:ty) => {{
            let counter = Counter(Cell::new(0));
            assert_build_error(short_borrow_for_effect::ir(&counter));
            assert_build_error(short_borrow_for_effect::build::<$backend>(
                short_borrow_for_effect::BuildArgs { counter: &counter },
            ));
            assert_eq!(counter.0.get(), 0);
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

    // MAC-07: an unannotated literal local takes the `usize` type Rust
    // infers from its uses; an annotated mutable one keeps its annotation.
    #[kernel]
    pub(super) fn sum_while(xs: &[i32]) -> i32 {
        let mut i = 0;
        let mut sum: i32 = 0;
        while i < xs.len() {
            sum = sum.wrapping_add(xs[i]);
            i += 1;
        }
        sum
    }

    // A counter compared with a `usize` local, and a name the body binds
    // twice: each declaration takes its own type.
    #[kernel]
    pub(super) fn sum_shadowed(xs: &[i32]) -> i32 {
        let n = xs.len();
        let mut i = 0;
        let mut sum: i32 = 0;
        while i < n {
            sum = sum.wrapping_add(xs[i]);
            i += 1;
        }
        let i = 2;
        sum.wrapping_mul(i)
    }

    #[kernel]
    pub(super) fn sum_annotated(xs: &[i32]) -> i32 {
        let mut i: usize = 0;
        let mut sum: i32 = 0;
        while i < xs.len() {
            sum = sum.wrapping_add(xs[i]);
            i += 1;
        }
        sum
    }

    // `back` is a `usize` only through `back - 1 < xs.len()`; below zero it
    // wraps in release and panics in debug, as in Rust.
    #[kernel]
    pub(super) fn last_before(xs: &[i32], skip: i32) -> i32 {
        let mut back = 0;
        back += skip as usize;
        if back - 1 < xs.len() {
            xs[back - 1]
        } else {
            -1
        }
    }

    // A literal arm joins a `usize` arm.
    #[kernel]
    pub(super) fn length_or_zero(xs: &[i32], empty: bool) -> usize {
        if empty { 0 } else { xs.len() }
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
        assert!(
            !ir.instructions()
                .iter()
                .any(|i| matches!(i, splice::hir::Instr::RustCall { .. }))
        );
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
fn literal_locals_and_arms_take_the_usize_type_rust_infers() {
    use index_arithmetic::{last_before, length_or_zero, sum_annotated, sum_shadowed, sum_while};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    fn check<B: Codegen<Output: Executable>>() {
        let backend = std::any::type_name::<B>();
        let while_kernel = sum_while::build::<B>(sum_while::BuildArgs {}).unwrap();
        let annotated = sum_annotated::build::<B>(sum_annotated::BuildArgs {}).unwrap();
        let shadowed = sum_shadowed::build::<B>(sum_shadowed::BuildArgs {}).unwrap();
        let last = last_before::build::<B>(last_before::BuildArgs {}).unwrap();
        let length = length_or_zero::build::<B>(length_or_zero::BuildArgs {}).unwrap();
        for xs in [&[][..], &[4], &[1, 2, i32::MAX]] {
            assert_eq!(
                while_kernel.call(sum_while::CallArgs { xs }),
                Ok(sum_while(xs))
            );
            assert_eq!(
                annotated.call(sum_annotated::CallArgs { xs }),
                Ok(sum_annotated(xs))
            );
            assert_eq!(
                shadowed.call(sum_shadowed::CallArgs { xs }),
                Ok(sum_shadowed(xs))
            );
            for empty in [false, true] {
                assert_eq!(
                    length.call(length_or_zero::CallArgs { xs, empty }),
                    Ok(length_or_zero(xs, empty))
                );
            }
            for skip in [0, 1, 3] {
                let jit = last.call(last_before::CallArgs { xs, skip });
                match catch_unwind(AssertUnwindSafe(|| last_before(xs, skip))) {
                    Ok(value) => assert_eq!(jit, Ok(value), "{backend} {xs:?} {skip}"),
                    Err(_) => assert!(jit.is_err(), "{backend} {xs:?} {skip}: {jit:?}"),
                }
            }
        }
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

// `min`/`max` are tested here as registered intrinsics; `clamp` is not one.
#[allow(clippy::manual_clamp)]
#[kernel]
fn integer_methods(xs: &[i32], a: i32, b: i32, n: usize) -> i32 {
    let len = xs.len();
    let index = len.saturating_sub(n).min(len.wrapping_sub(1));
    let clamp = a.max(-100).min(100);
    if index < len {
        xs[index]
            .saturating_add(a.saturating_mul(b))
            .wrapping_sub(a.wrapping_neg())
            .wrapping_add(clamp)
    } else {
        a.saturating_sub(b).wrapping_add(n.max(3) as i32)
    }
}

/// MAC-13, MAC-14: integer methods of runtime `i32` and `usize` values are
/// registered intrinsics: each stages one portable operation, without a
/// Rust call, and matches Rust on both build paths.
#[test]
fn integer_methods_are_intrinsics_on_every_integer_type() {
    use splice::hir::{Binary, Instr};
    let ir = integer_methods::ir().unwrap();
    let ops: Vec<_> = splice_testkit::instructions(&ir)
        .filter_map(|op| match op {
            Instr::Bin(_, op, _, _) => Some(*op),
            _ => None,
        })
        .collect();
    for op in [
        Binary::SaturatingAdd,
        Binary::SaturatingSub,
        Binary::SaturatingMul,
        Binary::Min,
        Binary::Max,
    ] {
        assert!(ops.contains(&op), "{op:?} in {ops:?}");
    }
    assert!(!splice_testkit::instructions(&ir).any(|op| matches!(op, Instr::RustCall { .. })));
    fn check<B: Codegen<Output: Executable>>() {
        let direct = integer_methods::build::<B>(integer_methods::BuildArgs {}).unwrap();
        let portable = B::compile(&integer_methods::ir().unwrap()).unwrap();
        let data = [7, i32::MAX, -3, i32::MIN];
        for xs in [&data[..], &[]] {
            for (a, b) in [
                (1, 2),
                (i32::MAX, 2),
                (i32::MIN, 3),
                (-5, i32::MIN),
                (250, -7),
            ] {
                for n in [0, 1, 3, usize::MAX] {
                    let expected = integer_methods(xs, a, b, n);
                    for code in [&direct, &portable] {
                        let actual = code.call(integer_methods::CallArgs { xs, a, b, n });
                        assert_eq!(actual, Ok(expected), "{xs:?} {a} {b} {n}");
                    }
                }
            }
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
}

mod intrinsic_methods {
    use splice::kernel;

    pub struct Math;

    #[kernel]
    impl Math {
        #[intrinsic("splice.i32.max")]
        pub fn max(a: i32, b: i32) -> i32 {
            if a > b { a } else { b }
        }

        #[intrinsic("acme.i32.max")]
        pub fn unregistered(a: i32, b: i32) -> i32 {
            if a > b { a } else { b }
        }
    }

    #[kernel]
    pub fn registered(x: i32, y: i32) -> i32 {
        Math::max(x, y)
    }

    #[kernel]
    pub fn fallback(x: i32, y: i32) -> i32 {
        Math::unregistered(x, y)
    }
}

/// MAC-13: a `#[intrinsic("name")]` method stages the registered operation
/// in place of its body; an unregistered name stages the body.
#[test]
fn intrinsic_methods_stage_the_registered_operation_or_their_body() {
    use intrinsic_methods::{fallback, registered};
    use splice::hir::{Binary, Instr};
    let branches = |ir: &splice::Function| {
        splice_testkit::instructions(ir).any(|op| matches!(op, Instr::Branch { .. }))
    };
    let maxima = |ir: &splice::Function| {
        splice_testkit::instructions(ir).any(|op| matches!(op, Instr::Bin(_, Binary::Max, ..)))
    };
    let ir = registered::ir().unwrap();
    assert!(maxima(&ir) && !branches(&ir));
    let ir = fallback::ir().unwrap();
    assert!(!maxima(&ir) && branches(&ir));
    for (x, y) in [(-5, 2), (7, 3), (i32::MIN, i32::MAX)] {
        let code = registered::build::<Lancy>(registered::BuildArgs {}).unwrap();
        assert_eq!(code.call(registered::CallArgs { x, y }), Ok(x.max(y)));
        let code = fallback::build::<Lancy>(fallback::BuildArgs {}).unwrap();
        assert_eq!(code.call(fallback::CallArgs { x, y }), Ok(x.max(y)));
    }
}

/// MAC-09: a primitive named by its full path is the same type.
#[kernel]
fn primitive_paths(
    xs: &[::core::primitive::i32],
    scale: core::primitive::i32,
    flag: std::primitive::bool,
) -> ::core::primitive::i32 {
    let index: ::core::primitive::usize = 0;
    let first = xs[index];
    if flag { first * scale } else { first }
}

#[test]
fn primitive_types_may_be_spelled_by_their_core_path() {
    fn check<B: Codegen<Output: Executable>>() {
        let code = primitive_paths::build::<B>(primitive_paths::BuildArgs {}).unwrap();
        for flag in [false, true] {
            let xs = [7, 1];
            assert_eq!(
                code.call(primitive_paths::CallArgs {
                    xs: &xs,
                    scale: 3,
                    flag
                }),
                Ok(primitive_paths(&xs, 3, flag))
            );
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
}

/// MAC-12: labeled loops, `break 'label` and `continue 'label` lower to IR
/// across runtime loops, unrolled loops and loops the jump stages.
mod labels {
    use splice::kernel;

    #[kernel(print)]
    pub(super) fn first_pair(xs: &[i32], target: i32) -> i32 {
        let mut found = -1;
        let mut i = 0;
        'outer: while i < xs.len() {
            let mut j = 0;
            while j < xs.len() {
                if j > i && xs[i] + xs[j] == target {
                    found = (i * 10 + j) as i32;
                    break 'outer;
                }
                j += 1;
            }
            i += 1;
        }
        found
    }

    #[kernel]
    pub(super) fn rows_without_negatives(xs: &[i32], width: i32) -> i32 {
        let mut rows = 0;
        let mut row = 0;
        'rows: while row * width < xs.len() as i32 {
            row += 1;
            for column in 0..width {
                if xs[((row - 1) * width + column) as usize] < 0 {
                    continue 'rows;
                }
            }
            rows += 1;
        }
        rows
    }

    #[kernel]
    pub(super) fn steps_found(#[comptime] plan: &[i32], xs: &[i32]) -> i32 {
        let mut hits = 0;
        'steps: for &step in plan {
            for &x in xs {
                if x == step {
                    hits += 1;
                    continue 'steps;
                }
            }
            hits += 100;
        }
        hits
    }

    #[kernel]
    pub(super) fn stop_early(xs: &[i32], #[comptime] tries: i32) -> i32 {
        let mut i = 0;
        let mut sum = 0;
        'scan: while i < xs.len() {
            for _try in 0..tries {
                if xs[i] == 0 {
                    break 'scan;
                }
                sum += xs[i];
            }
            i += 1;
        }
        sum
    }
}

#[test]
fn labeled_loops_lower_to_ir() {
    use labels::{first_pair, rows_without_negatives, steps_found, stop_early};
    fn check<B: Codegen<Output: Executable>>() {
        let pair = first_pair::build::<B>(first_pair::BuildArgs {}).unwrap();
        let rows =
            rows_without_negatives::build::<B>(rows_without_negatives::BuildArgs {}).unwrap();
        let plan = [3, 9, 4];
        let steps = steps_found::build::<B>(steps_found::BuildArgs { plan: &plan }).unwrap();
        let stop = stop_early::build::<B>(stop_early::BuildArgs { tries: 2 }).unwrap();
        for xs in [&[][..], &[1, 2, 3, 4], &[4, -1, 3, 0, 2, 2], &[5, 3, 0, 9]] {
            for target in [5, 7, 100] {
                assert_eq!(
                    pair.call(first_pair::CallArgs { xs, target }),
                    Ok(first_pair(xs, target))
                );
            }
            for width in [1, 2] {
                assert_eq!(
                    rows.call(rows_without_negatives::CallArgs { xs, width }),
                    Ok(rows_without_negatives(xs, width))
                );
            }
            assert_eq!(
                steps.call(steps_found::CallArgs { xs }),
                Ok(steps_found(&plan, xs))
            );
            assert_eq!(
                stop.call(stop_early::CallArgs { xs }),
                Ok(stop_early(xs, 2))
            );
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
    let printed = labels::first_pair::print().unwrap();
    assert!(printed.contains("'outer: while"), "{printed}");
    assert!(printed.contains("break 'outer"), "{printed}");
}

/// MAC-12: a runtime `match` that rustc accepts as exhaustive needs no
/// final `_` arm.
#[kernel]
fn exhaustive_matches(flag: bool, x: i32) -> i32 {
    let base = match flag {
        true => 10,
        false => 20,
    };
    let sign = match x {
        i32::MIN..=-1 => -1,
        0 => 0,
        1..=i32::MAX => 1,
    };
    base + sign
}

#[test]
fn exhaustive_runtime_matches_need_no_wildcard() {
    fn check<B: Codegen<Output: Executable>>() {
        let built = exhaustive_matches::build::<B>(exhaustive_matches::BuildArgs {}).unwrap();
        let compiled = B::compile(&exhaustive_matches::ir().unwrap()).unwrap();
        for code in [built, compiled] {
            for flag in [false, true] {
                for x in [i32::MIN, -3, 0, 5, i32::MAX] {
                    assert_eq!(
                        code.call(exhaustive_matches::CallArgs { flag, x }),
                        Ok(exhaustive_matches(flag, x))
                    );
                }
            }
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
}

/// MAC-12: `break value` leaves a runtime `loop` with its value, also from
/// a nested loop by label.
#[kernel(print)]
fn loop_values(xs: &[i32], target: i32) -> i32 {
    let mut i = 0;
    let found = loop {
        if i == xs.len() {
            break -1;
        }
        if xs[i] == target {
            break i as i32;
        }
        i += 1;
    };
    let mut j = 0;
    let first_negative = 'search: loop {
        while j < xs.len() {
            if xs[j] < 0 {
                break 'search xs[j];
            }
            j += 1;
        }
        if j == xs.len() {
            break 0;
        }
    };
    let mut n = 0;
    // A loop of the generator: its value is known.
    let steps = loop {
        n += 1;
        if n > 3 {
            break n;
        }
    };
    found * 100 + first_negative + comptime!(steps) * 1000
}

#[test]
fn break_values_leave_runtime_loops() {
    fn check<B: Codegen<Output: Executable>>() {
        let built = loop_values::build::<B>(loop_values::BuildArgs {}).unwrap();
        let compiled = B::compile(&loop_values::ir().unwrap()).unwrap();
        for code in [built, compiled] {
            for xs in [&[][..], &[3, 5, 7], &[4, -2, 5, -9]] {
                for target in [5, 6] {
                    assert_eq!(
                        code.call(loop_values::CallArgs { xs, target }),
                        Ok(loop_values(xs, target))
                    );
                }
            }
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
    let printed = loop_values::print().unwrap();
    assert!(printed.contains("break 'search"), "{printed}");
}

/// MAC-12: a guard with a runtime condition in a `match` on a known value
/// tests the guard at run time and falls through to the next arm whose
/// pattern matches.
#[kernel(print)]
fn known_match_guards(#[comptime] mode: i32, x: i32) -> i32 {
    match mode {
        0 if x > 10 => x - 10,
        0 | 1 if x < 0 => -x,
        2 => 7,
        n if x == n => 1000,
        n => n + x,
    }
}

#[test]
fn known_match_runtime_guards_fall_through() {
    fn check<B: Codegen<Output: Executable>>() {
        for mode in [0, 1, 2, 3] {
            let built =
                known_match_guards::build::<B>(known_match_guards::BuildArgs { mode }).unwrap();
            let compiled = B::compile(&known_match_guards::ir(mode).unwrap()).unwrap();
            for code in [built, compiled] {
                for x in [-4, 0, 3, 12] {
                    assert_eq!(
                        code.call(known_match_guards::CallArgs { x }),
                        Ok(known_match_guards(mode, x)),
                        "mode {mode}, x {x}"
                    );
                }
            }
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
    let printed = known_match_guards::print(0).unwrap();
    assert!(printed.contains("if x > 10"), "{printed}");
    let printed = known_match_guards::print(3).unwrap();
    assert!(printed.contains("if x == 3"), "{printed}");
    assert!(!printed.contains("x > 10"), "{printed}");
}

/// MAC-18: names resolve to their declarations. A local of a `comptime!`
/// block shadows the runtime parameter of the same name.
#[kernel]
fn shadowed_by_declarations(#[comptime] steps: &[i32], x: i32) -> i32 {
    let total = comptime!({
        let x = steps.len() as i32 * 2;
        x + 1
    });
    x + total
}

/// MAC-18: a closure parameter shadows the runtime parameter `x`: the
/// closure is an opaque operation, not a read of `x`.
#[kernel]
fn closure_shadow(#[comptime] steps: &[i32], x: i32) -> i32 {
    comptime!(steps.iter().map(|x| x + 1).sum::<i32>()) + x
}

#[test]
fn closure_and_block_locals_shadow_runtime_parameters() {
    fn check<B: Codegen<Output: Executable>>() {
        let steps = [1, 2, 3];
        let built = shadowed_by_declarations::build::<B>(shadowed_by_declarations::BuildArgs {
            steps: &steps,
        })
        .unwrap();
        let compiled = B::compile(&shadowed_by_declarations::ir(&steps).unwrap()).unwrap();
        for code in [built, compiled] {
            for x in [-4, 0, 7] {
                assert_eq!(
                    code.call(shadowed_by_declarations::CallArgs { x }),
                    Ok(shadowed_by_declarations(&steps, x))
                );
            }
        }
        let error = closure_shadow::build::<B>(closure_shadow::BuildArgs { steps: &steps });
        assert!(
            matches!(&error, Err(BuildError::Staging(message)) if message.contains("opaque operation")),
            "{:?}",
            error.err()
        );
    }
    splice_testkit::each_backend!(B => check::<B>());
}

/// MAC-06: one evaluator decides what a body evaluates: an index into a
/// known slice evaluates at specialization like the other nodes.
#[kernel]
fn element(#[comptime] table: &[i32], index: usize) -> i32 {
    table[index]
}

const TABLE: &[i32] = &[4, 5, 6];

#[kernel]
fn known_element(x: i32) -> i32 {
    comptime!(element(TABLE, 1)) + x
}

#[test]
fn the_evaluator_reads_known_slices() {
    fn check<B: Codegen<Output: Executable>>() {
        let built = known_element::build::<B>(known_element::BuildArgs {}).unwrap();
        let compiled = B::compile(&known_element::ir().unwrap()).unwrap();
        for code in [built, compiled] {
            assert_eq!(
                code.call(known_element::CallArgs { x: 2 }),
                Ok(known_element(2))
            );
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
}

/// MAC-07, CORE-01: runtime parameters, locals, literals, slice elements and
/// results of every kernel scalar type, with the types rustc infers for
/// unsuffixed literals.
#[kernel(print)]
fn typed_scalars(a: i64, b: u32, c: u64, x: f64, xs: &[i64], fs: &[f32], out: &mut [u32]) -> i64 {
    let mut sum = 0;
    let mut i = 0;
    while i < xs.len() {
        sum += xs[i] * 3 - 1;
        i += 1;
    }
    let mut total = 0.0;
    for &f in fs {
        total += f * 2.5;
    }
    let mut written = 0;
    let mut j = 0;
    while j < out.len() {
        out[j] = (b >> 1) + j as u32;
        written += out[j];
        j += 1;
    }
    for k in 0..c % 4 {
        written += k as u32;
    }
    let neg = -x;
    let scaled = if neg < 0.5 { neg * 2.0 } else { 1.5 };
    let wide = c / 3 + !c % 7;
    sum + a + written as i64 + wide as i64 + scaled as i64 + total as i64 + 1i64
}

#[test]
fn kernels_compute_with_every_scalar_type() {
    fn check<B: Codegen<Output: Executable>>() {
        let built = typed_scalars::build::<B>(typed_scalars::BuildArgs {}).unwrap();
        let compiled = B::compile(&typed_scalars::ir().unwrap()).unwrap();
        for code in [built, compiled] {
            for (a, b, c, x) in [
                (5i64, 9u32, 100u64, 0.25f64),
                (-1 << 40, 4_000_000_000, u64::MAX, -3.5),
            ] {
                for (xs, fs) in [
                    (&[][..], &[][..]),
                    (&[1i64 << 35, -7, 4][..], &[0.5f32, -1.25][..]),
                ] {
                    let (mut out, mut expected) = ([0u32; 3], [0u32; 3]);
                    let result = code.call(typed_scalars::CallArgs {
                        a,
                        b,
                        c,
                        x,
                        xs,
                        fs,
                        out: &mut out,
                    });
                    assert_eq!(result, Ok(typed_scalars(a, b, c, x, xs, fs, &mut expected)));
                    assert_eq!(out, expected);
                }
            }
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
    let printed = typed_scalars::print().unwrap();
    assert!(printed.contains("for k in"), "{printed}");
}
