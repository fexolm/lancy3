use splice::{BuildOptions, Codegen, Executable, Mode, kernel};
use splice_lancy::Lancy;
use std::{
    cell::Cell,
    sync::atomic::{AtomicUsize, Ordering},
};

#[kernel]
fn count_by_twos(#[comptime] target: i32, value: i32) -> i32 {
    let mut count = 0_i32;
    while count != target {
        count = count.wrapping_add(2);
    }
    value.wrapping_add(comptime!(count))
}

#[kernel]
fn mandatory_loop(#[comptime] target: i32, value: i32) -> i32 {
    let total = comptime!({
        let mut count = 0_i32;
        while count != target {
            count += 2;
        }
        let mut total = 0;
        for i in 0..count {
            total += i;
        }
        total
    });
    total.wrapping_add(value)
}

#[kernel]
fn unrolled(#[comptime] plan: &[i32], mut value: i32) -> i32 {
    for &delta in plan {
        value = value.wrapping_add(delta);
    }
    value
}

#[kernel]
fn recursive(#[comptime] depth: i32, value: i32) -> i32 {
    if depth > 0 {
        recursive(depth - 1, value)
    } else {
        value
    }
}

#[kernel]
fn ping(#[comptime] step: i32, value: i32) -> i32 {
    if value > 0 {
        pong(step, value - step)
    } else {
        value
    }
}

#[kernel]
fn pong(#[comptime] step: i32, value: i32) -> i32 {
    ping(step, value)
}

/// CORE-21: a call that needs the specialization it is part of could only
/// be staged by expanding it forever. It is reported as recursion on the
/// first repeated call, not after 64 nested expansions.
#[test]
fn recursion_with_the_same_specialization_is_reported_as_recursion() {
    fn check<B: Codegen>() {
        let error = ping::build::<B>(ping::BuildArgs { step: 1 })
            .err()
            .expect("recursive specialization must fail");
        assert!(error.to_string().contains("calls itself"), "{error}");
    }
    splice_testkit::each_backend!(B => check::<B>());
    let error = ping::ir(1).unwrap_err();
    assert!(error.to_string().contains("calls itself"), "{error}");
    assert_eq!(ping(1, 3), 0);
    // A recursion that changes its arguments ends after 64 expansions.
    let error = recursive::ir(100).unwrap_err();
    assert!(error.to_string().contains("64 nested"), "{error}");
}

#[kernel]
fn runtime_loop(n: i32) -> i32 {
    let mut i = 0;
    while i < n {
        i += 1;
    }
    i
}

static TICKS: AtomicUsize = AtomicUsize::new(0);

fn tick() -> i32 {
    TICKS.fetch_add(1, Ordering::Relaxed);
    1
}

#[kernel]
fn opaque_in_host_loop(#[comptime] iterations: i32) -> i32 {
    let mut i = 0;
    while i < iterations {
        i += tick();
    }
    i
}

fn check_limits<B: Codegen<Output: Executable>>() {
    let options = BuildOptions::host(Mode::Release).unwrap();
    let bad = count_by_twos::BuildArgs { target: 1 };
    assert!(count_by_twos::build_with_options::<B>(options, bad).is_err());
    assert!(count_by_twos::ir(1).is_err());
    assert!(count_by_twos::print(1).is_err());
    assert!(mandatory_loop::build::<B>(mandatory_loop::BuildArgs { target: 1 }).is_err());
    assert!(mandatory_loop::ir(1).is_err());

    let direct = mandatory_loop::build::<B>(mandatory_loop::BuildArgs { target: 6 }).unwrap();
    let portable = B::compile(&mandatory_loop::ir(6).unwrap()).unwrap();
    for code in [direct, portable] {
        assert_eq!(
            code.call(mandatory_loop::CallArgs { value: 1 }).unwrap(),
            16
        );
    }

    let good = count_by_twos::BuildArgs { target: 6 };
    let ir = count_by_twos::ir(6).unwrap();
    let direct = count_by_twos::build_with_options::<B>(options, good).unwrap();
    let portable = B::compile(&ir).unwrap();
    for code in [direct, portable] {
        let args = count_by_twos::CallArgs { value: 4 };
        assert_eq!(code.call(args).unwrap(), count_by_twos(6, 4));
    }

    let large = vec![1; 70_000];
    let over_budget = unrolled::BuildArgs { plan: &large };
    assert!(unrolled::build_with_options::<B>(options, over_budget).is_err());
    assert!(unrolled::ir(&large).is_err());
    assert!(unrolled::print(&large).is_err());
    let direct = unrolled::build::<B>(unrolled::BuildArgs { plan: &[1, 2] }).unwrap();
    assert_eq!(direct.call(unrolled::CallArgs { value: 4 }).unwrap(), 7);

    assert!(
        recursive::build_with_options::<B>(options, recursive::BuildArgs { depth: 100 }).is_err()
    );
    assert!(recursive::ir(100).is_err());
    let recursive = B::compile(&recursive::ir(2).unwrap()).unwrap();
    assert_eq!(recursive.call(recursive::CallArgs { value: 7 }).unwrap(), 7);

    let direct = runtime_loop::build::<B>(runtime_loop::BuildArgs {}).unwrap();
    let portable = B::compile(&runtime_loop::ir().unwrap()).unwrap();
    for code in [direct, portable] {
        assert_eq!(
            code.call(runtime_loop::CallArgs { n: 100_000 }).unwrap(),
            100_000
        );
    }
}

// `count_by_twos(1)` and `mandatory_loop(1)` never terminate without the
// iteration budget, and the budget is what the `is_err` assertions observe.
// A broken budget hangs this test; the CI job timeout reports it. A wall-clock
// watchdog would instead depend on how fast the host (or QEMU) runs codegen.
#[test]
fn only_specialization_obeys_the_iteration_and_expansion_budgets() {
    splice_testkit::each_backend!(B => check_limits::<B>());
}

#[kernel]
fn borrowed_call(value: i32) -> i32 {
    let delta = &tick();
    value.wrapping_add(*delta)
}

#[test]
fn opaque_calls_are_not_executed_in_host_loops() {
    assert!(opaque_in_host_loop::ir(1).is_err());
    assert!(borrowed_call::ir().is_err());
    assert!(
        opaque_in_host_loop::build::<Lancy>(opaque_in_host_loop::BuildArgs { iterations: 1 })
            .is_err()
    );
    assert_eq!(TICKS.load(Ordering::Relaxed), 0);
    splice_testkit::each_backend!(B => {
        let empty =
            opaque_in_host_loop::build::<B>(opaque_in_host_loop::BuildArgs { iterations: 0 })
                .unwrap();
        assert_eq!(empty.call(opaque_in_host_loop::CallArgs {}).unwrap(), 0);
    });
    assert_eq!(opaque_in_host_loop(1), 1);
    assert_eq!(TICKS.load(Ordering::Relaxed), 1);
}

#[derive(Default)]
struct CountingPlan {
    starts: Cell<usize>,
    advances: Cell<usize>,
}

struct CountingIter<'a> {
    plan: &'a CountingPlan,
    pending: Option<i32>,
}

impl Iterator for CountingIter<'_> {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        self.plan.advances.set(self.plan.advances.get() + 1);
        self.pending.take()
    }
}

impl<'a> IntoIterator for &'a CountingPlan {
    type Item = i32;
    type IntoIter = CountingIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.starts.set(self.starts.get() + 1);
        CountingIter {
            plan: self,
            pending: Some(3),
        }
    }
}

#[kernel]
fn opaque_iterator(#[comptime] plan: &CountingPlan, mut value: i32) -> i32 {
    for delta in plan {
        value = value.wrapping_add(delta);
    }
    value
}

#[test]
fn specialization_rejects_an_opaque_iterator_before_constructing_or_advancing_it() {
    let plan = CountingPlan::default();
    splice_testkit::each_backend!(B => assert!(
        opaque_iterator::build::<B>(opaque_iterator::BuildArgs { plan: &plan }).is_err()
    ));
    assert!(opaque_iterator::ir(&plan).is_err());
    assert!(opaque_iterator::print(&plan).is_err());
    assert_eq!((plan.starts.get(), plan.advances.get()), (0, 0));

    assert_eq!(opaque_iterator(&plan, 5), 8);
    assert_eq!((plan.starts.get(), plan.advances.get()), (1, 2));
}

#[derive(splice::KernelType)]
struct Steps {
    values: [i32; 1],
}

// A described type lists its own fields, so `plan.values` (a field of the
// `Deref` target) is rejected without calling `deref`.
#[derive(splice::KernelType)]
struct IndirectSteps {
    reads: Cell<usize>,
    steps: Steps,
}

impl std::ops::Deref for IndirectSteps {
    type Target = Steps;

    fn deref(&self) -> &Steps {
        self.reads.set(self.reads.get() + 1);
        &self.steps
    }
}

#[kernel]
fn indirect_steps(#[comptime] plan: &IndirectSteps, mut value: i32) -> i32 {
    for &step in plan.values.iter() {
        value = value.wrapping_add(step);
    }
    value
}

#[kernel]
fn indirect_field(#[comptime] plan: &IndirectSteps, value: i32) -> i32 {
    let values = &plan.values;
    value.wrapping_add(values[0])
}

impl IndirectSteps {
    fn mark(&self) {
        self.reads.set(self.reads.get() + 1);
    }
}

#[kernel]
fn discarded_method(#[comptime] plan: &IndirectSteps, value: i32) -> i32 {
    plan.mark();
    value
}

#[kernel]
fn described_step(#[comptime] plan: &Steps) -> i32 {
    let Steps { values } = plan;
    comptime!((*values)[0])
}

#[test]
fn checking_a_field_does_not_invoke_custom_deref() {
    let plan = IndirectSteps {
        reads: Cell::new(0),
        steps: Steps { values: [3] },
    };
    assert!(indirect_steps::ir(&plan).is_err());
    assert!(indirect_steps::build::<Lancy>(indirect_steps::BuildArgs { plan: &plan }).is_err());
    assert!(indirect_steps::print(&plan).is_err());
    assert!(indirect_field::ir(&plan).is_err());
    assert!(indirect_field::build::<Lancy>(indirect_field::BuildArgs { plan: &plan }).is_err());
    assert!(discarded_method::print(&plan).is_err());
    assert_eq!(
        described_step::build::<Lancy>(described_step::BuildArgs { plan: &plan.steps })
            .unwrap()
            .call(described_step::CallArgs {})
            .unwrap(),
        3
    );
    assert_eq!(plan.reads.get(), 0);
    assert_eq!(indirect_steps(&plan, 5), 8);
    assert_eq!(plan.reads.get(), 1);
}

#[kernel]
fn countdown(#[comptime] step: i32, value: i32) -> i32 {
    if value > 0 {
        countdown(step, value - step)
    } else {
        value
    }
}

#[kernel]
fn unrolled_countdown(#[comptime] step: i32, #[comptime] value: i32, x: i32) -> i32 {
    if value > 0 {
        unrolled_countdown(step, value - step, x) + 1
    } else {
        x
    }
}

#[kernel]
fn known_countdown(x: i32) -> i32 {
    countdown(1, 3) + x
}

/// A kernel that calls itself directly expands inline. Expanding the same
/// specialization inside itself is reported as recursion at once, while
/// recursion that changes a comptime argument still unrolls.
#[test]
fn direct_recursion_with_the_same_specialization_is_reported_as_recursion() {
    fn check<B: Codegen<Output: Executable>>() {
        let Err(error) = countdown::build::<B>(countdown::BuildArgs { step: 1 }) else {
            panic!("unbounded recursion built");
        };
        assert!(
            error
                .to_string()
                .contains("calls itself with the same specialization"),
            "{error}"
        );
        let code =
            unrolled_countdown::build::<B>(unrolled_countdown::BuildArgs { step: 1, value: 30 })
                .unwrap();
        assert_eq!(code.call(unrolled_countdown::CallArgs { x: 2 }), Ok(32));
        // Known runtime arguments fold the condition, so each step differs.
        let code = known_countdown::build::<B>(known_countdown::BuildArgs {}).unwrap();
        assert_eq!(code.call(known_countdown::CallArgs { x: 5 }), Ok(5));
    }
    splice_testkit::each_backend!(B => check::<B>());
    let Err(error) = countdown::ir(1) else {
        panic!("unbounded recursion staged");
    };
    assert!(
        error
            .to_string()
            .contains("calls itself with the same specialization"),
        "{error}"
    );
}
