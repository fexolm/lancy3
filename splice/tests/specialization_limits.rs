use splice::{BuildOptions, Codegen, Executable, kernel};
use splice_lancy::Lancy;
use splice_llvm::Llvm;
use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
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
    let options = BuildOptions::Release;
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

#[test]
fn only_specialization_obeys_the_iteration_and_expansion_budgets() {
    let (send, receive) = mpsc::sync_channel(1);
    let worker = thread::spawn(move || {
        check_limits::<Lancy>();
        check_limits::<Llvm>();
        send.send(()).unwrap();
    });
    receive
        .recv_timeout(Duration::from_secs(10))
        .expect("specialization must terminate");
    worker.join().unwrap();
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
    let empty =
        opaque_in_host_loop::build::<Llvm>(opaque_in_host_loop::BuildArgs { iterations: 0 })
            .unwrap();
    assert_eq!(empty.call(opaque_in_host_loop::CallArgs {}).unwrap(), 0);
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
    assert!(opaque_iterator::build::<Lancy>(opaque_iterator::BuildArgs { plan: &plan }).is_err());
    assert!(opaque_iterator::build::<Llvm>(opaque_iterator::BuildArgs { plan: &plan }).is_err());
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

#[kernel]
fn indirect_deref(#[comptime] plan: &IndirectSteps) -> i32 {
    let steps = &**plan;
    steps.values[0]
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
    assert!(indirect_deref::ir(&plan).is_err());
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
