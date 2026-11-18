use super::*;
use crate::hir::Builder;
use std::thread;

#[test]
fn dead_inputs_do_not_trigger_guard_exits() {
    let mut b = Builder::new(2);
    let dead = b.input(0);
    let live = b.input(1);
    b.mul(live, dead);
    let kernel = AdaptiveKernel::compile(b.finish(live).unwrap()).unwrap();
    let mut out = [0; 7];
    kernel.run(&[&[2; 7], &[5; 7]], &mut out).unwrap();
    kernel.wait_for_optimization().unwrap();
    assert_eq!(
        kernel.stats().assumptions,
        [Assumption {
            column: 1,
            value: 5
        }]
    );
    kernel.run(&[&[99; 7], &[5; 7]], &mut out).unwrap();
    assert_eq!(kernel.stats().deoptimizations, 0);
    assert_eq!(out, [5; 7]);
}

fn plus_one() -> AdaptiveKernel {
    let mut builder = Builder::new(1);
    let input = builder.input(0);
    let one = builder.constant(1);
    let result = builder.add(input, one);
    AdaptiveKernel::compile(builder.finish(result).unwrap()).unwrap()
}

#[test]
fn empty_and_invalid_batches_preserve_cold_state() {
    let kernel = plus_one();
    let mut empty = [];
    kernel.run(&[&[]], &mut empty).unwrap();
    assert!(kernel.run(&[], &mut empty).is_err());
    let mut out = [55; 3];
    assert!(kernel.run(&[&[1, 2]], &mut out).is_err());
    assert_eq!(out, [55; 3]);
    kernel.wait_for_optimization().unwrap();
    assert_eq!(kernel.stats().tier, Tier::Baseline);
}

#[test]
fn concurrent_exits_from_retired_code_install_only_one_replacement() {
    let kernel = plus_one();
    kernel.run(&[&[4; 32]], &mut [0; 32]).unwrap();
    kernel.wait_for_optimization().unwrap();
    let version = kernel
        .inner
        .state
        .lock()
        .unwrap()
        .specialized
        .clone()
        .unwrap();
    assert_eq!(
        version.guards,
        [Assumption {
            column: 0,
            value: 4
        }]
    );

    // Pin the version as a real dispatcher does, then force late exits even if
    // the generic replacement is already installed before a reader starts.
    thread::scope(|scope| {
        for mismatch in [0, 1, 2, 5, 8, 12, 15, 16] {
            let kernel = &kernel;
            let version = &version;
            scope.spawn(move || {
                let mut input = [4_i32; 17];
                input[mismatch] = i32::MAX;
                let mut out = [0; 17];
                let written = kernel
                    .run_optimized(version, &[&input], 0, &mut out)
                    .unwrap();
                assert_eq!(written, out.len());
                assert_eq!(out, input.map(|v| v.wrapping_add(1)));
            });
        }
    });
    kernel.wait_for_optimization().unwrap();
    let stats = kernel.stats();
    assert_eq!(stats.tier, Tier::Generic);
    assert_eq!(stats.compilations, 2);
    assert_eq!(stats.deoptimizations, 8);
    assert_eq!(stats.optimized_rows + stats.baseline_rows, 32 + 8 * 17);
}

#[test]
fn explicit_wait_finishes_work_deferred_by_a_full_queue() {
    let kernel = plus_one();
    kernel.inner.state.lock().unwrap().work = Work::Deferred(Vec::new());
    kernel.wait_for_optimization().unwrap();
    assert_eq!(kernel.stats().tier, Tier::Generic);
}

#[test]
fn long_batch_recovers_from_a_guard_exit_at_a_nonzero_offset() {
    let kernel = plus_one();
    kernel.run(&[&[4; 32]], &mut [0; 32]).unwrap();
    kernel.wait_for_optimization().unwrap();
    let len = 4 * BASELINE_CHUNK + 5;
    let mut input = vec![4; len];
    for (i, value) in input[BASELINE_CHUNK + 7..].iter_mut().enumerate() {
        *value = (i as i32).wrapping_sub(111);
    }
    let mut out = vec![0; len];
    kernel.run(&[&input], &mut out).unwrap();
    kernel.wait_for_optimization().unwrap();
    for (&v, result) in input.iter().zip(out) {
        assert_eq!(result, v.wrapping_add(1));
    }
    let stats = kernel.stats();
    assert_eq!(stats.tier, Tier::Generic);
    assert_eq!(stats.deoptimizations, 1);
    assert!(stats.optimized_rows >= BASELINE_CHUNK as u64);
    assert_eq!(stats.baseline_rows + stats.optimized_rows, 32 + len as u64);
}

#[test]
fn probes_skip_variable_batches_and_respecialize_when_the_distribution_changes() {
    let kernel = plus_one();
    let run = |input: &[i32; 32]| {
        let mut out = [0; 32];
        kernel.run(&[input], &mut out).unwrap();
        assert_eq!(out, input.map(|v| v.wrapping_add(1)));
    };
    run(&[4; 32]);
    kernel.wait_for_optimization().unwrap();
    run(&[5; 32]);
    kernel.wait_for_optimization().unwrap();
    assert_eq!(kernel.stats().tier, Tier::Generic);
    let changing = std::array::from_fn(|i| i as i32);
    for _ in 0..PROBE_INTERVAL {
        run(&changing);
    }
    kernel.wait_for_optimization().unwrap();
    assert_eq!(kernel.stats().compilations, 2);
    for _ in 0..PROBE_INTERVAL {
        run(&[11; 32]);
    }
    kernel.wait_for_optimization().unwrap();
    assert_eq!(
        kernel.stats().assumptions,
        [Assumption {
            column: 0,
            value: 11
        }]
    );
    assert_eq!(kernel.stats().compilations, 3);
    let baseline = kernel.stats().baseline_rows;
    run(&[12; 32]); // Cached generic immediately handles the whole suffix.
    for _ in 1..(2 * PROBE_INTERVAL) {
        run(&[12; 32]);
    }
    kernel.wait_for_optimization().unwrap();
    assert_eq!(kernel.stats().tier, Tier::Generic);
    assert_eq!(kernel.stats().compilations, 3);
    assert_eq!(kernel.stats().baseline_rows, baseline);
    run(&[12; 32]);
    kernel.wait_for_optimization().unwrap();
    assert_eq!(
        kernel.stats().assumptions,
        [Assumption {
            column: 0,
            value: 12
        }]
    );
    assert_eq!(kernel.stats().compilations, 4);
}

#[test]
fn pending_respecialization_serves_concurrent_calls_on_generic_simd() {
    let kernel = plus_one();
    let changing: [i32; 32] = std::array::from_fn(|i| i as i32);
    kernel.run(&[&changing], &mut [0; 32]).unwrap();
    kernel.wait_for_optimization().unwrap();
    for _ in 1..PROBE_INTERVAL {
        kernel.run(&[&changing], &mut [0; 32]).unwrap();
    }
    // Pause the profiling caller at the point where it owns the request.
    // The same Pending state spans profiling, queuing, and native compilation.
    let (generic, request) = kernel.dispatch(true);
    assert!(generic.unwrap().guards.is_empty());
    assert!(matches!(request, Some(Request::Profile)));
    let before = kernel.stats();
    assert_eq!(before.tier, Tier::Generic);
    assert!(before.optimization_pending);
    let mut out = [0; 32];
    kernel.run(&[&changing], &mut out).unwrap();
    assert_eq!(out, changing.map(|v| v + 1));
    let after = kernel.stats();
    assert!(after.optimization_pending);
    assert_eq!(before.baseline_rows, after.baseline_rows);
    assert_eq!(before.optimized_rows + 32, after.optimized_rows);
    kernel.submit(
        vec![Assumption {
            column: 0,
            value: 99,
        }],
        false,
    );
    kernel.wait_for_optimization().unwrap();
    assert_eq!(kernel.stats().tier, Tier::Specialized);
}
