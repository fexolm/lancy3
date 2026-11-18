use super::super::profile::SAMPLE_CALLS;
use super::*;
use crate::input::{Binary, FunctionBuilder};

fn multiply() -> Function {
    let mut b = FunctionBuilder::new(2);
    let result = b.binary(Binary::Mul, b.arg(0), b.arg(1));
    b.ret(result);
    b.finish().unwrap()
}

#[test]
fn observes_before_compilation_and_recovers_on_the_first_mismatch() {
    let f = AdaptiveScalar::<2>::profiled(multiply()).unwrap();
    for x in 0..i32::from(SAMPLE_CALLS) {
        assert_eq!(f.tier(), Tier::Baseline);
        assert_eq!(f.invoke([x, -7]), x * -7);
    }
    f.wait_for_optimization().unwrap();
    assert_eq!(f.tier(), Tier::Specialized);
    assert_eq!(f.invoke([1024, -7]), -7168);
    assert_eq!(f.tier(), Tier::Specialized);
    assert_eq!(f.invoke([1024, 13]), 13312);
    assert_eq!(f.tier(), Tier::Generic);
    assert_eq!(f.invoke([1024, -7]), -7168);
    assert_eq!(f.tier(), Tier::Generic);
}

#[test]
fn variable_arguments_and_early_waits_use_generic_code() {
    for samples in [0, 1, SAMPLE_CALLS - 1, SAMPLE_CALLS] {
        let f = AdaptiveScalar::<2>::profiled(multiply()).unwrap();
        for i in 0..i32::from(samples) {
            assert_eq!(f.invoke([i, -i]), -i * i);
        }
        f.wait_for_optimization().unwrap();
        assert_eq!(f.tier(), Tier::Generic);
        assert_eq!(f.invoke([12, 13]), 156);
    }
}

#[test]
fn concurrent_invocations_share_the_sample_and_check_every_entry() {
    let f = AdaptiveScalar::<2>::profiled(multiply()).unwrap();
    let gate = std::sync::Barrier::new(8);
    std::thread::scope(|s| {
        for i in 0..8 {
            let (f, gate) = (&f, &gate);
            s.spawn(move || {
                gate.wait();
                for j in 0..4 {
                    let x = i * 4 + j;
                    assert_eq!(f.invoke([5, x]), 5 * x);
                }
            });
        }
    });
    f.wait_for_optimization().unwrap();
    assert_eq!(f.tier(), Tier::Specialized);
    std::thread::scope(|s| {
        for x in 0..8 {
            let (f, gate) = (&f, &gate);
            s.spawn(move || {
                gate.wait();
                assert_eq!(f.invoke([x, 11]), x * 11);
            });
        }
    });
    assert_eq!(f.tier(), Tier::Generic);
}
