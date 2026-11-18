mod audio;
mod decoder;
mod joins;
mod rules;
mod sql;

use std::hint::black_box;
use std::time::{Duration, Instant};

fn percentile(samples: &mut [Duration], percentage: usize) -> Duration {
    samples.sort_unstable();
    samples[(samples.len() * percentage).div_ceil(100) - 1]
}

pub fn build<T>(label: &str, mut make: impl FnMut() -> Result<T, String>) -> Result<T, String> {
    let mut samples = Vec::with_capacity(31);
    for _ in 0..31 {
        let start = Instant::now();
        let kernel = black_box(make()?);
        samples.push(start.elapsed());
        drop(kernel);
    }
    println!(
        "build | {label} | {:.2} us p50",
        percentile(&mut samples, 50).as_secs_f64() * 1e6
    );
    make()
}

fn timed(f: &mut impl FnMut(&mut [i32]), output: &mut [i32], repetitions: u32) -> Duration {
    let start = Instant::now();
    for _ in 0..repetitions {
        f(black_box(&mut *output));
        black_box(&*output);
    }
    start.elapsed()
}

fn verify(label: &str, expected: &[i32], actual: &[i32]) -> Result<(), String> {
    if let Some(i) = expected.iter().zip(actual).position(|(a, b)| a != b) {
        return Err(format!(
            "{label}: output {i} is {}, expected {}",
            actual[i], expected[i]
        ));
    }
    Ok(())
}

/// Compare whole outputs before and after timing. Closures reset persistent
/// state to the same starting state on each invocation when required.
pub fn compare(
    label: &str,
    units: usize,
    expected: &[i32],
    cpp: impl FnMut(&mut [i32]),
    ir: impl FnMut(&mut [i32]),
) -> Result<(), String> {
    compare_with_workspace(label, units, expected, expected.len(), cpp, ir)
}

/// Both implementations return their packed rows at the start of the same
/// size workspace. This includes native scratch traffic in the measurement
/// and avoids copying JIT output into a second allocation for the harness.
fn compare_with_workspace(
    label: &str,
    units: usize,
    expected: &[i32],
    workspace_words: usize,
    mut cpp: impl FnMut(&mut [i32]),
    mut ir: impl FnMut(&mut [i32]),
) -> Result<(), String> {
    assert!(units > 0 && workspace_words >= expected.len());
    let mut a: Vec<_> = expected.iter().map(|v| !v).collect();
    a.resize(workspace_words, 0);
    let mut b = a.clone();
    cpp(&mut a);
    ir(&mut b);
    verify(label, expected, &a)?;
    verify(label, expected, &b)?;

    // Calibrate on the faster implementation so both timing intervals are
    // long enough for the timer. Use identical iteration counts and alternate order.
    let mut repetitions = 1;
    while repetitions < 65536 {
        let elapsed = timed(&mut cpp, &mut a, repetitions).min(timed(&mut ir, &mut b, repetitions));
        if elapsed >= Duration::from_micros(150) {
            break;
        }
        repetitions *= 2;
    }
    let mut cpp_times = Vec::new();
    let mut ir_times = Vec::new();
    for i in 0..31 {
        if i % 2 == 0 {
            cpp_times.push(timed(&mut cpp, &mut a, repetitions));
            ir_times.push(timed(&mut ir, &mut b, repetitions));
        } else {
            ir_times.push(timed(&mut ir, &mut b, repetitions));
            cpp_times.push(timed(&mut cpp, &mut a, repetitions));
        }
    }
    let units_timed = f64::from(repetitions) * units as f64;
    let cpp_ns = percentile(&mut cpp_times, 50).as_secs_f64() * 1e9 / units_timed;
    let ir_ns = percentile(&mut ir_times, 50).as_secs_f64() * 1e9 / units_timed;

    // Separate individual invocation measurements preserve callback latency
    // variation. These include clock overhead, especially visible on tiny inputs.
    cpp_times.clear();
    ir_times.clear();
    for i in 0..201 {
        if i % 2 == 0 {
            cpp_times.push(timed(&mut cpp, &mut a, 1));
            ir_times.push(timed(&mut ir, &mut b, 1));
        } else {
            ir_times.push(timed(&mut ir, &mut b, 1));
            cpp_times.push(timed(&mut cpp, &mut a, 1));
        }
    }
    verify(label, expected, &a)?;
    verify(label, expected, &b)?;
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    println!(
        "case | {label} | {units} | {cpp_ns:.3} | {ir_ns:.3} | {:.2}x | {:.2}/{:.2} | {:.2}/{:.2}",
        ir_ns / cpp_ns,
        us(percentile(&mut cpp_times, 50)),
        us(percentile(&mut cpp_times, 95)),
        us(percentile(&mut ir_times, 50)),
        us(percentile(&mut ir_times, 95))
    );
    Ok(())
}

fn main() -> Result<(), String> {
    if cfg!(feature = "avx2") && !is_x86_feature_detected!("avx2") {
        return Err("this C++ benchmark build requires AVX2".into());
    }
    let selected = std::env::args().nth(1);
    println!(
        "C++: {} | AVX2 build: {} | Rust debug assertions: {}",
        env!("CUSTOMER_CXX"),
        cfg!(feature = "avx2"),
        cfg!(debug_assertions)
    );
    println!(
        "case | workload | units/call | C++ ns/unit | IR ns/unit | IR/C++ | C++ us/call p50/p95 | IR us/call p50/p95"
    );
    type Workload = fn() -> Result<(), String>;
    let workloads: [(&str, Workload); 5] = [
        ("rules", rules::run),
        ("decoder", decoder::run),
        ("audio", audio::run),
        ("sql", sql::run),
        ("joins", joins::run),
    ];
    if let Some(name) = selected.as_ref()
        && !workloads.iter().any(|(n, _)| n == name)
    {
        return Err(format!(
            "unknown workload {name}; choose rules, decoder, audio, sql or joins"
        ));
    }
    for (name, run) in workloads {
        if selected.as_deref().is_none_or(|s| s == name) {
            run()?;
        }
    }
    Ok(())
}
