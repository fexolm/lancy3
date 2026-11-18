#[path = "../examples/support/mod.rs"]
mod support;

use lancy_jit::{AdaptiveKernel, Assumption, NativeKernel};
use std::{hint::black_box, time::Instant};

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn compile_cost(label: &str, mut compile: impl FnMut() -> NativeKernel) {
    let mut times = Vec::new();
    for _ in 0..101 {
        let start = Instant::now();
        let kernel = black_box(compile());
        times.push(start.elapsed().as_secs_f64() * 1e6);
        drop(kernel);
    }
    println!("{label:>14}   {:8.2} us/compile (median)", median(times));
}

fn scan_cost(label: &str, kernel: &NativeKernel, columns: &[&[i32]], out: &mut [i32]) {
    let rows = out.len();
    let scan = |out: &mut [i32]| {
        assert_eq!(
            kernel.run(black_box(columns), black_box(out)).unwrap(),
            rows
        );
        black_box(&*out);
    };
    scan(out);
    let mut samples = Vec::new();
    for _ in 0..15 {
        let start = Instant::now();
        for _ in 0..128 {
            scan(out);
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / (128 * rows) as f64);
    }
    println!(
        "{label:>14}   {:8.3} ns/row (median), {} code bytes",
        median(samples),
        kernel.code_size()
    );
}

fn first_result_cost(columns: &[&[i32]], out: &mut [i32]) {
    let mut samples = Vec::with_capacity(101);
    for _ in 0..101 {
        let start = Instant::now();
        let kernel = AdaptiveKernel::compile(support::revenue()).unwrap();
        kernel.run(black_box(columns), black_box(out)).unwrap();
        samples.push(start.elapsed().as_secs_f64() * 1e6);
        // Drain this request outside the timing interval so the next sample
        // starts with an idle worker.
        kernel.wait_for_optimization().unwrap();
    }
    println!(
        "New HIR to first {} output rows: {:.2} us (median, shared worker).",
        out.len(),
        median(samples)
    );
}

fn main() {
    let program = support::revenue();
    let assumptions = [Assumption {
        column: 1,
        value: 3,
    }];
    let scalar = || NativeKernel::baseline(&program).unwrap();
    let generic = || NativeKernel::specialize(&program, &[]).unwrap();
    let specialized = || NativeKernel::specialize(&program, &assumptions).unwrap();
    println!(
        "Revenue projection, {} HIR operations; HIR construction timed separately.",
        program.ops().len()
    );
    let mut builds = Vec::new();
    for _ in 0..101 {
        let start = Instant::now();
        for _ in 0..100 {
            black_box(support::revenue());
        }
        builds.push(start.elapsed().as_secs_f64() * 1e9 / 100.0);
    }
    println!(
        "{:>14}   {:8.1} ns/build+drop (median)",
        "HIR",
        median(builds)
    );
    compile_cost("scalar", scalar);
    compile_cost("generic SIMD", generic);
    compile_cost("guarded SIMD", specialized);
    let price: Vec<_> = (0..65_536i32).map(|i| i % 8192).collect();
    let quantity = vec![3; price.len()];
    let stock: Vec<_> = (0..price.len()).map(|i| i32::from(i % 7 != 0)).collect();
    let columns = [&price[..], &quantity[..], &stock[..]];
    let mut out = vec![0; price.len()];
    let expected: Vec<_> = price
        .iter()
        .zip(&stock)
        .map(|(p, s)| if *s > 0 { p * 3 } else { 0 })
        .collect();
    println!("{} rows per batch, same buffers for each tier.", out.len());
    for (label, kernel) in [
        ("scalar", scalar()),
        ("generic SIMD", generic()),
        ("guarded SIMD", specialized()),
    ] {
        out.fill(i32::MIN);
        scan_cost(label, &kernel, &columns, &mut out);
        assert_eq!(out, expected);
    }
    // A uniform predicate lets specialization remove both price and quantity
    // loads. Measure the same query and check the entire resulting projection.
    let empty_stock = vec![0; price.len()];
    let out_of_stock = [&price[..], &quantity[..], &empty_stock[..]];
    let sold_out = NativeKernel::specialize(
        &program,
        &[Assumption {
            column: 2,
            value: 0,
        }],
    )
    .unwrap();
    println!("Same query, all stock values zero.");
    for (label, kernel) in [("generic SIMD", generic()), ("guarded SIMD", sold_out)] {
        out.fill(i32::MIN);
        scan_cost(label, &kernel, &out_of_stock, &mut out);
        assert!(out.iter().all(|v| *v == 0));
    }
    let first_batch = columns.map(|col| &col[..64]);
    first_result_cost(&first_batch, &mut out[..64]);
}
