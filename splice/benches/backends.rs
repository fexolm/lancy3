//! Compile the same staged function with both JITs and time whole invocations.
#[path = "../examples/relational/mod.rs"]
mod relational;

use relational::{Catalog, pricing_plan, query};
use splice::{BufferArg, Codegen, Executable, Function, kernel};
use splice_lancy::Lancy;
use splice_llvm::Llvm;
use std::hint::black_box;
use std::time::{Duration, Instant};

#[kernel(batch)]
fn line_total(#[comptime] minimum: i32, price: i32, quantity: i32) -> i32 {
    if quantity >= minimum {
        price.wrapping_mul(quantity)
    } else {
        0
    }
}

fn timed<T>(make: impl FnOnce() -> Result<T, String>) -> Result<(T, Duration), String> {
    let start = Instant::now();
    let result = make()?;
    Ok((result, start.elapsed()))
}

fn median_us(mut samples: Vec<Duration>) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2].as_secs_f64() * 1e6
}

fn compare(
    name: &str,
    stage: impl Fn() -> Result<Function, String>,
    run: impl Fn(&dyn Executable, &mut [i32]) -> Result<usize, String>,
    expected: &[i32],
    words: usize,
) -> Result<(), String> {
    const SAMPLES: usize = 15;
    const CALLS: u32 = 32;
    let ir = stage()?;
    let lancy = Lancy::compile(&ir)?;
    let llvm = Llvm::compile(&ir)?;
    let mut arenas = [vec![-1; words], vec![-1; words]];
    let kernels: [&dyn Executable; 2] = [&lancy, &llvm];
    let verify = |i: usize, arena: &mut [i32]| -> Result<(), String> {
        let used = run(kernels[i], arena)?;
        if arena.get(..used) != Some(expected) {
            return Err(format!("{name}: incorrect backend {i} output"));
        }
        Ok(())
    };
    for (i, arena) in arenas.iter_mut().enumerate() {
        verify(i, arena)?;
    }
    let mut staging = Vec::new();
    let mut compiling = [Vec::new(), Vec::new()];
    let mut executing = [Vec::new(), Vec::new()];
    for sample in 0..SAMPLES {
        staging.push(timed(&stage)?.1);
        // Alternate order, reuse inputs and scratch, and exclude destruction of
        // the executable from the compilation timer. Each compilation is fresh.
        for i in [sample % 2, 1 - sample % 2] {
            compiling[i].push(if i == 0 {
                timed(|| Lancy::compile(black_box(&ir)))?.1
            } else {
                timed(|| Llvm::compile(black_box(&ir)))?.1
            });
            let (_, elapsed) = timed(|| {
                for _ in 0..CALLS {
                    black_box(run(kernels[i], black_box(&mut arenas[i]))?);
                }
                Ok(())
            })?;
            executing[i].push(elapsed / CALLS);
        }
    }
    for (i, arena) in arenas.iter_mut().enumerate() {
        verify(i, arena)?;
    }
    println!("{name} | stage {:.2} us", median_us(staging));
    for ((backend, build), execute) in ["Lancy", "LLVM"].into_iter().zip(compiling).zip(executing) {
        println!(
            "  {backend:5} | compile {:>10.2} us | execute {:>9.2} us",
            median_us(build),
            median_us(execute)
        );
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let (major, minor, patch) = Llvm::version()?;
    println!(
        "LLVM {major}.{minor}.{patch}, {} {}, median of 15 samples, 32 calls per execution sample",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    let price: Vec<i32> = (0..65_536).map(|i| i % 997).collect();
    let qty: Vec<i32> = (0..65_536).map(|i| i % 13).collect();
    let expected: Vec<_> = price
        .iter()
        .zip(&qty)
        .map(|(&p, &q)| line_total(2, p, q))
        .collect();
    compare(
        "line totals (65536 rows)",
        || line_total::ir(2),
        |kernel, dst| {
            kernel
                .call(
                    &[],
                    &mut [
                        BufferArg::Read(&price),
                        BufferArg::Read(&qty),
                        BufferArg::Write(dst),
                    ],
                )
                .map(|n| n as usize)
        },
        &expected,
        price.len(),
    )?;

    let sales: Vec<_> = (0..8192).map(|i| (i % 512, i % 8)).collect();
    let prices: Vec<_> = (0..512).map(|i| (i, i + 1)).collect();
    let flags: Vec<_> = (0..512).map(|i| (i, i % 2)).collect();
    let promos: Vec<_> = (0..512).map(|i| (i, i % 5)).collect();
    let adjustments: Vec<_> = (0..512).map(|i| (i, i % 16)).collect();
    let catalog = Catalog::new(&[&sales, &prices, &flags, &promos, &adjustments])?;
    let plan = pricing_plan();
    let mut expected = vec![0; 131_072];
    let count = query(&plan, catalog.offsets(), catalog.data(), &mut expected);
    let rows = usize::try_from(count).map_err(|_| "reference query ran out of workspace")?;
    compare(
        "pricing (5 tables, 4 joins, 8192 sales)",
        || query::ir(&plan),
        |kernel, arena| catalog.run(kernel, arena).map(<[i32]>::len),
        &expected[..rows * 2],
        expected.len(),
    )
}
