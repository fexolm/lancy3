//! Measure direct builds from staged Rust, compilation from one portable IR,
//! and execution of the native kernel produced by each route.
#[path = "../examples/relational/mod.rs"]
mod relational;

use relational::{Catalog, pricing_plan, query};
use splice::{BufferArg, Codegen, Executable, Function, kernel};
use splice_lancy::{Lancy, LancyKernel};
use splice_llvm::{Llvm, LlvmKernel};
use std::hint::black_box;
use std::time::{Duration, Instant};

#[kernel]
fn line_total(#[comptime] minimum: i32, price: i32, quantity: i32) -> i32 {
    if quantity >= minimum {
        price.wrapping_mul(quantity)
    } else {
        0
    }
}

#[kernel]
fn line_totals(
    #[comptime] minimum: i32,
    prices: &[i32],
    quantities: &[i32],
    out: &mut [i32],
) -> i32 {
    for i in 0..out.len() as i32 {
        out[i as usize] = line_total(minimum, prices[i as usize], quantities[i as usize]);
    }
    out.len() as i32
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
    build_lancy: impl Fn() -> Result<LancyKernel, String>,
    build_llvm: impl Fn() -> Result<LlvmKernel, String>,
    run: impl Fn(&dyn Executable, &mut [i32]) -> Result<usize, String>,
    expected: &[i32],
    words: usize,
) -> Result<(), String> {
    const SAMPLES: usize = 15;
    const CALLS: u32 = 32;
    let ir = stage()?;
    let lancy = Lancy::compile(&ir)?;
    let llvm = Llvm::compile(&ir)?;
    let direct_lancy = build_lancy()?;
    let direct_llvm = build_llvm()?;
    let kernels: [&dyn Executable; 4] = [&lancy, &llvm, &direct_lancy, &direct_llvm];
    let names = [
        "Lancy from IR",
        "LLVM from IR",
        "Lancy direct",
        "LLVM direct",
    ];
    let mut arenas: [_; 4] = std::array::from_fn(|_| vec![-1; words]);
    let verify = |i: usize, arena: &mut [i32]| -> Result<(), String> {
        for (word, &want) in arena.iter_mut().zip(expected) {
            *word = !want;
        }
        let used = run(kernels[i], arena)?;
        if arena.get(..used) != Some(expected) {
            return Err(format!("{name}: incorrect {} output", names[i]));
        }
        Ok(())
    };
    for (i, arena) in arenas.iter_mut().enumerate() {
        verify(i, arena)?;
    }
    let mut staging = Vec::new();
    let mut compiling: [_; 4] = std::array::from_fn(|_| Vec::new());
    let mut executing: [_; 4] = std::array::from_fn(|_| Vec::new());
    for sample in 0..SAMPLES {
        staging.push(timed(&stage)?.1);
        // Rotate routes. Each build is fresh and timed through installation.
        // Destruction happens after the elapsed time has been captured.
        for step in 0..kernels.len() {
            let i = (sample + step) % kernels.len();
            let elapsed = match i {
                0 => timed(|| Lancy::compile(black_box(&ir)))?.1,
                1 => timed(|| Llvm::compile(black_box(&ir)))?.1,
                2 => timed(&build_lancy)?.1,
                3 => timed(&build_llvm)?.1,
                _ => unreachable!(),
            };
            compiling[i].push(elapsed);
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
    println!("{name} | stage portable IR {:.2} us", median_us(staging));
    for ((backend, build), execute) in names.into_iter().zip(compiling).zip(executing) {
        println!(
            "  {backend:13} | build {:>10.2} us | execute {:>9.2} us",
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
    println!(
        "Direct build includes staging and codegen. From-IR build starts with the shared, verified IR."
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
        || line_totals::ir(2),
        || line_totals::build::<Lancy>(2),
        || line_totals::build::<Llvm>(2),
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
        || query::build::<Lancy>(&plan),
        || query::build::<Llvm>(&plan),
        |kernel, arena| catalog.run(kernel, arena).map(<[i32]>::len),
        &expected[..rows * 2],
        expected.len(),
    )
}
