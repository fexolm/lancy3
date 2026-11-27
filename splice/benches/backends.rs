//! JIT compilation and warm execution on the same inputs for all three backends.

/// One workload's builders with the arguments written once: `$b` names each
/// benchmarked backend in turn.
macro_rules! builders {
    ($b:ident => $build:expr) => {
        $crate::Builders {
            lancy: &|| {
                type $b = splice::lancy::Lancy;
                $build
            },
            cranelift: &|| {
                type $b = splice::cranelift::Cranelift;
                $build
            },
            llvm: &|| {
                type $b = splice::llvm::LlvmManualSimd;
                $build
            },
        }
    };
}

mod common;
mod gaps;
#[path = "large/mod.rs"]
mod large;
#[path = "../examples/workloads/matmul.rs"]
mod matmul;
#[path = "../examples/workloads/probe.rs"]
mod probe;
#[path = "../examples/relational/mod.rs"]
mod relational;

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use relational::{Catalog, pricing_plan, query};
use splice::cranelift::CraneliftKernel;
use splice::lancy::LancyKernel;
use splice::llvm::LlvmKernel;
use splice::{
    BufferArg, BuildError, Codegen, Executable, KernelScalar, TypedIr, TypedKernel, kernel,
};
use std::{cell::LazyCell, hint::black_box, ops::Not};

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

struct Builders<'a, S> {
    lancy: &'a dyn Fn() -> Result<TypedKernel<S, LancyKernel>, BuildError>,
    llvm: &'a dyn Fn() -> Result<TypedKernel<S, LlvmKernel>, BuildError>,
    cranelift: &'a dyn Fn() -> Result<TypedKernel<S, CraneliftKernel>, BuildError>,
}

struct Workload<'a, Run, T> {
    name: &'static str,
    elements: usize,
    workspace_words: usize,
    expected: &'a dyn Fn() -> &'a [T],
    equal: fn(&[T], &[T]) -> bool,
    run: Run,
}

fn bench_compile<K, E: std::fmt::Debug>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    build: impl Fn() -> Result<K, E>,
) {
    group.bench_function(backend, |b| {
        // Keep installation in the sample. Release the executable mapping
        // after the timer stops and before starting the next JIT compilation.
        b.iter_batched(
            || (),
            |_| build().expect("compile kernel"),
            BatchSize::PerIteration,
        );
    });
}

impl<Run, T> Workload<'_, Run, T>
where
    Run: Fn(&dyn Executable, &mut [T]) -> Result<usize, Box<dyn std::error::Error>>,
    T: Copy + Default + Not<Output = T>,
{
    fn verify(&self, kernel: &dyn Executable, workspace: &mut [T], backend: &str) {
        let expected = (self.expected)();
        for (word, &value) in workspace.iter_mut().zip(expected) {
            *word = !value;
        }
        let used = (self.run)(kernel, workspace).expect("execute kernel");
        assert_eq!(
            used,
            expected.len(),
            "{} / {backend}: result length",
            self.name
        );
        assert!(
            (self.equal)(&workspace[..used], expected),
            "{} / {backend}: incorrect output",
            self.name
        );
    }

    fn bench_execute<S, K: Executable>(
        &self,
        group: &mut BenchmarkGroup<'_, WallTime>,
        backend: &str,
        build: impl Fn() -> Result<TypedKernel<S, K>, BuildError>,
    ) {
        let mut state = None;
        group.bench_function(BenchmarkId::new(backend, self.elements), |b| {
            let (kernel, workspace) = state.get_or_insert_with(|| {
                let kernel = build().expect("compile execution benchmark").into_inner();
                let mut workspace = vec![!T::default(); self.workspace_words];
                self.verify(&kernel, &mut workspace, backend);
                (kernel, workspace)
            });
            b.iter(|| {
                (self.run)(black_box(&*kernel), black_box(workspace.as_mut_slice()))
                    .expect("execute kernel")
            });
        });
        if let Some((kernel, mut workspace)) = state {
            self.verify(&kernel, &mut workspace, backend);
        }
    }

    fn bench_route<S>(&self, c: &mut Criterion, route: &str, builds: Builders<'_, S>) {
        let mut compile = c.benchmark_group(format!("{}/compile/{route}", self.name));
        bench_compile(&mut compile, "lancy", builds.lancy);
        bench_compile(&mut compile, "cranelift", builds.cranelift);
        bench_compile(&mut compile, "llvm_manual_simd", builds.llvm);
        compile.finish();

        let mut execute = c.benchmark_group(format!("{}/execute/{route}", self.name));
        execute.throughput(Throughput::Elements(self.elements as u64));
        self.bench_execute(&mut execute, "lancy", builds.lancy);
        self.bench_execute(&mut execute, "cranelift", builds.cranelift);
        self.bench_execute(&mut execute, "llvm_manual_simd", builds.llvm);
        execute.finish();
    }

    fn compare<S>(
        &self,
        c: &mut Criterion,
        stage: impl Fn() -> Result<TypedIr<S>, BuildError>,
        builds: Builders<'_, S>,
    ) {
        c.bench_function(&format!("{}/stage_ir", self.name), |b| {
            b.iter_batched(
                || (),
                |_| stage().expect("stage IR"),
                BatchSize::PerIteration,
            );
        });
        self.bench_route(c, "build", builds);
        let ir = stage().expect("stage shared IR");
        self.bench_route(c, "from_ir", builders!(B => B::compile(black_box(&ir))));
    }
}

fn projection(c: &mut Criterion) {
    let prices: Vec<i32> = (0..65_536).map(|i| i % 997).collect();
    let quantities: Vec<i32> = (0..65_536).map(|i| i % 13).collect();
    let expected: Vec<_> = prices
        .iter()
        .zip(&quantities)
        .map(|(&p, &q)| line_total(2, p, q))
        .collect();
    let workload = Workload {
        name: "projection",
        elements: prices.len(),
        workspace_words: prices.len(),
        expected: &|| &expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, dst: &mut [i32]| {
            let count = kernel.call(
                &[],
                &mut [
                    BufferArg::read(&prices),
                    BufferArg::read(&quantities),
                    BufferArg::write(dst),
                ],
            )?;
            usize::try_from(i32::from_word(count)).map_err(|_| "negative projection length".into())
        },
    };
    workload.compare(
        c,
        || line_totals::ir(black_box(2)),
        builders!(B => line_totals::build::<B>(line_totals::BuildArgs {
            minimum: black_box(2),
        })),
    );
}

fn sql_plan(c: &mut Criterion) {
    const SALES: i32 = 28 * 1024 * 1024;
    const PRODUCTS: i32 = SALES / 16;
    // Keep the workspace identical across scalar and SIMD benchmark runs.
    let workspace_words =
        (4 * SALES + 3 * PRODUCTS) as usize + (PRODUCTS as usize).next_power_of_two();
    let plan = pricing_plan();
    let data = LazyCell::new(|| {
        let catalog = {
            let sales: Vec<_> = (0..SALES).map(|i| (i % PRODUCTS, i % 8)).collect();
            let prices: Vec<_> = (0..PRODUCTS).map(|i| (i, i + 1)).collect();
            let flags: Vec<_> = (0..PRODUCTS).map(|i| (i, i % 2)).collect();
            let promos: Vec<_> = (0..PRODUCTS).map(|i| (i, i % 5)).collect();
            let adjustments: Vec<_> = (0..PRODUCTS).map(|i| (i, i % 16)).collect();
            Catalog::new(&[&sales, &prices, &flags, &promos, &adjustments])
                .expect("build pricing catalog")
        };
        // Each odd product has one flag and one promotion. Derive the expected
        // rows directly from the fixture, independently of the query operators.
        let mut expected = Vec::with_capacity(SALES as usize * 3 / 4);
        for sale in 0..SALES {
            let product = sale % PRODUCTS;
            let quantity = sale % 8;
            if quantity >= 2 && product % 2 == 1 {
                let revenue = quantity.wrapping_mul(product + 1);
                let discount = (1 + product % 5) * (2 * (product % 16)).min(20);
                expected.extend([product, revenue.wrapping_add(discount).wrapping_add(1)]);
            }
        }
        assert_eq!(expected.len(), SALES as usize * 3 / 4);
        (catalog, expected.into_boxed_slice())
    });
    let workload = Workload {
        name: "sql_plan",
        elements: SALES as usize,
        workspace_words: workspace_words + plan.state_words(),
        expected: &|| &data.1,
        equal: |actual, expected| {
            // Every sale for a given product has the same quantity. Compare
            // its value and occurrence count independently of emission order.
            let mut products = vec![(0, 0u32); PRODUCTS as usize];
            for &[key, value] in expected.as_chunks::<2>().0 {
                let entry = &mut products[key as usize];
                entry.0 = value;
                entry.1 += 1;
            }
            actual.as_chunks::<2>().0.iter().all(|&[key, value]| {
                let Some(entry) = products.get_mut(key as usize) else {
                    return false;
                };
                if entry.0 != value || entry.1 == 0 {
                    return false;
                }
                entry.1 -= 1;
                true
            })
        },
        run: |kernel: &dyn Executable, arena: &mut [i32]| {
            let (arena, state) = arena.split_at_mut(workspace_words);
            Ok(data.0.run(kernel, arena, state)?.len())
        },
    };
    workload.compare(
        c,
        || query::ir(&plan.prepare()),
        builders!(B => query::build::<B>(query::BuildArgs {
            program: &plan.prepare(),
        })),
    );
}

fn simd_matmul(c: &mut Criterion) {
    use matmul::matmul as multiply;
    const M: usize = 96;
    const K: usize = 128;
    const N: usize = 128;
    let a: Vec<_> = (0..M * K).map(|i| (i % 37) as i32 - 18).collect();
    let b: Vec<_> = (0..K * N).map(|i| (i % 53) as i32 - 26).collect();
    let expected = matmul::reference(M, K, N, &a, &b);
    let workload = Workload {
        name: "simd_matmul",
        elements: M * K * N, // One element of throughput is one multiply-accumulate.
        workspace_words: M * N,
        expected: &|| &expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, dst: &mut [i32]| {
            let written = kernel.call(
                &[M as u64, K as u64, N as u64],
                &mut [
                    BufferArg::read(&a),
                    BufferArg::read(&b),
                    BufferArg::write(dst),
                ],
            )?;
            usize::try_from(i32::from_word(written)).map_err(|_| "invalid matrix dimensions".into())
        },
    };
    workload.compare(
        c,
        multiply::ir,
        builders!(B => multiply::build::<B>(multiply::BuildArgs {})),
    );
}

fn simd_hash_lookup(c: &mut Criterion) {
    use probe::lookup;
    let (table, queries) = probe::fixture(16_384, 12_288, 32_768);
    let expected = probe::reference(&table, &queries);
    let workload = Workload {
        name: "simd_hash_lookup",
        elements: queries.len(),
        workspace_words: queries.len(),
        expected: &|| &expected,
        equal: <[i32]>::eq,
        run: |kernel: &dyn Executable, dst: &mut [i32]| {
            let written = kernel.call(
                &[],
                &mut [
                    BufferArg::read(&table),
                    BufferArg::read(&queries),
                    BufferArg::write(dst),
                ],
            )?;
            usize::try_from(i32::from_word(written)).map_err(|_| "invalid hash table".into())
        },
    };
    workload.compare(
        c,
        lookup::ir,
        builders!(B => lookup::build::<B>(lookup::BuildArgs {})),
    );
}

criterion_group! {
    name = backends;
    config = Criterion::default().sample_size(30);
    targets = projection, sql_plan, simd_matmul, simd_hash_lookup, large::compare, gaps::compare
}
criterion_main!(backends);
