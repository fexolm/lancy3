//! JIT compilation and warm execution on the same inputs for all three backends.
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
use splice::{BufferArg, Codegen, Executable, Function, kernel};
use splice_cranelift::{Cranelift, CraneliftKernel};
use splice_lancy::{Lancy, LancyKernel};
use splice_llvm::{Llvm, LlvmKernel};
use std::{cell::LazyCell, hint::black_box};

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

struct Builders<'a> {
    lancy: &'a dyn Fn() -> splice::Result<LancyKernel>,
    llvm: &'a dyn Fn() -> splice::Result<LlvmKernel>,
    cranelift: &'a dyn Fn() -> splice::Result<CraneliftKernel>,
}

struct Workload<'a, Run> {
    name: &'static str,
    elements: usize,
    workspace_words: usize,
    expected: &'a dyn Fn() -> &'a [i32],
    run: Run,
}

fn bench_compile<K>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    backend: &str,
    build: impl Fn() -> splice::Result<K>,
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

impl<Run> Workload<'_, Run>
where
    Run: Fn(&dyn Executable, &mut [i32]) -> splice::Result<usize>,
{
    fn verify(&self, kernel: &dyn Executable, workspace: &mut [i32], backend: &str) {
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
            workspace[..used] == *expected,
            "{} / {backend}: incorrect output",
            self.name
        );
    }

    fn bench_execute<K: Executable>(
        &self,
        group: &mut BenchmarkGroup<'_, WallTime>,
        backend: &str,
        build: impl Fn() -> splice::Result<K>,
    ) {
        let mut state = None;
        group.bench_function(BenchmarkId::new(backend, self.elements), |b| {
            let (kernel, workspace) = state.get_or_insert_with(|| {
                let kernel = build().expect("compile execution benchmark");
                let mut workspace = vec![-1; self.workspace_words];
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

    fn bench_route(&self, c: &mut Criterion, route: &str, builds: Builders<'_>) {
        let mut compile = c.benchmark_group(format!("{}/compile/{route}", self.name));
        bench_compile(&mut compile, "lancy", builds.lancy);
        bench_compile(&mut compile, "cranelift", builds.cranelift);
        bench_compile(&mut compile, "llvm", builds.llvm);
        compile.finish();

        let mut execute = c.benchmark_group(format!("{}/execute/{route}", self.name));
        execute.throughput(Throughput::Elements(self.elements as u64));
        self.bench_execute(&mut execute, "lancy", builds.lancy);
        self.bench_execute(&mut execute, "cranelift", builds.cranelift);
        self.bench_execute(&mut execute, "llvm", builds.llvm);
        execute.finish();
    }

    fn compare(
        &self,
        c: &mut Criterion,
        stage: impl Fn() -> splice::Result<Function>,
        builds: Builders<'_>,
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
        self.bench_route(
            c,
            "from_ir",
            Builders {
                lancy: &|| Lancy::compile(black_box(&ir)),
                cranelift: &|| Cranelift::compile(black_box(&ir)),
                llvm: &|| Llvm::compile(black_box(&ir)),
            },
        );
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
        run: |kernel: &dyn Executable, dst: &mut [i32]| {
            let count = kernel.call(
                &[],
                &mut [
                    BufferArg::Read(&prices),
                    BufferArg::Read(&quantities),
                    BufferArg::Write(dst),
                ],
            )?;
            usize::try_from(count).map_err(|_| "negative projection length".into())
        },
    };
    workload.compare(
        c,
        || line_totals::ir(black_box(2)),
        Builders {
            lancy: &|| line_totals::build::<Lancy>(black_box(2)),
            cranelift: &|| line_totals::build::<Cranelift>(black_box(2)),
            llvm: &|| line_totals::build::<Llvm>(black_box(2)),
        },
    );
}

fn sql_plan(c: &mut Criterion) {
    const SALES: i32 = 28 * 1024 * 1024;
    const PRODUCTS: i32 = SALES / 16;
    // Each product matches at most once in every dimension. Allow two sales
    // buffers, a dimension and its hash table (bucket heads and chain links).
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
        let mut expected = vec![0; workspace_words];
        let count = query(&plan, catalog.offsets(), catalog.data(), &mut expected);
        let rows = usize::try_from(count).expect("reference query ran out of workspace");
        assert_eq!(rows, SALES as usize * 3 / 8);
        expected.truncate(rows * 2);
        (catalog, expected.into_boxed_slice())
    });
    let workload = Workload {
        name: "sql_plan",
        elements: SALES as usize,
        workspace_words,
        expected: &|| &data.1,
        run: |kernel: &dyn Executable, arena: &mut [i32]| {
            data.0.run(kernel, arena).map(<[i32]>::len)
        },
    };
    workload.compare(
        c,
        || query::ir(black_box(&plan)),
        Builders {
            lancy: &|| query::build::<Lancy>(black_box(&plan)),
            cranelift: &|| query::build::<Cranelift>(black_box(&plan)),
            llvm: &|| query::build::<Llvm>(black_box(&plan)),
        },
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
        run: |kernel: &dyn Executable, dst: &mut [i32]| {
            let written = kernel.call(
                &[M as i32, K as i32, N as i32],
                &mut [
                    BufferArg::Read(&a),
                    BufferArg::Read(&b),
                    BufferArg::Write(dst),
                ],
            )?;
            usize::try_from(written).map_err(|_| "invalid matrix dimensions".into())
        },
    };
    workload.compare(
        c,
        multiply::ir,
        Builders {
            lancy: &multiply::build::<Lancy>,
            cranelift: &multiply::build::<Cranelift>,
            llvm: &multiply::build::<Llvm>,
        },
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
        run: |kernel: &dyn Executable, dst: &mut [i32]| {
            let written = kernel.call(
                &[],
                &mut [
                    BufferArg::Read(&table),
                    BufferArg::Read(&queries),
                    BufferArg::Write(dst),
                ],
            )?;
            usize::try_from(written).map_err(|_| "invalid hash table".into())
        },
    };
    workload.compare(
        c,
        lookup::ir,
        Builders {
            lancy: &lookup::build::<Lancy>,
            cranelift: &lookup::build::<Cranelift>,
            llvm: &lookup::build::<Llvm>,
        },
    );
}

criterion_group! {
    name = backends;
    config = Criterion::default().sample_size(30);
    targets = projection, sql_plan, simd_matmul, simd_hash_lookup
}
criterion_main!(backends);
