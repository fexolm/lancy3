//! One place that lists the Splice backends, so every test covers all of
//! them and a new backend is added once.
pub use splice_cranelift::Cranelift;
pub use splice_lancy::Lancy;
pub use splice_llvm::{Llvm, LlvmManualSimd};

/// Run `body` once per backend with `$b` naming the backend type:
/// `each_backend!(B => check::<B>(&ir))`.
#[macro_export]
macro_rules! each_backend {
    ($b:ident => $body:expr) => {{
        {
            type $b = $crate::Lancy;
            $body
        };
        {
            type $b = $crate::Cranelift;
            $body
        };
        {
            type $b = $crate::Llvm;
            $body
        };
        {
            type $b = $crate::LlvmManualSimd;
            $body
        };
    }};
}

use splice_core::frontend::{self, KernelBuilder};
use splice_core::{BufferArg, BufferKind, BuildOptions, Codegen, Executable};
use std::rc::Rc;

/// A kernel body that can be staged repeatedly.
pub type Stage = Rc<dyn Fn(&mut dyn KernelBuilder) -> Result<(), String>>;

/// Calls one compiled kernel.
pub type Invoke<'a> = &'a dyn Fn(&[i32], &mut [BufferArg<'_>]) -> splice_core::Result<i32>;

/// Compile `stage` with every backend, staged (`build`) and from portable IR
/// (`compile`), in Debug and Release. `observe` runs each kernel and reports
/// what it saw, including buffers after an error; every build must report the
/// same, which is returned for comparison with a Rust reference.
pub fn agree(
    name: &str,
    arity: usize,
    kinds: &[BufferKind],
    stage: &Stage,
    mut observe: impl FnMut(Invoke<'_>) -> Vec<String>,
) -> Vec<String> {
    fn run<B: Codegen<Output: Executable>>(
        staged: bool,
        options: BuildOptions,
        arity: usize,
        kinds: &[BufferKind],
        stage: &Stage,
        observe: &mut dyn FnMut(Invoke<'_>) -> Vec<String>,
    ) -> (String, Vec<String>) {
        let body = {
            let stage = Rc::clone(stage);
            Box::new(move |b: &mut dyn KernelBuilder| stage(b))
        };
        let built = if staged {
            B::build_with_options(arity, kinds, body, options)
        } else {
            frontend::materialize_with_options(arity, kinds, body, options)
                .and_then(|f| B::compile_with_options(&f, options))
        };
        let route = if staged { "build" } else { "compile" };
        let label = format!("{}/{route}/{options:?}", std::any::type_name::<B>());
        let seen = match built {
            Ok(kernel) => observe(&|args, buffers| kernel.call(args, buffers)),
            Err(error) => vec![format!("build error: {error}")],
        };
        (label, seen)
    }
    let mut runs = Vec::new();
    for options in [BuildOptions::Debug, BuildOptions::Release] {
        for staged in [true, false] {
            each_backend!(B => runs.push(run::<B>(staged, options, arity, kinds, stage, &mut observe)));
        }
    }
    let (first, expected) = &runs[0];
    for (label, seen) in &runs[1..] {
        let diff: Vec<_> = expected
            .iter()
            .zip(seen)
            .filter(|(a, b)| a != b)
            .map(|(a, b)| format!("  {first}: {a}\n  {label}: {b}"))
            .collect();
        assert!(
            diff.is_empty() && expected.len() == seen.len(),
            "{name}: {label} differs from {first}\n{}",
            diff.join("\n")
        );
    }
    runs.swap_remove(0).1
}
