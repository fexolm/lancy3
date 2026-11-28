//! One place that lists the Splice backends, so every test covers all of
//! them and a new backend is added once.
//!
//! Lancy and Cranelift are always covered. LLVM needs a system libLLVM, so it
//! is covered when the crate that expands [`each_backend!`] enables its own
//! `llvm` feature, which must enable `splice-kernel/llvm`; `splice-kernel`'s
//! feature is exactly that, and this crate's default one forwards to it.

/// Run `body` once per backend with `$b` naming the backend type:
/// `each_backend!(B => check::<B>(&ir))`.
#[macro_export]
macro_rules! each_backend {
    ($b:ident => $body:expr) => {{
        {
            type $b = ::splice::lancy::Lancy;
            $body
        };
        {
            type $b = ::splice::cranelift::Cranelift;
            $body
        };
        #[cfg(feature = "llvm")]
        {
            type $b = ::splice::llvm::Llvm;
            $body
        };
        #[cfg(feature = "llvm")]
        {
            type $b = ::splice::llvm::LlvmManualSimd;
            $body
        };
    }};
}

use splice::frontend::{self, KernelBuilder};
use splice::hir::{Instr, Simd};
use splice::{BufferArg, BuildOptions, Codegen, Executable, Function, KernelScalar, Mode};
use std::rc::Rc;

/// Every instruction of `function` and of the functions it calls, each
/// shared body once: tests assert what staging produced on these.
pub fn instructions(function: &Function) -> impl Iterator<Item = &Instr> {
    function
        .reachable_functions()
        .into_iter()
        .flat_map(Function::instructions)
}

/// The SIMD operations of `function` and its callees, in order.
pub fn simd_ops(function: &Function) -> Vec<Simd> {
    instructions(function)
        .filter_map(|instruction| match instruction {
            Instr::CallIntrinsic {
                intrinsic: splice::hir::Intrinsic::Simd(inst),
                ..
            } => Some(*inst),
            _ => None,
        })
        .collect()
}

/// Call a kernel whose scalar parameters and result are all `i32`, as most
/// tests' kernels are.
pub fn call_i32(
    kernel: &dyn Executable,
    args: &[i32],
    buffers: &mut [BufferArg<'_>],
) -> Result<i32, splice::CallError> {
    let words: Vec<u64> = args.iter().map(|&arg| arg.to_word()).collect();
    kernel.call(&words, buffers).map(i32::from_word)
}

/// A kernel body that can be staged repeatedly.
pub type Stage = Rc<dyn Fn(&mut dyn KernelBuilder) -> splice::Result<()>>;

/// Calls one compiled kernel.
pub type Invoke<'a> = &'a dyn Fn(&[u64], &mut [BufferArg<'_>]) -> Result<u64, splice::CallError>;

/// Compile `stage` with every backend, staged (`build`) and from portable IR
/// (`compile`), in Debug and Release. `observe` runs each kernel and reports
/// what it saw, including buffers after an error; every build must report the
/// same, which is returned for comparison with a Rust reference.
pub fn agree(
    name: &str,
    signature: frontend::Signature<'_>,
    stage: &Stage,
    mut observe: impl FnMut(Invoke<'_>) -> Vec<String>,
) -> Vec<String> {
    fn run<B: Codegen<Output: Executable>>(
        staged: bool,
        options: BuildOptions,
        signature: frontend::Signature<'_>,
        stage: &Stage,
        observe: &mut dyn FnMut(Invoke<'_>) -> Vec<String>,
    ) -> (String, Vec<String>) {
        let body = {
            let stage = Rc::clone(stage);
            Box::new(move |b: &mut dyn KernelBuilder| stage(b))
        };
        let built = if staged {
            B::build_with_options(signature, body, options)
        } else {
            frontend::materialize(signature, body, options)
                .and_then(|f| B::compile_with_options(&f, options))
        };
        let route = if staged { "build" } else { "compile" };
        let label = format!("{}/{route}/{:?}", std::any::type_name::<B>(), options.mode);
        let seen = match built {
            Ok(kernel) => observe(&|args, buffers| kernel.call(args, buffers)),
            Err(error) => vec![format!("build error: {error}")],
        };
        (label, seen)
    }
    let mut runs = Vec::new();
    for mode in [Mode::Debug, Mode::Release] {
        let options = BuildOptions::host(mode).expect("SPLICE_SIMD names a target of this machine");
        for staged in [true, false] {
            each_backend!(B => runs.push(run::<B>(staged, options, signature, stage, &mut observe)));
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
