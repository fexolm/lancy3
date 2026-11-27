use splice::{Codegen, Executable, kernel};

#[kernel]
fn child(base: i32) -> i32 {
    let adjusted = base + 3;
    adjusted * 2
}

#[kernel]
fn accumulate(input: &[i32], output: &mut [i32], bias: i32) -> i32 {
    let mut sum = 0;
    for index in 0..input.len() {
        let value = input[index] + bias;
        let adjusted = child(value);
        output[index] = adjusted;
        sum += adjusted;
    }
    sum
}

fn run<B: Codegen<Output: Executable>>(
    repetitions: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let kernel = accumulate::build::<B>(accumulate::BuildArgs {})?;
    let input = [4, 8];
    let mut output = [0, 0];
    for _ in 0..repetitions {
        let sum = kernel.call(accumulate::CallArgs {
            input: &input,
            output: &mut output,
            bias: 1,
        })?;
        assert_eq!((sum, output), (40, [16, 24]));
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let backend = args.next();
    let repetitions = args
        .next()
        .map_or(Ok(1), |value| value.parse::<usize>())
        .map_err(|error| error.to_string())?;
    match backend.as_deref() {
        #[cfg(feature = "llvm")]
        Some("llvm") => run::<splice::llvm::Llvm>(repetitions),
        #[cfg(feature = "llvm")]
        Some("llvm_manual_simd") => run::<splice::llvm::LlvmManualSimd>(repetitions),
        Some("cranelift") => run::<splice::cranelift::Cranelift>(repetitions),
        Some("lancy") | None => run::<splice::lancy::Lancy>(repetitions),
        Some(other) => Err(format!("unknown backend {other}").into()),
    }
}
