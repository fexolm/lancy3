use splice::{BufferArg, Codegen, Executable, kernel};

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

fn run<B: Codegen<Output: Executable>>(repetitions: usize) -> splice::Result<()> {
    let kernel = accumulate::build::<B>()?;
    let input = [4, 8];
    let mut output = [0, 0];
    for _ in 0..repetitions {
        let sum = kernel.call(
            &[1],
            &mut [BufferArg::Read(&input), BufferArg::Write(&mut output)],
        )?;
        assert_eq!((sum, output), (40, [16, 24]));
    }
    Ok(())
}

fn main() -> splice::Result<()> {
    let mut args = std::env::args().skip(1);
    let backend = args.next();
    let repetitions = args
        .next()
        .map_or(Ok(1), |value| value.parse::<usize>())
        .map_err(|error| error.to_string())?;
    match backend.as_deref() {
        Some("llvm") => run::<splice_llvm::Llvm>(repetitions),
        Some("llvm_manual_simd") => run::<splice_llvm::LlvmManualSimd>(repetitions),
        Some("cranelift") => run::<splice_cranelift::Cranelift>(repetitions),
        Some("lancy") | None => run::<splice_lancy::Lancy>(repetitions),
        Some(other) => Err(format!("unknown backend {other}")),
    }
}
