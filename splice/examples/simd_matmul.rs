#[path = "workloads/matmul.rs"]
mod matrix;
use matrix::{matmul, reference};
use splice::cranelift::Cranelift;
use splice::lancy::Lancy;
#[cfg(feature = "llvm")]
use splice::llvm::Llvm;
use splice::{BufferArg, Executable};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kernel: Box<dyn Executable> = match std::env::args().nth(1).as_deref() {
        None | Some("lancy") => {
            Box::new(matmul::build::<Lancy>(matmul::BuildArgs {})?.into_inner())
        }
        Some("cranelift") => {
            Box::new(matmul::build::<Cranelift>(matmul::BuildArgs {})?.into_inner())
        }
        #[cfg(feature = "llvm")]
        Some("llvm") => Box::new(matmul::build::<Llvm>(matmul::BuildArgs {})?.into_inner()),
        Some(other) => {
            return Err(format!("unknown backend {other}, choose lancy, cranelift or llvm").into());
        }
    };
    let a = [1, 2, 3, 4, 5, 6];
    let b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let mut c = [0; 8];
    let count = kernel.call(
        &[2, 3, 4],
        &mut [
            BufferArg::read(&a),
            BufferArg::read(&b),
            BufferArg::write(&mut c),
        ],
    )?;
    assert_eq!(count, 8);
    assert_eq!(c.as_slice(), reference(2, 3, 4, &a, &b));
    println!("A (2 × 3) × B (3 × 4)");
    for row in c.chunks_exact(4) {
        println!("{row:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use splice::{Codegen, KernelScalar};

    fn kernels() -> Vec<(&'static str, Box<dyn Executable>)> {
        let ir = matmul::ir().unwrap();
        vec![
            (
                "Lancy/direct",
                Box::new(
                    matmul::build::<Lancy>(matmul::BuildArgs {})
                        .unwrap()
                        .into_inner(),
                ),
            ),
            (
                "Lancy/ir",
                Box::new(Lancy::compile(&ir).unwrap().into_inner()),
            ),
            (
                "Cranelift/direct",
                Box::new(
                    matmul::build::<Cranelift>(matmul::BuildArgs {})
                        .unwrap()
                        .into_inner(),
                ),
            ),
            (
                "Cranelift/ir",
                Box::new(Cranelift::compile(&ir).unwrap().into_inner()),
            ),
            #[cfg(feature = "llvm")]
            (
                "LLVM/direct",
                Box::new(
                    matmul::build::<Llvm>(matmul::BuildArgs {})
                        .unwrap()
                        .into_inner(),
                ),
            ),
            #[cfg(feature = "llvm")]
            (
                "LLVM/ir",
                Box::new(Llvm::compile(&ir).unwrap().into_inner()),
            ),
        ]
    }

    #[test]
    fn tiles_tails_and_invalid_shapes_match_scalar_dot_products() {
        const SENTINEL: i32 = 0x0bad_cafe;
        let kernels = kernels();
        let width = splice::simd::lanes::<i32>();
        let cases = [
            (2, 3, 4),
            (16, 32, 4 * width),
            (17, 35, 4 * width + 3),
            (1, 1, width - 1),
            (0, 3, 4),
            (2, 0, width + 1),
            (4, 3, 0),
        ];
        for (rows, inner, cols) in cases {
            let a: Vec<_> = (0..rows * inner)
                .map(|i| (i as i32).wrapping_mul(1_664_525).wrapping_add(i32::MAX))
                .collect();
            let b: Vec<_> = (0..inner * cols)
                .map(|i| {
                    (i as i32)
                        .wrapping_mul(1_013_904_223)
                        .wrapping_add(i32::MIN)
                })
                .collect();
            let mut expected = reference(rows, inner, cols, &a, &b);
            let count = expected.len();
            expected.extend([SENTINEL; 3]);
            let dims = [rows as i32, inner as i32, cols as i32];
            let mut dst = vec![SENTINEL; count + 3];
            for (name, kernel) in &kernels {
                dst.fill(SENTINEL);
                let actual = kernel
                    .call(
                        &dims.map(i32::to_word),
                        &mut [
                            BufferArg::read(&a),
                            BufferArg::read(&b),
                            BufferArg::write(&mut dst),
                        ],
                    )
                    .unwrap();
                assert_eq!(actual, count as u64, "{name} {dims:?}");
                assert_eq!(dst, expected, "{name} {dims:?}");
            }
        }
        for dims in [
            [-1, 1, 1],
            [2, 1, 1],
            [1, 1, 2],
            [1, 0, 4],
            [i32::MAX, 0, 2],
        ] {
            // One input element and three output elements make these shapes invalid.
            let mut out = [SENTINEL; 3];
            for (name, kernel) in &kernels {
                let result = kernel
                    .call(
                        &dims.map(i32::to_word),
                        &mut [
                            BufferArg::read::<i32>(&[3]),
                            BufferArg::read::<i32>(&[4]),
                            BufferArg::write(&mut out),
                        ],
                    )
                    .unwrap();
                assert_eq!(result, (-1i32).to_word(), "{name} {dims:?}");
                assert_eq!(out, [SENTINEL; 3], "{name} {dims:?}");
            }
        }
    }
}
