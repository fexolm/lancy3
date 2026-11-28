//! BE-01: without AVX, Cranelift 0.132 folds an unaligned vector load into a
//! legacy-SSE `cmpps`/`cmppd` memory operand, which faults unless the address
//! is 16-byte aligned.
use splice_core::frontend::{self, KernelBuilder, Signature};
use splice_core::hir::{Constant, ScalarType};
use splice_core::hir::{Simd, SimdOp};
use splice_core::target::Profile;
use splice_core::{BufferArg, BufferKind, BuildOptions, Codegen, Executable, Mode, Target};
use splice_cranelift::Cranelift;

/// `count(buf[0..] == buf[1..])`: the second load starts one element past the
/// slice's base, so it is misaligned for any 16-byte-aligned allocation.
fn shifted_compare(
    element: ScalarType,
    bytes: usize,
) -> impl Fn(&mut dyn KernelBuilder) -> splice_core::Result<()> {
    move |b| {
        let zero = b.constant(Constant::from(0usize))?;
        let one = b.constant(Constant::from(1usize))?;
        let load = SimdOp::Load(b.buffer(0)?);
        let x = b.simd(Simd { element, op: load }, &[zero])?.unwrap();
        let y = b.simd(Simd { element, op: load }, &[one])?.unwrap();
        let mask = b
            .simd(
                Simd {
                    element,
                    op: SimdOp::Eq,
                },
                &[x, y],
            )?
            .unwrap();
        // Count takes 32-bit lanes; wider masks report whether any lane matched.
        let result = if element.lanes(bytes) == ScalarType::I32.lanes(bytes) {
            let count = Simd {
                element: ScalarType::I32,
                op: SimdOp::Count,
            };
            b.simd(count, &[mask])?
        } else {
            let any = b.simd(
                Simd {
                    element,
                    op: SimdOp::Any,
                },
                &[mask],
            )?;
            Some(b.cast(any.unwrap(), ScalarType::I32)?)
        };
        b.ret(result.unwrap())?;
        Ok(())
    }
}

fn check_shifted_compares(options: BuildOptions) {
    let bytes = options.target.vector_bytes();
    for element in [ScalarType::F32, ScalarType::F64] {
        let kinds = [BufferKind::new(element, false)];
        let lanes = element.lanes(bytes);
        // Pairs of equal values: lane i equals lane i + 1 exactly when i is even.
        let expected = if element == ScalarType::F32 {
            lanes.div_ceil(2) as u64
        } else {
            1 // `Any` is true.
        };
        let stage = || Box::new(shifted_compare(element, bytes));
        let ir = frontend::materialize(
            Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &kinds),
            stage(),
            options,
        )
        .unwrap();
        for code in [
            Cranelift::build_with_options(
                Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &kinds),
                stage(),
                options,
            )
            .unwrap(),
            Cranelift::compile(&ir).unwrap(),
        ] {
            let result = match element {
                ScalarType::F32 => {
                    let data: Vec<f32> = (0..=lanes).map(|i| (i / 2) as f32).collect();
                    code.call(&[], &mut [BufferArg::from(&data[..])])
                }
                _ => {
                    let data: Vec<f64> = (0..=lanes).map(|i| (i / 2) as f64).collect();
                    code.call(&[], &mut [BufferArg::from(&data[..])])
                }
            };
            assert_eq!(result, Ok(expected), "{element:?} on {:?}", options.target);
        }
    }
}

/// Every profile this machine runs, including SSE without AVX, where the
/// faulting encoding is chosen, in one process.
#[test]
fn unaligned_float_compare_operands() {
    let mut targets: Vec<Target> = Profile::ALL
        .iter()
        .filter_map(|&profile| Target::from_profile(profile).ok())
        .filter(|target| target.check_execution().is_ok())
        .collect();
    targets.dedup();
    #[cfg(target_arch = "x86_64")]
    assert!(targets.contains(&Target::from_profile(Profile::Sse).unwrap()));
    for target in targets {
        check_shifted_compares(BuildOptions::new(Mode::Release, target));
    }
}
