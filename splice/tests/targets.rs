//! The build target is a value: one process builds a kernel for every
//! profile this machine runs, with every backend, through both build routes.
use splice::std::intrinsics::simd::{Lane, Lane as Packet};
use splice::target::Profile;
use splice::{BuildOptions, Codegen, Executable, Mode, Target, kernel};

#[kernel]
fn widened_sum(src: &[i32], out: &mut [i32]) -> i32 {
    let low = Packet::<i32>::load(src, 0);
    let high = Lane::<i32>::load(src, Packet::<i32>::element_width());
    (low + high).store(out, 0);
    (splice::std::intrinsics::simd::Lane::<i32>::byte_width() / 4 + Packet::<i32>::element_width())
        as i32
        / 2
}

/// Every distinct target that this machine can execute.
fn runnable() -> Vec<Target> {
    let mut targets: Vec<Target> = Vec::new();
    for profile in Profile::ALL {
        if let Ok(target) = Target::from_profile(profile)
            && target.check_execution().is_ok()
            && !targets.contains(&target)
        {
            targets.push(target);
        }
    }
    targets
}

struct Recorder(usize);

impl Recorder {
    fn record(&mut self, width: usize) -> usize {
        self.0 += width;
        self.0
    }
}

#[kernel]
fn record_widths(recorder: &mut Recorder) -> usize {
    use splice::std::intrinsics::Lane as block;
    recorder.record(block::<f64>::element_width() + block::<u16>::byte_width())
}

#[allow(non_camel_case_types)]
type packet = Lane<i32>;

mod metrics {
    pub fn byte_width() -> usize {
        7
    }
}

struct Report;
impl Report {
    fn byte_width() -> &'static str {
        "abc"
    }
}

#[kernel]
fn mixed_widths(recorder: &mut Recorder) -> usize {
    recorder.record(Report::byte_width().len() + Lane::<i32>::byte_width())
}

#[kernel]
fn qualified_alias(recorder: &mut Recorder) -> usize {
    recorder.record(<packet>::byte_width() + metrics::byte_width())
}

#[kernel]
fn nested_bit_count(recorder: &mut Recorder, value: u32) -> usize {
    recorder.record(value.count_ones() as usize)
}

#[test]
fn opaque_rust_expressions_cannot_execute_integer_intrinsic_bodies() {
    let error = nested_bit_count::ir().unwrap_err().to_string();
    assert!(
        error.contains("nested intrinsic in an opaque Rust object expression is unsupported"),
        "{error}"
    );
}

#[test]
fn rust_receives_widths_computed_for_the_compilation_target() {
    use splice::hir::{Instr, SimdOp};
    for target in runnable() {
        for mode in [Mode::Debug, Mode::Release] {
            let options = BuildOptions::new(mode, target);
            let ir = record_widths::ir_with_options(options).unwrap();
            let alias = qualified_alias::ir_with_options(options).unwrap();
            let mixed = mixed_widths::ir_with_options(options).unwrap();
            let mut meter = Recorder(0);
            assert_eq!(
                splice::lancy::Lancy::compile(&mixed)
                    .unwrap()
                    .call(mixed_widths::CallArgs {
                        recorder: &mut meter
                    }),
                Ok(3 + target.vector_bytes())
            );
            assert_eq!(splice_testkit::simd_ops(&alias)[0].op, SimdOp::ByteWidth);
            let mut recorder = Recorder(0);
            assert_eq!(
                splice::lancy::Lancy::compile(&alias)
                    .unwrap()
                    .call(qualified_alias::CallArgs {
                        recorder: &mut recorder
                    }),
                Ok(target.vector_bytes() + 7)
            );
            assert_eq!(
                splice_testkit::simd_ops(&ir)
                    .iter()
                    .map(|op| op.op)
                    .collect::<Vec<_>>(),
                [SimdOp::ElementWidth, SimdOp::ByteWidth]
            );
            let packets: Vec<_> = splice_testkit::instructions(&ir)
                .filter_map(|i| match i {
                    Instr::RustCall { args, .. } => Some(args.len()),
                    _ => None,
                })
                .collect();
            assert_eq!(packets, [2]);
            splice_testkit::each_backend!(B => {
                for code in [record_widths::build_with_options::<B>(options, record_widths::BuildArgs {}).unwrap(), B::compile(&ir).unwrap()] {
                    let mut recorder = Recorder(3);
                    let expected = 3 + target.vector_bytes() / 8 + target.vector_bytes();
                    assert_eq!(code.call(record_widths::CallArgs { recorder: &mut recorder }), Ok(expected));
                    assert_eq!(recorder.0, expected);
                }
            });
        }
    }
}

fn check<B: Codegen<Output: Executable>>(target: Target) {
    let name = std::any::type_name::<B>();
    let lanes = target.vector_bytes() / 4;
    let src: Vec<i32> = (0..2 * lanes as i32).collect();
    let expected: Vec<i32> = (0..lanes as i32)
        .map(|lane| 2 * lane + lanes as i32)
        .collect();
    for mode in [Mode::Debug, Mode::Release] {
        let options = BuildOptions::new(mode, target);
        let ir = widened_sum::ir_with_options(options).unwrap();
        assert_eq!(ir.options(), options);
        let mut elements = 0;
        let mut bytes = 0;
        for instruction in splice_testkit::instructions(&ir) {
            use splice::hir::{Instr, Intrinsic, Simd, SimdOp};
            match instruction {
                Instr::CallIntrinsic {
                    intrinsic:
                        Intrinsic::Simd(Simd {
                            op: SimdOp::ElementWidth,
                            ..
                        }),
                    ..
                } => elements += 1,
                Instr::CallIntrinsic {
                    intrinsic:
                        Intrinsic::Simd(Simd {
                            op: SimdOp::ByteWidth,
                            ..
                        }),
                    ..
                } => bytes += 1,
                Instr::RustCall { .. } => {
                    panic!("unexpected Rust call in SIMD kernel: {instruction:?}")
                }
                _ => {}
            }
        }
        assert_eq!((elements, bytes), (2, 1));
        let kernels = [
            widened_sum::build_with_options::<B>(options, widened_sum::BuildArgs {}).unwrap(),
            B::compile(&ir).unwrap(),
        ];
        for kernel in kernels {
            let mut out = vec![0; lanes];
            let result = kernel.call(widened_sum::CallArgs {
                src: &src,
                out: &mut out,
            });
            assert_eq!(result, Ok(lanes as i32), "{name} {mode:?} {target:?}");
            assert_eq!(out, expected, "{name} {mode:?} {target:?}");
        }
    }
}

#[test]
fn one_process_builds_every_runnable_profile_with_every_backend() {
    let targets = runnable();
    // SSE2 runs on every x86-64 machine next to the host's own profile.
    #[cfg(target_arch = "x86_64")]
    assert!(targets.contains(&Target::from_profile(Profile::Sse).unwrap()));
    for target in targets {
        splice_testkit::each_backend!(B => check::<B>(target));
    }
}

#[test]
fn portable_ir_compiles_for_another_target_than_it_was_staged_for() {
    let targets = runnable();
    let staged = BuildOptions::new(Mode::Release, targets[0]);
    let ir = widened_sum::ir_with_options(staged).unwrap();
    for &target in &targets {
        let lanes = target.vector_bytes() / 4;
        let src = vec![1; 2 * lanes];
        splice_testkit::each_backend!(B => {
            let kernel =
                B::compile_with_options(&ir, BuildOptions::new(Mode::Release, target)).unwrap();
            let mut out = vec![0; lanes];
            let result = kernel.call(widened_sum::CallArgs {
                src: &src,
                out: &mut out,
            });
            assert_eq!(result, Ok(lanes as i32), "{target:?}");
            assert_eq!(out, vec![2; lanes], "{target:?}");
        });
    }
}

#[test]
fn backends_without_the_convention_reject_it() {
    #[cfg(target_arch = "x86_64")]
    {
        use splice::target::CallingConvention;
        let foreign = if cfg!(windows) {
            CallingConvention::SysV
        } else {
            CallingConvention::Win64
        };
        let target = Target::from_profile(Profile::Sse)
            .unwrap()
            .with_calling_convention(foreign)
            .unwrap();
        let options = BuildOptions::new(Mode::Release, target);
        let error = widened_sum::build_with_options::<splice::cranelift::Cranelift>(
            options,
            widened_sum::BuildArgs {},
        )
        .err()
        .unwrap();
        assert!(
            matches!(error, splice::BuildError::Target(splice::TargetError::HostCalls(c)) if c == foreign),
            "{error:?}"
        );
    }
}

#[kernel]
fn reversed(src: &[i32], indices: &[i32], out: &mut [i32]) -> i32 {
    let values = Lane::<i32>::load(src, 0);
    values.swizzle(Lane::load(indices, 0)).store(out, 0);
    0
}

/// Byte shuffles without SSSE3 become a Cranelift libcall that the JIT must
/// resolve; SSE2 targets exercise it on every x86-64 machine.
#[test]
fn swizzle_runs_on_every_runnable_profile() {
    for target in runnable() {
        let lanes = target.vector_bytes() / 4;
        let src: Vec<i32> = (0..lanes as i32).map(|i| 10 * i).collect();
        let indices: Vec<i32> = (0..lanes as i32).rev().collect();
        let expected: Vec<i32> = src.iter().rev().copied().collect();
        splice_testkit::each_backend!(B => {
            let options = BuildOptions::new(Mode::Release, target);
            let kernel = reversed::build_with_options::<B>(options, reversed::BuildArgs {}).unwrap();
            let mut out = vec![0; lanes];
            let result = kernel.call(reversed::CallArgs {
                src: &src,
                indices: &indices,
                out: &mut out,
            });
            assert_eq!(result, Ok(0), "{} {target:?}", std::any::type_name::<B>());
            assert_eq!(out, expected, "{} {target:?}", std::any::type_name::<B>());
        });
    }
}
