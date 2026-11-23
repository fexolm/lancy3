//! The build target is a value: one process builds a kernel for every
//! profile this machine runs, with every backend, through both build routes.
use splice::simd::Lane;
use splice::target::Profile;
use splice::{BuildOptions, Codegen, Executable, Mode, Target, kernel};

#[kernel]
fn widened_sum(src: &[i32], out: &mut [i32]) -> i32 {
    let low = Lane::<i32>::load(src, 0);
    let high = Lane::<i32>::load(src, Lane::<i32>::lanes());
    (low + high).store(out, 0);
    Lane::<i32>::lanes() as i32
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
        let error = widened_sum::build_with_options::<splice::Cranelift>(
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
