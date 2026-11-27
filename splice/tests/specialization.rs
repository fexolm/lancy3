use splice::frontend::Signature;
use splice::hir::{Constant, Instr, ScalarType};
use splice::{Codegen, ComptimeKey, Executable, Function, kernel};

/// Distinct bodies of the kernel function `name` in the call graph of
/// `root`, and calls to them.
fn bodies_and_calls(root: &Function, name: &str) -> (usize, usize) {
    let named = |f: &Function| f.source().is_some_and(|source| source.name == name);
    let bodies = root.reachable_functions();
    let calls = splice_testkit::instructions(root)
        .filter(
            |instruction| matches!(instruction, Instr::Call { function, .. } if named(function)),
        )
        .count();
    (bodies.into_iter().filter(|f| named(f)).count(), calls)
}

#[derive(ComptimeKey)]
enum Mode<'a> {
    Add(&'a [i32]),
    Sub(&'a [i32]),
}

#[kernel]
fn increment(x: i32) -> i32 {
    x + 1
}

#[kernel]
fn apply(#[comptime] mode: &Mode<'_>, #[comptime] scale: i32, x: i32) -> i32 {
    let (values, sign) = match mode {
        Mode::Add(values) => (*values, 1),
        Mode::Sub(values) => (*values, -1),
    };
    let mut bias = 0;
    for &value in values {
        bias += value * sign;
    }
    increment(x) + bias * scale
}

#[kernel]
fn chain(#[comptime] modes: &[Mode<'_>], mut x: i32) -> i32 {
    for mode in modes {
        x = apply(mode, 1, x);
    }
    apply(&modes[0], 2, increment(x))
}

fn check_direct<B: Codegen>(modes: &[Mode<'_>], expected: i32)
where
    B::Output: Executable,
{
    let code = chain::build::<B>(chain::BuildArgs { modes }).unwrap();
    assert_eq!(code.call(chain::CallArgs { x: 12 }), Ok(expected));
}

#[test]
fn repeated_values_skip_staging_across_parents_and_keep_all_calls() {
    let a = vec![2, 3];
    let b = a.clone();
    // Different allocations with the same contents share a callee. Add([5])
    // emits the same bias but is a different set of specialization arguments.
    let modes = [Mode::Add(&a), Mode::Add(&b), Mode::Sub(&a), Mode::Add(&[5])];
    let expected = chain(&modes, 12);
    splice_testkit::each_backend!(B => check_direct::<B>(&modes, expected));

    let ir = chain::ir(&modes).unwrap();
    assert_eq!(bodies_and_calls(&ir, "apply"), (4, 5));
    assert_eq!(bodies_and_calls(&ir, "increment"), (1, 5));
    let args = || chain::CallArgs { x: 12 };
    splice_testkit::each_backend!(B => {
        assert_eq!(B::compile(&ir).unwrap().call(args()), Ok(expected));
    });
}

trait Flavor: ComptimeKey {
    const VALUE: i32;
}
#[derive(ComptimeKey)]
struct First;
#[derive(ComptimeKey)]
struct Second;
impl Flavor for First {
    const VALUE: i32 = 2;
}
impl Flavor for Second {
    const VALUE: i32 = 7;
}

#[kernel]
fn affine<T: Flavor, const N: i32>(x: i32) -> i32 {
    x * T::VALUE + N
}
#[kernel]
fn instances(x: i32) -> i32 {
    let a = affine::<First, 1>(x);
    let b = affine::<Second, 1>(a);
    let c = affine::<Second, 2>(b);
    affine::<First, 1>(c)
}
#[test]
fn type_and_const_arguments_distinguish_monomorphizations() {
    let ir = instances::ir().unwrap();
    assert_eq!(bodies_and_calls(&ir, "affine"), (3, 4));
    let expected = instances(3);
    splice_testkit::each_backend!(B => {
        let direct = instances::build::<B>(instances::BuildArgs {}).unwrap();
        for code in [direct, B::compile(&ir).unwrap()] {
            assert_eq!(code.call(instances::CallArgs { x: 3 }), Ok(expected));
        }
    });
}

fn retry(root: &mut dyn splice::frontend::KernelBuilder) -> splice::Result<()> {
    use splice::frontend::Specialization;
    struct Parent;
    struct Child;
    let failed = root.call_specialized(
        Specialization::new::<Parent>(Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[])),
        &[],
        &[],
        Box::new(|b| {
            b.call_specialized(
                Specialization::new::<Child>(Signature::new(
                    &[ScalarType::I32; 0],
                    ScalarType::I32,
                    &[],
                )),
                &[],
                &[],
                Box::new(|c| {
                    let value = c.constant(Constant::from(17))?;
                    c.ret(value)?;
                    Ok(())
                }),
            )?;
            Err(splice::BuildError::unsupported("rejected"))
        }),
    );
    assert!(
        matches!(failed, Err(splice::BuildError::Unsupported(feature)) if feature == "rejected")
    );
    let result = root.call_specialized(
        Specialization::new::<Parent>(Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[])),
        &[],
        &[],
        Box::new(|b| {
            let reused = b.call_specialized(
                Specialization::new::<Child>(Signature::new(
                    &[ScalarType::I32; 0],
                    ScalarType::I32,
                    &[],
                )),
                &[],
                &[],
                Box::new(|_| panic!("restaged a completed callee")),
            )?;
            b.ret(reused)?;
            Ok(())
        }),
    )?;
    root.ret(result)?;
    Ok(())
}
#[test]
fn a_failed_body_can_be_retried_and_its_completed_children_reused() {
    splice_testkit::each_backend!(B => {
        let code = B::build_with_options(Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[]), Box::new(retry), splice::BuildOptions::host(splice::Mode::Release).unwrap()).unwrap();
        assert_eq!(splice_testkit::call_i32(&code,&[], &mut []), Ok(17));
    });
}

#[test]
fn invalid_arguments_are_rejected_before_staging_or_reusing_a_callee()
-> Result<(), Box<dyn std::error::Error>> {
    use splice::frontend::{KernelBuilder, Specialization};
    use splice::hir::ScalarType;
    use splice::hir::{Buffer, Reg, ValueType};
    use splice::{BufferArg, BufferKind};

    fn stage(root: &mut dyn KernelBuilder) -> splice::Result<()> {
        struct Child;
        let arg = root.arg(0)?;
        let vector = root.local(ValueType::Vector(ScalarType::I32))?;
        let mask = root.local(ValueType::Mask(ScalarType::I32))?;
        fn unexpected(_: &mut dyn KernelBuilder) -> splice::Result<()> {
            panic!("staged an invalid call")
        }
        for args in [&[][..], &[Reg(u32::MAX)], &[vector], &[mask]] {
            assert!(
                root.call_kernel(
                    Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
                    args,
                    &[],
                    Box::new(unexpected)
                )
                .is_err()
            );
        }
        let read = [BufferKind::READ_I32];
        for (kinds, buffers) in [
            (&read[..], &[][..]),
            (&read, &[Buffer(2)]),
            (&[BufferKind::WRITE_I32], &[Buffer(0)]),
            (&[BufferKind::new(ScalarType::I64, false)], &[Buffer(0)]),
            (
                &[BufferKind::READ_I32, BufferKind::WRITE_I32],
                &[Buffer(1), Buffer(1)],
            ),
        ] {
            assert!(
                root.call_kernel(
                    Signature::new(&[ScalarType::I32; 1], ScalarType::I32, kinds),
                    &[arg],
                    buffers,
                    Box::new(unexpected)
                )
                .is_err()
            );
        }
        let key = || {
            Specialization::new::<Child>(Signature::new(
                &[ScalarType::I32; 1],
                ScalarType::I32,
                &[BufferKind::READ_I32, BufferKind::READ_I32],
            ))
        };
        let shared = [Buffer(1), Buffer(1)];
        assert!(
            root.call_specialized(key(), &[vector], &shared, Box::new(unexpected))
                .is_err()
        );
        let result = root.call_specialized(
            key(),
            &[arg],
            &shared,
            Box::new(|child| {
                child.ret(child.arg(0)?)?;
                Ok(())
            }),
        )?;
        assert!(
            root.call_specialized(key(), &[mask], &shared, Box::new(unexpected))
                .is_err()
        );
        root.ret(result)?;
        Ok(())
    }
    let kinds = [BufferKind::READ_I32, BufferKind::WRITE_I32];
    let ir = splice::frontend::materialize(
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &kinds),
        Box::new(stage),
        splice::BuildOptions::host(splice::Mode::Release)?,
    )?;
    splice_testkit::each_backend!(B => {
        for code in [
            Box::new(B::build_with_options(Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &kinds), Box::new(stage), splice::BuildOptions::host(splice::Mode::Release).unwrap())?) as Box<dyn Executable>,
            Box::new(B::compile(&ir)?),
        ] {
            assert_eq!(
                splice_testkit::call_i32(&code,
                    &[37],
                    &mut [BufferArg::read::<i32>(&[]), BufferArg::write::<i32>(&mut [])]
                )?,
                37
            );
        }
    });
    Ok(())
}

#[kernel]
fn store_ratio(dst: &mut [i32], x: i32, d: i32) -> i32 {
    dst[0] = x;
    let value = x / d;
    dst[0] = value;
    value
}
#[kernel]
fn two_buffers(left: &mut [i32], right: &mut [i32], d: i32) -> i32 {
    let first = store_ratio(left, 18, 3);
    store_ratio(right, first, d)
}
#[test]
fn shared_fallible_body_observes_each_calls_buffers_and_arguments()
-> Result<(), Box<dyn std::error::Error>> {
    let ir = two_buffers::ir()?;
    assert_eq!(bodies_and_calls(&ir, "store_ratio"), (1, 2));
    fn check<B: Codegen<Output: Executable>>(
        ir: &splice::TypedIr<two_buffers::Signature>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for code in [
            two_buffers::build::<B>(two_buffers::BuildArgs {})?,
            B::compile(ir)?,
        ] {
            let (mut left, mut right) = ([0], [0]);
            assert_eq!(
                code.call(two_buffers::CallArgs {
                    left: &mut left,
                    right: &mut right,
                    d: 2
                })?,
                3
            );
            assert_eq!((left, right), ([6], [3]));
            assert!(
                code.call(two_buffers::CallArgs {
                    left: &mut left,
                    right: &mut right,
                    d: 0
                })
                .is_err()
            );
            assert_eq!((left, right), ([6], [6]));
        }
        Ok(())
    }

    splice_testkit::each_backend!(B => check::<B>(&ir)?);
    Ok(())
}
