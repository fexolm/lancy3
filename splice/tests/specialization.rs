use splice::{Codegen, ComptimeKey, Executable, kernel};
use splice_cranelift::Cranelift;
#[cfg(target_arch = "x86_64")]
use splice_lancy::Lancy;
use splice_llvm::Llvm;
use splice_text::Text;
use std::cell::Cell;

thread_local! {
    static PLAIN: Cell<usize> = const { Cell::new(0) };
    static SPECIALIZED: Cell<usize> = const { Cell::new(0) };
}

fn one() -> i32 {
    PLAIN.set(PLAIN.get() + 1);
    1
}

#[derive(ComptimeKey)]
enum Mode<'a> {
    Add(&'a [i32]),
    Sub(&'a [i32]),
}

impl Mode<'_> {
    fn bias(&self) -> i32 {
        SPECIALIZED.set(SPECIALIZED.get() + 1);
        match self {
            Self::Add(values) => values.iter().sum(),
            Self::Sub(values) => -values.iter().sum::<i32>(),
        }
    }
}

#[kernel]
fn increment(x: i32) -> i32 {
    x + one()
}

#[kernel]
fn apply(#[comptime] mode: &Mode<'_>, #[comptime] scale: i32, x: i32) -> i32 {
    increment(x) + mode.bias() * scale
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
    PLAIN.set(0);
    SPECIALIZED.set(0);
    let code = chain::build::<B>(modes).unwrap();
    assert_eq!((PLAIN.get(), SPECIALIZED.get()), (1, 4));
    assert_eq!(code.call(&[12], &mut []), Ok(expected));
}

#[test]
fn repeated_values_skip_staging_across_parents_and_keep_all_calls() {
    let a = vec![2, 3];
    let b = a.clone();
    // Different allocations with the same contents share a callee. Add([5])
    // emits the same bias but is a different set of specialization arguments.
    let modes = [Mode::Add(&a), Mode::Add(&b), Mode::Sub(&a), Mode::Add(&[5])];
    let expected = chain(&modes, 12);
    #[cfg(target_arch = "x86_64")]
    check_direct::<Lancy>(&modes, expected);
    check_direct::<Llvm>(&modes, expected);
    check_direct::<Cranelift>(&modes, expected);

    PLAIN.set(0);
    SPECIALIZED.set(0);
    let ir = chain::ir(&modes).unwrap();
    assert_eq!((PLAIN.get(), SPECIALIZED.get()), (1, 4));
    let listing = Text::compile(&ir).unwrap();
    assert_eq!(listing.matches("fn apply.").count(), 4);
    assert_eq!(listing.matches(" = apply.").count(), 5);
    assert_eq!(listing.matches("fn increment.").count(), 1);
    assert_eq!(listing.matches(" = increment.").count(), 5);
    assert_eq!(chain::build::<Text>(&modes).unwrap(), listing);
    let kernels: Vec<Box<dyn Executable>> = vec![
        #[cfg(target_arch = "x86_64")]
        Box::new(Lancy::compile(&ir).unwrap()),
        Box::new(Llvm::compile(&ir).unwrap()),
        Box::new(Cranelift::compile(&ir).unwrap()),
    ];
    for code in kernels {
        assert_eq!(code.call(&[12], &mut []), Ok(expected));
    }
}

trait Flavor: ComptimeKey {
    fn value() -> i32;
}
#[derive(ComptimeKey)]
struct First;
#[derive(ComptimeKey)]
struct Second;
impl Flavor for First {
    fn value() -> i32 {
        2
    }
}
impl Flavor for Second {
    fn value() -> i32 {
        7
    }
}

#[kernel]
fn affine<T: Flavor, const N: i32>(x: i32) -> i32 {
    x * T::value() + N
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
    let listing = instances::build::<Text>().unwrap();
    assert_eq!(listing.matches("fn affine.").count(), 3);
    assert_eq!(listing.matches(" = affine.").count(), 4);
    let expected = instances(3);
    #[cfg(target_arch = "x86_64")]
    assert_eq!(
        instances::build::<Lancy>().unwrap().call(&[3], &mut []),
        Ok(expected)
    );
    assert_eq!(
        instances::build::<Llvm>().unwrap().call(&[3], &mut []),
        Ok(expected)
    );
}

fn retry(root: &mut dyn splice::frontend::KernelBuilder) -> splice::Result<()> {
    use splice::frontend::Specialization;
    struct Parent;
    struct Child;
    let failed = root.call_specialized(
        Specialization::new::<Parent>(0, &[]),
        &[],
        &[],
        Box::new(|b| {
            b.call_specialized(
                Specialization::new::<Child>(0, &[]),
                &[],
                &[],
                Box::new(|c| {
                    let value = c.constant(17);
                    c.ret(value);
                    Ok(())
                }),
            )?;
            Err("rejected".into())
        }),
    );
    assert_eq!(failed, Err("rejected".into()));
    let result = root.call_specialized(
        Specialization::new::<Parent>(0, &[]),
        &[],
        &[],
        Box::new(|b| {
            let reused = b.call_specialized(
                Specialization::new::<Child>(0, &[]),
                &[],
                &[],
                Box::new(|_| panic!("restaged a completed callee")),
            )?;
            b.ret(reused);
            Ok(())
        }),
    )?;
    root.ret(result);
    Ok(())
}
#[test]
fn a_failed_body_can_be_retried_and_its_completed_children_reused() {
    #[cfg(target_arch = "x86_64")]
    let lancy = Lancy::build(0, &[], Box::new(retry)).unwrap();
    let llvm = Llvm::build(0, &[], Box::new(retry)).unwrap();
    let cranelift = Cranelift::build(0, &[], Box::new(retry)).unwrap();
    let kernels: &[&dyn Executable] = &[
        #[cfg(target_arch = "x86_64")]
        &lancy,
        &llvm,
        &cranelift,
    ];
    for code in kernels {
        assert_eq!(code.call(&[], &mut []), Ok(17));
    }
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
fn shared_fallible_body_observes_each_calls_buffers_and_arguments() -> splice::Result<()> {
    use splice::BufferArg::Write;
    let ir = two_buffers::ir()?;
    assert_eq!(Text::compile(&ir)?.matches("fn store_ratio.").count(), 1);
    let kernels: Vec<Box<dyn Executable>> = vec![
        #[cfg(target_arch = "x86_64")]
        Box::new(two_buffers::build::<Lancy>()?),
        #[cfg(target_arch = "x86_64")]
        Box::new(Lancy::compile(&ir)?),
        Box::new(two_buffers::build::<Llvm>()?),
        Box::new(Llvm::compile(&ir)?),
        Box::new(two_buffers::build::<Cranelift>()?),
        Box::new(Cranelift::compile(&ir)?),
    ];
    for code in kernels {
        let (mut left, mut right) = ([0], [0]);
        assert_eq!(
            code.call(&[2], &mut [Write(&mut left), Write(&mut right)])?,
            3
        );
        assert_eq!((left, right), ([6], [3]));
        assert!(
            code.call(&[0], &mut [Write(&mut left), Write(&mut right)])
                .is_err()
        );
        assert_eq!((left, right), ([6], [6]));
    }
    Ok(())
}
