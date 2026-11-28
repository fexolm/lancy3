//! A kernel can only produce valid `char`s: the builder API cannot stage a
//! `char` constant that is a surrogate or above `char::MAX`, nor a cast to
//! `char` from anything but `u8`, on any backend or build route, so a safe
//! call never writes an invalid `char` into a `&mut [char]`.
use splice::frontend::{self, KernelBuilder, Signature};
use splice::hir::{Constant, ScalarType};
use splice::{BufferArg, BufferKind, BuildOptions, Codegen, Executable, Mode};

/// A runner of the compiled kernel (`None` for the portable IR alone), or
/// the build error.
/// A body part that stages the stored value.
type Value = fn(&mut dyn KernelBuilder) -> splice::Result<splice::hir::Reg>;

type Outcome = Result<Option<Box<dyn Fn(u64, &mut [char]) -> i32>>, String>;

const KINDS: [BufferKind; 1] = [BufferKind::new(ScalarType::Char, true)];

/// `out[0] = value(param)`; returns 0.
fn store_char(value: Value) -> impl Fn(&mut dyn KernelBuilder) -> splice::Result<()> {
    move |b| {
        let value = value(b)?;
        let zero = b.constant(Constant::index(0))?;
        b.store(b.buffer(0)?, zero, value)?;
        let result = b.constant(Constant::from(0))?;
        b.ret(result)
    }
}

/// Stage `body` with one parameter of `param` through `materialize` and
/// through each backend's direct build, in both modes.
fn each_route(param: ScalarType, body: Value, mut check: impl FnMut(&str, Outcome)) {
    let params = [param];
    for mode in [Mode::Release, Mode::Debug] {
        let options = BuildOptions::host(mode).unwrap();
        let signature = || Signature::new(&params, ScalarType::I32, &KINDS);
        let ir = frontend::materialize(signature(), Box::new(store_char(body)), options);
        let portable = ir.as_ref().map(|_| None).map_err(|e| e.to_string());
        check(&format!("materialize {mode:?}"), portable);
        fn run<B: Codegen>(code: splice::Result<B::Output>) -> Outcome
        where
            B::Output: Executable + 'static,
        {
            code.map_err(|e| e.to_string()).map(|code| {
                Some(Box::new(move |word: u64, out: &mut [char]| {
                    let result = code.call(&[word], &mut [BufferArg::write(out)]).unwrap();
                    result as i32
                }) as Box<dyn Fn(u64, &mut [char]) -> i32>)
            })
        }
        splice_testkit::each_backend!(B => {
            let name = std::any::type_name::<B>();
            let direct = B::build_with_options(signature(), Box::new(store_char(body)), options);
            check(&format!("{name} build {mode:?}"), run::<B>(direct));
            if let Ok(ir) = &ir {
                check(&format!("{name} compile {mode:?}"), run::<B>(B::compile(ir)));
            }
        });
    }
}

fn rejected(route: &str, result: Outcome, what: &str) {
    match result {
        Ok(_) => panic!("{route}: {what} was accepted"),
        Err(error) => assert!(error.contains(what), "{route}: {error}"),
    }
}

#[test]
fn invalid_char_constants_are_rejected_on_every_route() {
    fn surrogate(b: &mut dyn KernelBuilder) -> splice::Result<splice::hir::Reg> {
        b.constant(Constant::new(ScalarType::Char, 0xD800))
    }
    fn last_surrogate(b: &mut dyn KernelBuilder) -> splice::Result<splice::hir::Reg> {
        b.constant(Constant::new(ScalarType::Char, 0xDFFF))
    }
    fn above_max(b: &mut dyn KernelBuilder) -> splice::Result<splice::hir::Reg> {
        b.constant(Constant::new(ScalarType::Char, 0x11_0000))
    }
    for body in [surrogate as Value, last_surrogate, above_max] {
        each_route(ScalarType::I32, body, |route, result| {
            rejected(route, result, "is not a char");
        });
    }
}

#[test]
fn valid_char_constants_are_stored() {
    fn max(b: &mut dyn KernelBuilder) -> splice::Result<splice::hir::Reg> {
        b.constant(Constant::new(ScalarType::Char, char::MAX as u64))
    }
    fn before_surrogates(b: &mut dyn KernelBuilder) -> splice::Result<splice::hir::Reg> {
        b.constant(Constant::new(ScalarType::Char, 0xD7FF))
    }
    for (body, want) in [(max as Value, char::MAX), (before_surrogates, '\u{D7FF}')] {
        each_route(ScalarType::I32, body, |route, result| {
            let run = result.unwrap_or_else(|e| panic!("{route}: {e}"));
            if let Some(run) = run {
                let mut out = ['a'];
                assert_eq!(run(0, &mut out), 0, "{route}");
                assert_eq!(out, [want], "{route}");
            }
        });
    }
}

#[test]
fn only_u8_casts_to_char_on_every_route() {
    fn param_as_char(b: &mut dyn KernelBuilder) -> splice::Result<splice::hir::Reg> {
        let param = b.arg(0)?;
        b.cast(param, ScalarType::Char)
    }
    for wide in [
        ScalarType::I8,
        ScalarType::U16,
        ScalarType::I32,
        ScalarType::U32,
        ScalarType::U64,
        ScalarType::Index,
        ScalarType::F32,
        ScalarType::Bool,
    ] {
        each_route(wide, param_as_char, |route, result| {
            rejected(&format!("{wide} {route}"), result, "no cast from");
        });
    }
    each_route(ScalarType::U8, param_as_char, |route, result| {
        let run = result.unwrap_or_else(|e| panic!("{route}: {e}"));
        if let Some(run) = run {
            let mut out = ['a'];
            assert_eq!(run(0xE9, &mut out), 0, "{route}");
            assert_eq!(out, ['é'], "{route}");
        }
    });
}
