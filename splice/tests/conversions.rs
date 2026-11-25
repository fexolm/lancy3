//! Every lane conversion the verifier accepts, on every backend and both build
//! paths, against Rust `as` on boundary inputs (NaN, infinities, signed zero,
//! saturation edges and wrapping integers).
use splice::frontend::{self, KernelBuilder, Signature};
use splice::hir::{Buffer, Simd, SimdOp};
use splice::hir::{Constant, ScalarType};
use splice::simd::Element;
use splice::{BufferElement, BufferKind, BuildOptions, Codegen, Executable, Mode};

fn options() -> BuildOptions {
    BuildOptions::host(Mode::Release).unwrap()
}

trait Lane: BufferElement + Copy + std::fmt::Debug {
    fn samples() -> Vec<Self>;
    /// Equality that treats every NaN as the same value.
    fn same(self, other: Self) -> bool;
}

macro_rules! float_lane {
    ($ty:ident) => {
        impl Lane for $ty {
            fn samples() -> Vec<Self> {
                let mut values = vec![
                    $ty::NAN,
                    $ty::INFINITY,
                    $ty::NEG_INFINITY,
                    0.0,
                    -0.0,
                    0.5,
                    -0.5,
                    1.5,
                    -1.5,
                    2.5,
                    65535.0,
                    65536.0,
                    2147483647.0,
                    2147483648.0,
                    -2147483649.0,
                    4294967296.0,
                    9.3e18,
                    -9.3e18,
                    1.0e-40_f64 as $ty,
                    $ty::MAX,
                    $ty::MIN,
                ];
                values.push(1.0e300_f64 as $ty);
                values
            }
            fn same(self, other: Self) -> bool {
                (self.is_nan() && other.is_nan()) || self.to_bits() == other.to_bits()
            }
        }
    };
}
float_lane!(f32);
float_lane!(f64);

macro_rules! int_lane {
    ($ty:ident) => {
        impl Lane for $ty {
            fn samples() -> Vec<Self> {
                [
                    0i128,
                    1,
                    -1,
                    2,
                    -2,
                    127,
                    255,
                    32767,
                    65535,
                    65536,
                    -65536,
                    i32::MAX as i128,
                    i32::MIN as i128,
                    u32::MAX as i128,
                    1 << 32,
                    (1 << 53) + 1,
                    i64::MAX as i128,
                    i64::MIN as i128,
                ]
                .into_iter()
                .map(|v| v as $ty)
                .collect()
            }
            fn same(self, other: Self) -> bool {
                self == other
            }
        }
    };
}
int_lane!(u16);
int_lane!(i32);
int_lane!(u32);
int_lane!(i64);

trait As<T> {
    fn to(self) -> T;
}
macro_rules! as_impls {
    ($($from:ty => [$($to:ty),*]),* $(,)?) => {
        $($(impl As<$to> for $from {
            fn to(self) -> $to {
                self as $to
            }
        })*)*
    };
}
as_impls! {
    u16 => [u16, i32, u32],
    i32 => [i32, u32, f32, u16, i64],
    u32 => [i32, u32, f32, u16, i64],
    f32 => [i32, u32, f32, f64],
    i64 => [i64, f64, i32, u32],
    f64 => [i64, f64, f32],
}

#[derive(Clone, Copy, Debug)]
enum Conversion {
    Cast,
    Widen { high: bool },
    Narrow,
}

fn stage(
    from: Element,
    to: Element,
    conversion: Conversion,
) -> impl Fn(&mut dyn KernelBuilder) -> splice::Result<()> {
    move |b| {
        let load = Simd {
            element: from,
            op: SimdOp::Load(Buffer(0)),
        };
        let zero = b.constant(Constant::from(0usize))?;
        let low = b.simd(load, &[zero])?.unwrap();
        let (op, args) = match conversion {
            Conversion::Cast => (SimdOp::Cast(to), vec![low]),
            Conversion::Widen { high } => (SimdOp::Widen { to, high }, vec![low]),
            Conversion::Narrow => {
                let next =
                    b.constant(Constant::from(from.lanes(options().target.vector_bytes())))?;
                let high = b.simd(load, &[next])?.unwrap();
                (SimdOp::Narrow(to), vec![low, high])
            }
        };
        let converted = b.simd(Simd { element: from, op }, &args)?.unwrap();
        let store = Simd {
            element: to,
            op: SimdOp::Store(Buffer(1)),
        };
        b.simd(store, &[zero, converted])?;
        let result = b.constant(Constant::from(0))?;
        b.ret(result)?;
        Ok(())
    }
}

fn run<B: Codegen, F: Lane, T: Lane>(conversion: Conversion, input: &[F], expected: &[T])
where
    B::Output: Executable,
{
    let (from, to) = (F::ELEMENT, T::ELEMENT);
    let kinds = [BufferKind::new(from, false), BufferKind::new(to, true)];
    let ir = frontend::materialize(
        Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &kinds),
        Box::new(stage(from, to, conversion)),
        options(),
    )
    .unwrap();
    let paths = [
        (
            "build",
            B::build_with_options(
                Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &kinds),
                Box::new(stage(from, to, conversion)),
                options(),
            ),
        ),
        ("compile", B::compile(&ir)),
    ];
    for (path, code) in paths {
        let name = std::any::type_name::<B>();
        let code =
            code.unwrap_or_else(|e| panic!("{name} {path} {conversion:?} {from:?}->{to:?}: {e:?}"));
        let mut out = vec![T::default(); expected.len()];
        let result = code.call(
            &[],
            &mut [
                splice::BufferArg::read(input),
                splice::BufferArg::write(&mut out),
            ],
        );
        assert_eq!(
            result,
            Ok(0),
            "{name} {path} {conversion:?} {from:?}->{to:?}"
        );
        for (lane, (&got, &want)) in out.iter().zip(expected).enumerate() {
            assert!(
                got.same(want),
                "{name} {path} {conversion:?} {from:?}->{to:?} lane {lane} of {:?}: got {got:?}, expected {want:?}",
                input
            );
        }
    }
}

fn check<F: Lane + As<T>, T: Lane>(conversion: Conversion) {
    let lanes = F::ELEMENT.lanes(options().target.vector_bytes());
    let samples = F::samples();
    // Rotate the samples through every lane position across several windows.
    for window in 0..samples.len().div_ceil(lanes).max(1) + 1 {
        let input: Vec<F> = (0..2 * lanes)
            .map(|i| samples[(window * lanes + i) % samples.len()])
            .collect();
        let expected: Vec<T> = match conversion {
            Conversion::Cast => input[..lanes].iter().map(|&x| x.to()).collect(),
            Conversion::Widen { high } => {
                let half = lanes / 2;
                let start = if high { half } else { 0 };
                input[start..start + half].iter().map(|&x| x.to()).collect()
            }
            Conversion::Narrow => input.iter().map(|&x| x.to()).collect(),
        };
        splice_testkit::each_backend!(B => run::<B, F, T>(conversion, &input, &expected));
    }
}

#[test]
fn same_width_casts() {
    use Conversion::Cast;
    check::<u16, u16>(Cast);
    check::<i32, i32>(Cast);
    check::<i32, u32>(Cast);
    check::<i32, f32>(Cast);
    check::<u32, i32>(Cast);
    check::<u32, u32>(Cast);
    check::<u32, f32>(Cast);
    check::<f32, i32>(Cast);
    check::<f32, u32>(Cast);
    check::<f32, f32>(Cast);
    check::<i64, i64>(Cast);
    check::<i64, f64>(Cast);
    check::<f64, i64>(Cast);
    check::<f64, f64>(Cast);
}

#[test]
fn widening_conversions() {
    for high in [false, true] {
        let widen = Conversion::Widen { high };
        check::<u16, i32>(widen);
        check::<u16, u32>(widen);
        check::<i32, i64>(widen);
        check::<u32, i64>(widen);
        check::<f32, f64>(widen);
    }
}

#[test]
fn narrowing_conversions() {
    use Conversion::Narrow;
    check::<i32, u16>(Narrow);
    check::<u32, u16>(Narrow);
    check::<i64, i32>(Narrow);
    check::<i64, u32>(Narrow);
    check::<f64, f32>(Narrow);
}

#[test]
fn rejected_conversions_stay_rejected() {
    let kinds = [
        BufferKind::new(Element::F64, false),
        BufferKind::new(Element::I32, true),
    ];
    let error = frontend::materialize(
        Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &kinds),
        Box::new(stage(Element::F64, Element::I32, Conversion::Cast)),
        options(),
    );
    assert!(error.is_err());
}
