//! `#[intrinsic("name")]` functions and methods: a backend that implements
//! the name emits its own code, any other compiles the body or, under
//! `IntrinsicPolicy::Error`, fails the build. Every backend, both build
//! paths and both modes.
use splice::frontend::Signature;
use splice::hir::{Instr, ScalarType};
use splice::intrinsic::Intrinsic;
use splice::{BuildOptions, Codegen, Executable, IntrinsicPolicy, Mode, kernel};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

mod ops {
    use splice::kernel;

    /// Implemented by every backend.
    #[kernel]
    #[intrinsic("count_ones")]
    pub fn popcount(x: u64) -> u32 {
        let mut v = x;
        let mut count = 0u32;
        while v != 0 {
            count += (v & 1) as u32;
            v >>= 1;
        }
        count
    }

    /// Implemented by Cranelift and LLVM, not by Lancy.
    #[allow(clippy::manual_rotate)]
    #[kernel]
    #[intrinsic("rotate_left")]
    pub fn rotl(x: i32, n: u32) -> i32 {
        let v = x as u32;
        let s = n % 32;
        if s == 0 {
            x
        } else {
            ((v << s) | (v >> (32 - s))) as i32
        }
    }

    /// Implemented by LLVM where the target fuses it.
    #[kernel]
    #[intrinsic("mul_add")]
    pub fn fma(a: f64, b: f64, c: f64) -> f64 {
        a * b + c
    }

    pub struct Bits;

    #[kernel]
    impl Bits {
        /// Implemented by every backend.
        #[intrinsic("leading_zeros")]
        pub fn leading_zeros(x: i32) -> u32 {
            let v = x as u32;
            let mut n = 0u32;
            while n < 32 && (v & (0x8000_0000u32 >> n)) == 0 {
                n += 1;
            }
            n
        }

        /// Implemented by no backend.
        #[intrinsic("acme.low_bit")]
        pub fn low_bit(x: u32) -> u32 {
            x & 1
        }
    }
}

#[kernel]
fn count(x: u64) -> u32 {
    ops::popcount(x)
}

#[kernel]
fn rotate(x: i32, n: u32) -> i32 {
    ops::rotl(x, n)
}

#[kernel]
fn fused(a: f64, b: f64, c: f64) -> f64 {
    ops::fma(a, b, c)
}

#[kernel]
fn leading(x: i32) -> u32 {
    ops::Bits::leading_zeros(x)
}

/// Two calls with one signature share one staged fallback.
#[kernel]
fn low_bits(x: u32, y: u32) -> u32 {
    ops::Bits::low_bit(x) + ops::Bits::low_bit(y)
}

/// The events logged with target `splice::intrinsic`.
#[derive(Clone, Default)]
struct Log(Arc<Mutex<Vec<String>>>);

struct Message(String);

impl Visit for Message {
    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        use std::fmt::Write;
        write!(self.0, " {}={value:?}", field.name()).unwrap();
    }
}

impl<S: tracing::Subscriber> Layer<S> for Log {
    fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
        if event.metadata().target() == "splice::intrinsic" {
            let mut message = Message(String::new());
            event.record(&mut message);
            self.0.lock().unwrap().push(message.0);
        }
    }
}

/// `build` with a subscriber that keeps the intrinsic log.
fn logged<T>(build: impl FnOnce() -> T) -> (T, Vec<String>) {
    let log = Log::default();
    let subscriber = tracing_subscriber::registry().with(log.clone());
    let result = tracing::subscriber::with_default(subscriber, build);
    let events = log.0.lock().unwrap().clone();
    (result, events)
}

/// The name a backend gives itself in diagnostics.
fn backend<B>() -> &'static str {
    let name = std::any::type_name::<B>();
    if name.ends_with("::Lancy") {
        "lancy"
    } else if name.ends_with("::Cranelift") {
        "cranelift"
    } else if name.ends_with("::LlvmManualSimd") {
        "llvm-manual-simd"
    } else {
        assert!(name.ends_with("::Llvm"), "{name}");
        "llvm"
    }
}

fn host_fuses() -> bool {
    let target = splice::Target::host().unwrap();
    target.x86().is_none_or(|isa| isa.fma)
}

#[derive(Clone, Copy, Debug)]
enum Path {
    /// `build::<B>`.
    Direct,
    /// `ir()` and `B::compile`.
    Portable,
}

/// Build with `build` on both paths in both modes under both policies:
/// a supported intrinsic builds without a log; an unsupported one builds
/// its body with a log naming it and the backend, or fails under `Error`.
/// Each built kernel goes to `check`.
fn matrix<B: Codegen, K>(
    intrinsic: Intrinsic<'_>,
    supported: bool,
    build: impl Fn(BuildOptions, Path) -> splice::Result<K>,
    check: impl Fn(&K),
) {
    let name = backend::<B>();
    let target = splice::Target::host().unwrap();
    assert_eq!(
        B::supports_intrinsic(&intrinsic, &target),
        supported,
        "{name} {intrinsic}"
    );
    for mode in [Mode::Release, Mode::Debug] {
        for policy in [IntrinsicPolicy::Fallback, IntrinsicPolicy::Error] {
            let options = BuildOptions::new(mode, target).with_unsupported_intrinsic(policy);
            for path in [Path::Direct, Path::Portable] {
                let context = format!("{name} {path:?} {mode:?} {policy:?} {intrinsic}");
                let (built, log) = logged(|| build(options, path));
                match (supported, policy) {
                    (true, _) => {
                        assert!(log.is_empty(), "{context}: {log:?}");
                        check(&built.expect(&context));
                    }
                    (false, IntrinsicPolicy::Fallback) => {
                        assert!(!log.is_empty(), "{context}: no log");
                        for event in &log {
                            assert!(
                                event.contains(intrinsic.name) && event.contains(name),
                                "{context}: {event}"
                            );
                        }
                        check(&built.expect(&context));
                    }
                    (false, IntrinsicPolicy::Error) => {
                        let Err(error) = built else {
                            panic!("{context}: built");
                        };
                        let message = error.to_string();
                        assert!(
                            message.contains(intrinsic.name)
                                && message.contains(&format!("`{name}`")),
                            "{context}: {message}"
                        );
                    }
                }
            }
        }
    }
}

fn signature(params: &'static [ScalarType], result: ScalarType) -> Signature<'static> {
    Signature::new(params, result, &[])
}

#[test]
fn every_backend_implements_count_ones() {
    fn check<B: Codegen<Output: Executable>>() {
        let intrinsic =
            Intrinsic::new("count_ones", signature(&[ScalarType::U64], ScalarType::U32));
        matrix::<B, _>(
            intrinsic,
            true,
            |options, path| match path {
                Path::Direct => count::build_with_options::<B>(options, count::BuildArgs {}),
                Path::Portable => B::compile(&count::ir_with_options(options)?),
            },
            |code| {
                for x in [0, 1, u64::MAX, 1 << 63, 0x5555_5555_5555_5555, 0xf0] {
                    assert_eq!(
                        code.call(count::CallArgs { x }),
                        Ok(ops::popcount(x)),
                        "{x}"
                    );
                    assert_eq!(ops::popcount(x), x.count_ones());
                }
            },
        );
    }
    splice_testkit::each_backend!(B => check::<B>());
}

#[test]
fn every_backend_implements_leading_zeros_of_a_method() {
    fn check<B: Codegen<Output: Executable>>() {
        let intrinsic = Intrinsic::new(
            "leading_zeros",
            signature(&[ScalarType::I32], ScalarType::U32),
        );
        matrix::<B, _>(
            intrinsic,
            true,
            |options, path| match path {
                Path::Direct => leading::build_with_options::<B>(options, leading::BuildArgs {}),
                Path::Portable => B::compile(&leading::ir_with_options(options)?),
            },
            |code| {
                for x in [0, 1, -1, i32::MIN, i32::MAX, 0x00f0_0000] {
                    let expected = ops::Bits::leading_zeros(x);
                    assert_eq!(expected, x.leading_zeros());
                    assert_eq!(code.call(leading::CallArgs { x }), Ok(expected), "{x}");
                }
            },
        );
    }
    splice_testkit::each_backend!(B => check::<B>());
}

#[test]
fn rotate_left_is_native_except_on_lancy() {
    fn check<B: Codegen<Output: Executable>>() {
        let intrinsic = Intrinsic::new(
            "rotate_left",
            signature(&[ScalarType::I32, ScalarType::U32], ScalarType::I32),
        );
        matrix::<B, _>(
            intrinsic,
            backend::<B>() != "lancy",
            |options, path| match path {
                Path::Direct => rotate::build_with_options::<B>(options, rotate::BuildArgs {}),
                Path::Portable => B::compile(&rotate::ir_with_options(options)?),
            },
            |code| {
                for (x, n) in [
                    (0, 0),
                    (1, 31),
                    (i32::MIN, 1),
                    (-1, 7),
                    (0x1234_5678, 32),
                    (0x1234_5678, u32::MAX),
                    (i32::MAX, 33),
                ] {
                    let expected = ops::rotl(x, n);
                    assert_eq!(expected, x.rotate_left(n));
                    assert_eq!(
                        code.call(rotate::CallArgs { x, n }),
                        Ok(expected),
                        "{x} {n}"
                    );
                }
            },
        );
    }
    splice_testkit::each_backend!(B => check::<B>());
}

#[test]
fn mul_add_is_native_on_llvm_with_fused_multiply_add() {
    fn check<B: Codegen<Output: Executable>>() {
        let intrinsic =
            Intrinsic::new("mul_add", signature(&[ScalarType::F64; 3], ScalarType::F64));
        matrix::<B, _>(
            intrinsic,
            backend::<B>().starts_with("llvm") && host_fuses(),
            |options, path| match path {
                Path::Direct => fused::build_with_options::<B>(options, fused::BuildArgs {}),
                Path::Portable => B::compile(&fused::ir_with_options(options)?),
            },
            |code| {
                // Exact products: the fused and the unfused result agree.
                for (a, b, c) in [
                    (0.0, 0.0, 0.0),
                    (-0.0, 1.0, 0.0),
                    (1.5, 2.0, -3.0),
                    (f64::NAN, 1.0, 1.0),
                    (f64::INFINITY, 0.0, 1.0),
                    (f64::MAX, 1.0, f64::MAX),
                    (f64::MIN_POSITIVE, 0.5, 0.0),
                ] {
                    let expected = ops::fma(a, b, c);
                    assert!(expected.to_bits() == a.mul_add(b, c).to_bits() || expected.is_nan());
                    let actual = code.call(fused::CallArgs { a, b, c }).unwrap();
                    if expected.is_nan() {
                        assert!(actual.is_nan(), "{a} {b} {c}: {actual}");
                    } else {
                        assert_eq!(actual.to_bits(), expected.to_bits(), "{a} {b} {c}");
                    }
                }
            },
        );
    }
    splice_testkit::each_backend!(B => check::<B>());
}

#[test]
fn an_unknown_intrinsic_compiles_its_body_or_fails() {
    fn check<B: Codegen<Output: Executable>>() {
        let intrinsic = Intrinsic::new(
            "acme.low_bit",
            signature(&[ScalarType::U32], ScalarType::U32),
        );
        matrix::<B, _>(
            intrinsic,
            false,
            |options, path| match path {
                Path::Direct => low_bits::build_with_options::<B>(options, low_bits::BuildArgs {}),
                Path::Portable => B::compile(&low_bits::ir_with_options(options)?),
            },
            |code| {
                for (x, y) in [(0, 0), (1, u32::MAX), (2, 3)] {
                    assert_eq!(code.call(low_bits::CallArgs { x, y }), Ok(low_bits(x, y)));
                }
            },
        );
    }
    splice_testkit::each_backend!(B => check::<B>());
}

/// The portable IR records each call as an intrinsic with the body as its
/// fallback, staged once for calls with one signature.
#[test]
fn portable_ir_records_the_intrinsic_and_one_fallback() {
    let ir = low_bits::ir().unwrap();
    let fallbacks: Vec<_> = ir
        .instructions()
        .iter()
        .filter_map(|op| match op {
            Instr::Intrinsic { name, fallback, .. } => {
                assert_eq!(&**name, "acme.low_bit");
                Some(fallback)
            }
            _ => None,
        })
        .collect();
    assert_eq!(fallbacks.len(), 2);
    assert!(Arc::ptr_eq(fallbacks[0], fallbacks[1]));
    assert!(
        !ir.instructions()
            .iter()
            .any(|op| matches!(op, Instr::Call { .. }))
    );
}

/// A supported intrinsic compiles to the backend's instruction, without
/// the body; an unsupported one calls its body.
#[test]
fn a_supported_intrinsic_lowers_without_its_body() {
    let target = splice::Target::host().unwrap();
    let lancy = |ir: &splice::Function| splice::lancy::lower(ir, &target).unwrap().to_string();
    let listing = lancy(&count::ir().unwrap());
    assert!(
        listing.contains("popcnt") && !listing.contains("call "),
        "{listing}"
    );
    let listing = lancy(&rotate::ir().unwrap());
    assert!(listing.contains("call "), "{listing}");
    #[cfg(feature = "llvm")]
    {
        let llvm = |ir: &splice::Function| splice::llvm::to_llvm_ir(ir, &target).unwrap();
        // The kernel and its public entry point, and a function per body.
        let functions = |listing: &str| listing.matches("\ndefine ").count();
        let listing = llvm(&rotate::ir().unwrap());
        assert!(
            listing.contains("@llvm.fshl") && functions(&listing) == 2,
            "{listing}"
        );
        let listing = llvm(&low_bits::ir().unwrap());
        assert_eq!(functions(&listing), 3, "{listing}");
    }
}

/// An intrinsic function built as a kernel itself is its body.
#[test]
fn an_intrinsic_built_as_a_kernel_is_its_body() {
    fn check<B: Codegen<Output: Executable>>() {
        let options = BuildOptions::host(Mode::Release)
            .unwrap()
            .with_unsupported_intrinsic(IntrinsicPolicy::Error);
        let code = ops::rotl::build_with_options::<B>(options, ops::rotl::BuildArgs {}).unwrap();
        assert_eq!(code.call(ops::rotl::CallArgs { x: -1, n: 3 }), Ok(-1));
    }
    splice_testkit::each_backend!(B => check::<B>());
}

#[test]
fn the_default_policy_compiles_the_body() {
    let options = BuildOptions::host(Mode::Release).unwrap();
    assert_eq!(options.unsupported_intrinsic, IntrinsicPolicy::Fallback);
}
