//! Kernel intrinsics: body-less methods of `#[intrinsic]` traits that every
//! backend implements. `splice::Math` on every integer type (8 to 64 bits), every backend,
//! both build paths and both modes; a user's intrinsic that no backend
//! implements fails the build and names the intrinsic and the backend.
use splice::hir::Instr;
use splice::intrinsic::IntrinsicId;
use splice::{BuildOptions, Codegen, Executable, Math, Mode, kernel};

macro_rules! math_kernels {
    ($($name:ident: $ty:ident, $unsigned:ident;)*) => {$(
        /// Every `Math` operation on `x`, packed into one word.
        #[kernel]
        fn $name(x: $ty, n: u32) -> u64 {
            let ones: u32 = x.count_ones();
            let leading: u32 = x.leading_zeros();
            let trailing: u32 = x.trailing_zeros();
            let rotated: $ty = x.rotate_left(n);
            ((ones as u64) | (leading as u64) << 8 | (trailing as u64) << 16)
                ^ (rotated as $unsigned as u64) << 24
        }
    )*};
}
math_kernels! {
    math_i8: i8, u8;
    math_u8: u8, u8;
    math_i16: i16, u16;
    math_u16: u16, u16;
    math_i32: i32, u32;
    math_u32: u32, u32;
    math_i64: i64, u64;
    math_u64: u64, u64;
}

fn modes() -> [BuildOptions; 2] {
    [Mode::Release, Mode::Debug].map(|mode| BuildOptions::host(mode).unwrap())
}

const COUNTS: [u32; 6] = [0, 1, 7, 31, 32, 100];

/// The last segment of the backend's type name, as its errors spell it.
fn backend_name<B>() -> &'static str {
    std::any::type_name::<B>().rsplit("::").next().unwrap()
}

#[test]
fn math_intrinsics_match_rust_on_every_backend_path_and_mode() {
    fn check<B: Codegen<Output: Executable>>() {
        macro_rules! run {
            ($name:ident, $values:expr) => {
                for options in modes() {
                    let direct =
                        $name::build_with_options::<B>(options, $name::BuildArgs {}).unwrap();
                    let portable = B::compile(&$name::ir_with_options(options).unwrap()).unwrap();
                    for x in $values {
                        for n in COUNTS {
                            for code in [&direct, &portable] {
                                assert_eq!(
                                    code.call($name::CallArgs { x, n }),
                                    Ok($name(x, n)),
                                    "{} {} {x} {n}",
                                    backend_name::<B>(),
                                    stringify!($name)
                                );
                            }
                        }
                    }
                }
            };
        }
        run!(math_i8, [0, 1, -1, 0x30, i8::MIN, i8::MAX]);
        run!(math_u8, [0, 1, u8::MAX, 0x81, 12]);
        run!(math_i16, [0, 1, -1, 0x0f00, i16::MIN, i16::MAX]);
        run!(math_u16, [0, 1, u16::MAX, 0x8001, 12]);
        run!(math_i32, [0, 1, -1, 0x00f0_0000, i32::MIN, i32::MAX]);
        run!(math_u32, [0, 1, u32::MAX, 0x8000_0001, 12]);
        run!(math_i64, [0, 1, -1, 1 << 40, i64::MIN, i64::MAX]);
        run!(math_u64, [0, 1, u64::MAX, 1 << 63, 0x0f00]);
    }
    splice_testkit::each_backend!(B => check::<B>());
}

/// A call reaches the portable IR as one intrinsic, named structurally.
#[test]
fn math_calls_record_structural_intrinsic_ids() {
    let ir = math_i64::ir().unwrap();
    let ids: Vec<String> = splice_testkit::instructions(&ir)
        .filter_map(|op| match op {
            Instr::Intrinsic { id, .. } => Some(id.to_string()),
            _ => None,
        })
        .collect();
    let math = "splice_core::math::Math";
    assert_eq!(
        ids,
        [
            format!("`<i64 as {math}>::count_ones(i64) -> u32`"),
            format!("`<i64 as {math}>::leading_zeros(i64) -> u32`"),
            format!("`<i64 as {math}>::trailing_zeros(i64) -> u32`"),
            format!("`<i64 as {math}>::rotate_left(i64, u32) -> i64`"),
        ]
    );
}

/// Every backend implements every intrinsic Splice declares.
#[test]
fn every_backend_implements_every_declared_intrinsic() {
    let declared = splice::intrinsic::declared();
    assert_eq!(declared.len(), 32, "{declared:?}");
    let target = splice::Target::host().unwrap();
    fn check<B: Codegen>(declared: &[IntrinsicId], target: &splice::Target) {
        for id in declared {
            assert!(
                B::supports_intrinsic(id, target),
                "{} does not implement {id}",
                backend_name::<B>()
            );
        }
        assert!(!B::supports_intrinsic(
            &<u32 as acme::LowBit>::__SPLICE_INTRINSICS[0],
            target
        ));
    }
    splice_testkit::each_backend!(B => check::<B>(&declared, &target));
}

mod acme {
    /// An intrinsic no backend implements.
    #[splice::intrinsic]
    pub trait LowBit {
        fn low_bit(self) -> u32;
    }

    #[splice::intrinsic]
    impl LowBit for u32 {
        fn low_bit(self) -> u32;
    }

    /// An opaque type only backends could represent, and its intrinsics.
    #[splice::intrinsic]
    pub struct Atomic<T>;

    #[splice::intrinsic]
    impl<T> Atomic<T> {
        pub fn compare_and_swap(&self, val: T, val2: T) -> bool;
    }
}

/// An intrinsic struct's methods are intrinsics named by the struct, its
/// generic arguments and the signature; no backend implements them.
#[test]
fn intrinsic_struct_methods_have_structural_ids() {
    let [id] = acme::Atomic::<i64>::__SPLICE_INTRINSICS else {
        panic!("one intrinsic")
    };
    let atomic = format!("{}::acme::Atomic<i64>", module_path!());
    assert_eq!(
        id.to_string(),
        format!("`{atomic}::compare_and_swap({atomic}, i64, i64) -> bool`")
    );
    assert_ne!(*id, acme::Atomic::<i32>::__SPLICE_INTRINSICS[0]);
    let target = splice::Target::host().unwrap();
    fn check<B: Codegen>(id: &IntrinsicId, target: &splice::Target) {
        assert!(
            !B::supports_intrinsic(id, target),
            "{}",
            backend_name::<B>()
        );
    }
    splice_testkit::each_backend!(B => check::<B>(id, &target));
}

#[kernel]
fn low_bit(x: u32) -> u32 {
    use acme::LowBit;
    x.low_bit() + 1
}

/// An intrinsic the backend does not implement fails the build on both
/// paths, in both modes, naming the intrinsic and the backend.
#[test]
fn an_unimplemented_intrinsic_fails_the_build_naming_it_and_the_backend() {
    fn check<B: Codegen<Output: Executable>>() {
        let expected = format!(
            "intrinsic `<u32 as {}::acme::LowBit>::low_bit(u32) -> u32` is not implemented by backend `{}`",
            module_path!(),
            backend_name::<B>()
        );
        for options in modes() {
            let direct = low_bit::build_with_options::<B>(options, low_bit::BuildArgs {});
            let ir = low_bit::ir_with_options(options).unwrap();
            for error in [direct.err(), B::compile(&ir).err()] {
                let error = error.expect("the build fails").to_string();
                assert!(error.contains(&expected), "{error}");
            }
        }
    }
    splice_testkit::each_backend!(B => check::<B>());
}

/// Outside a kernel an intrinsic panics: only a kernel calls it.
#[test]
#[should_panic(expected = "is a kernel intrinsic")]
fn an_intrinsic_called_from_rust_panics() {
    let _ = Math::count_ones(std::hint::black_box(3u64));
}
