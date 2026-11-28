//! Integer intrinsics on each backend, staging path and build mode.
use splice::hir::Instr;
use splice::hir::{Intrinsic, MathOp, ScalarType};
use splice::{BuildOptions, Codegen, Executable, Mode, kernel};

macro_rules! math_kernels {
    ($($name:ident: $ty:ident, $unsigned:ident;)*) => {$(
        /// Exercise every integer intrinsic on `x`.
        #[kernel]
        fn $name(x: $ty, n: u32) -> u64 {
            let ones: u32 = x.count_ones();
            let leading: u32 = x.leading_zeros();
            let trailing: u32 = x.trailing_zeros();
            let rotated: $ty = x.rotate_left(n);
            ((ones as u64) | (leading as u64) << 8 | (trailing as u64) << 16)
                ^ (rotated as $unsigned as u64)
        }
    )*};
}
math_kernels! {
    math_i32: i32, u32;
    math_u32: u32, u32;
    math_i64: i64, u64;
    math_u64: u64, u64;
}

fn modes() -> [BuildOptions; 2] {
    [Mode::Release, Mode::Debug].map(|mode| BuildOptions::host(mode).unwrap())
}

const COUNTS: [u32; 9] = [0, 1, 7, 31, 32, 63, 64, 100, u32::MAX];

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
        run!(math_i32, [0, 1, -1, 0x00f0_0000, i32::MIN, i32::MAX]);
        run!(math_u32, [0, 1, u32::MAX, 0x8000_0001, 12]);
        run!(math_i64, [0, 1, -1, 1 << 40, i64::MIN, i64::MAX]);
        run!(math_u64, [0, 1, u64::MAX, 1 << 63, 0x0f00]);
    }
    splice_testkit::each_backend!(B => check::<B>());
}

#[test]
fn integer_calls_record_typed_intrinsics() {
    let ir = math_i64::ir().unwrap();
    let ops: Vec<_> = splice_testkit::instructions(&ir)
        .filter_map(|inst| match inst {
            Instr::CallIntrinsic {
                intrinsic:
                    Intrinsic::Math {
                        op,
                        ty: ScalarType::I64,
                    },
                ..
            } => Some(*op),
            _ => None,
        })
        .collect();
    assert_eq!(
        ops,
        [
            MathOp::CountOnes,
            MathOp::LeadingZeros,
            MathOp::TrailingZeros,
            MathOp::RotateLeft
        ]
    );
}

#[test]
fn vector_types_are_markers_and_widths_are_kernel_declarations() {
    use splice::std::intrinsics::{Lane, Mask};
    assert_eq!(size_of::<Lane<f64>>(), 0);
    assert_eq!(size_of::<Mask<i32>>(), 0);
    assert!(std::panic::catch_unwind(Lane::<f64>::element_width).is_err());
    assert!(std::panic::catch_unwind(Lane::<i32>::byte_width).is_err());
}

#[kernel(print)]
fn integer_spellings(#[comptime] seed: u64, x: u64) -> u32 {
    seed.count_ones()
        + <u64>::count_ones(x)
        + u64::count_ones(x)
        + core::primitive::u64::leading_zeros(x)
}

#[test]
fn qualified_calls_and_known_receivers_preserve_integer_semantics() {
    let ir = integer_spellings::ir(15).unwrap();
    assert!(splice_testkit::instructions(&ir).all(|inst| !matches!(inst, Instr::RustCall { .. })));
    splice_testkit::each_backend!(B => {
        let code = B::compile(&ir).unwrap();
        assert_eq!(code.call(integer_spellings::CallArgs { x: 3 }), Ok(4 + 2 + 2 + 62));
    });
}

type V = splice::std::intrinsics::Lane<i32>;
type M = splice::std::intrinsics::Mask<i32>;

#[kernel]
fn masked_packet(src: &[i32], dst: &mut [i32], count: usize) {
    use core::ops::AddAssign;
    use splice::std::intrinsics::simd::Lane;
    let mut v: V = V::load(src, 0);
    v.add_assign(v);
    let mask: M = Lane::<i32>::first_n(count);
    let selected = v.add_masked(mask, Lane::splat(1), Lane::splat(-1));
    Lane::store(selected, dst, 0);
}

#[test]
fn masked_add_is_one_typed_call_and_stores_the_assigned_value() {
    use splice::hir::SimdOp;
    for options in modes() {
        let ir = masked_packet::ir_with_options(options).unwrap();
        let ops = splice_testkit::simd_ops(&ir);
        assert_eq!(ops.iter().filter(|i| i.op == SimdOp::AddMasked).count(), 1);
        assert!(
            splice_testkit::instructions(&ir).all(|inst| !matches!(inst, Instr::RustCall { .. }))
        );
        let width = options.target.vector_bytes() / 4;
        let src: Vec<_> = (3..3 + width as i32).collect();
        let expected: Vec<_> = (0..width as i32)
            .map(|i| if i == 0 { 7 } else { -1 })
            .collect();
        splice_testkit::each_backend!(B => {
            for code in [masked_packet::build_with_options::<B>(options, masked_packet::BuildArgs {}).unwrap(), B::compile(&ir).unwrap()] {
                let mut dst = vec![0; width];
                code.call(masked_packet::CallArgs { src: &src, dst: &mut dst, count: 1 }).unwrap();
                assert_eq!(dst, expected);
            }
        });
    }
}

mod names {
    use splice::std::intrinsics::simd::Lane as Packet;
    use splice::{Codegen, kernel, lancy::Lancy};
    type Fixed = Packet<i32>;
    use core::ops::Add;

    #[kernel]
    fn width_sum() -> usize {
        Packet::<f64>::byte_width() / 8
            + Fixed::element_width()
            + Packet::<f64>::element_width()
            + Packet::<i32>::element_width()
    }

    #[kernel]
    fn last_pair(src: &[i32]) -> i32 {
        let value = Fixed::load(src, 0);
        let doubled = <Packet<i32> as Add>::add(value, value);
        <Packet<i32> as core::ops::Add>::add(doubled, doubled).last()
    }

    #[kernel(print)]
    fn constant_count() -> u32 {
        u32::count_ones(15u32)
    }

    struct Lane;
    struct Meter;
    impl Meter {
        fn byte_width() -> i32 {
            -2
        }
    }
    impl Lane {
        fn element_width() -> usize {
            999
        }
        fn count_ones(x: usize, y: usize) -> usize {
            x + y
        }
    }
    #[kernel]
    fn ordinary(x: usize) -> usize {
        Lane::count_ones(x, Lane::element_width())
    }
    #[kernel]
    fn ordinary_width() -> i32 {
        Meter::byte_width().wrapping_add(1)
    }

    #[test]
    fn names_resolve_to_the_declaring_type() {
        let width_ir = width_sum::ir().unwrap();
        let pair_ir = last_pair::ir().unwrap();
        let count_ir = constant_count::ir().unwrap();
        for ir in [&*width_ir, &*pair_ir, &*count_ir] {
            assert!(
                splice_testkit::instructions(ir)
                    .all(|i| !matches!(i, splice::hir::Instr::RustCall { .. }))
            );
        }
        let bytes = width_ir.options().target.vector_bytes();
        assert_eq!(
            Lancy::compile(&width_ir)
                .unwrap()
                .call(width_sum::CallArgs {}),
            Ok(2 * (bytes / 8 + bytes / 4))
        );
        let src: Vec<_> = (1..=bytes as i32 / 4).collect();
        assert_eq!(
            Lancy::compile(&pair_ir)
                .unwrap()
                .call(last_pair::CallArgs { src: &src }),
            Ok(4 * src.last().unwrap())
        );
        assert_eq!(
            Lancy::compile(&count_ir)
                .unwrap()
                .call(constant_count::CallArgs {}),
            Ok(4)
        );
        assert!(constant_count::print().unwrap().contains("count_ones"));
        let ir = ordinary::ir().unwrap();
        assert_eq!(
            Lancy::compile(&ordinary_width::ir().unwrap())
                .unwrap()
                .call(ordinary_width::CallArgs {}),
            Ok(-1)
        );
        assert_eq!(
            splice_testkit::instructions(&ir)
                .filter(|i| matches!(i, splice::hir::Instr::RustCall { .. }))
                .count(),
            2
        );
        assert_eq!(
            Lancy::compile(&ir)
                .unwrap()
                .call(ordinary::CallArgs { x: 3 }),
            Ok(1002)
        );
    }
}

#[kernel]
mod vector_module {
    use splice::std::intrinsics::*;

    const OFFSET: usize = 0;
    fn double<T: LaneElement>(v: Lane<T>) -> Lane<T> {
        v + v
    }
    struct PacketOps;
    trait VectorCopy {
        fn copy_vector<T: LaneElement>(v: Lane<T>) -> Lane<T>;
    }
    impl VectorCopy for PacketOps {
        fn copy_vector<T: LaneElement>(v: Lane<T>) -> Lane<T> {
            v
        }
    }
    impl PacketOps {
        fn copy<T: LaneElement>(v: Lane<T>) -> Lane<T> {
            v
        }
        fn step(x: i32) -> i32 {
            x + 1
        }
    }
    pub fn sample(src: &[i32]) -> i32 {
        let pair: Lane<i32> = double::<i32>(Lane::load(src, OFFSET));
        let copy: Lane<i32> = PacketOps::copy::<i32>(pair);
        let value: Lane<i32> = <PacketOps as VectorCopy>::copy_vector::<i32>(copy);
        PacketOps::step(value.last())
    }
    #[cfg(any())]
    fn unavailable() -> String {
        String::new()
    }
    pub mod nested {
        pub fn width() -> usize {
            splice::std::intrinsics::Lane::<f64>::byte_width()
        }
    }
}

#[test]
fn module_imports_apply_to_signatures_impls_and_nested_kernels() {
    use splice::lancy::Lancy;
    let ir = vector_module::sample::ir().unwrap();
    assert!(splice_testkit::instructions(&ir).all(|inst| !matches!(inst, Instr::RustCall { .. })));
    let width = ir.options().target.vector_bytes() / size_of::<i32>();
    let src: Vec<_> = (1..=width as i32).collect();
    assert_eq!(
        Lancy::compile(&ir)
            .unwrap()
            .call(vector_module::sample::CallArgs { src: &src }),
        Ok(2 * width as i32 + 1)
    );
    let ir = vector_module::nested::width::ir().unwrap();
    assert_eq!(
        Lancy::compile(&ir)
            .unwrap()
            .call(vector_module::nested::width::CallArgs {}),
        Ok(ir.options().target.vector_bytes())
    );
}

#[kernel]
mod own_type {
    #[allow(unused_imports)]
    use splice::std::intrinsics::*;
    struct Lane;
    impl Lane {
        fn load(x: i32, y: i32) -> i32 {
            x + y
        }
    }
    pub fn sample(x: i32) -> i32 {
        Lane::load(x, 2)
    }
}

#[test]
fn a_module_type_can_shadow_a_glob_import() {
    let ir = own_type::sample::ir().unwrap();
    assert!(splice_testkit::simd_ops(&ir).is_empty());
    assert_eq!(
        splice::lancy::Lancy::compile(&ir)
            .unwrap()
            .call(own_type::sample::CallArgs { x: 3 }),
        Ok(5)
    );
}
