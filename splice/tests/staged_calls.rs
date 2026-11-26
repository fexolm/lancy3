use splice::{Codegen, Executable, hir::Instr, kernel};
use std::sync::atomic::{AtomicUsize, Ordering};

#[kernel]
fn published(#[comptime] value: i32) -> i32 {
    value.wrapping_add(17)
}

#[kernel]
fn runtime_step(value: i32) -> i32 {
    value.wrapping_mul(2)
}

#[kernel]
fn matched_stages(#[comptime] planned: bool, #[comptime] offset: i32, value: i32) -> i32 {
    if planned {
        runtime_step(value).wrapping_add(published(offset))
    } else {
        published(value)
    }
}

#[test]
fn comptime_parameters_reject_runtime_arguments_at_build() {
    fn check<B: Codegen<Output: Executable>>() {
        assert!(matched_stages::ir(false, 3).is_err());
        assert!(
            matched_stages::build::<B>(matched_stages::BuildArgs {
                planned: false,
                offset: 3,
            })
            .is_err()
        );
        let ir = matched_stages::ir(true, 3).unwrap();
        for code in [
            matched_stages::build::<B>(matched_stages::BuildArgs {
                planned: true,
                offset: 3,
            })
            .unwrap(),
            B::compile(&ir).unwrap(),
        ] {
            assert_eq!(
                code.call(matched_stages::CallArgs { value: 11 }).unwrap(),
                42
            );
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

struct SlicePair {
    known: i32,
    value: i32,
}

#[kernel]
fn slice_aggregate(#[comptime] known: i32, output: &mut [i32], value: i32) -> i32 {
    let pair = SlicePair { known, value };
    let result = pair.known.wrapping_add(pair.value);
    output[0] = result;
    result
}

#[test]
fn slice_kernels_keep_native_aggregates() {
    fn check<B: Codegen<Output: Executable>>() {
        let ir = slice_aggregate::ir(5).unwrap();
        assert!(
            ir.instructions()
                .iter()
                .any(|op| matches!(op, Instr::Store { .. }))
        );
        let direct = slice_aggregate::build::<B>(slice_aggregate::BuildArgs { known: 5 }).unwrap();
        for code in [direct, B::compile(&ir).unwrap()] {
            let mut output = [0];
            assert_eq!(
                code.call(slice_aggregate::CallArgs {
                    output: &mut output,
                    value: 12
                })
                .unwrap(),
                17
            );
            assert_eq!(output, [17]);
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

trait PlainWideRule {
    fn widen(value: i32) -> u64;
}

#[kernel]
trait PublishedWideRule {
    fn widen(value: i32) -> u64;
    fn widen_known(#[comptime] value: i32) -> u64;
    fn pure(value: i32) -> u64;
}

static WIDE_CALLS: AtomicUsize = AtomicUsize::new(0);

fn record_wide(value: i32) -> u64 {
    WIDE_CALLS.fetch_add(1, Ordering::Relaxed);
    (1_u64 << 40) | u64::from(value as u32)
}

struct WideOwner;

impl PlainWideRule for WideOwner {
    fn widen(value: i32) -> u64 {
        record_wide(value)
    }
}

impl PublishedWideRule for WideOwner {
    fn widen(value: i32) -> u64 {
        record_wide(value.wrapping_mul(3))
    }

    fn widen_known(value: i32) -> u64 {
        record_wide(value)
    }

    fn pure(value: i32) -> u64 {
        record_wide(value)
    }
}

struct PublishedWideOwner;

#[kernel]
impl PublishedWideRule for PublishedWideOwner {
    fn widen(value: i32) -> u64 {
        record_wide(value.wrapping_add(7))
    }

    fn widen_known(#[comptime] value: i32) -> u64 {
        record_wide(value)
    }

    fn pure(value: i32) -> u64 {
        (1_u64 << 40) | ((value + 11) as u32 as u64)
    }
}

#[kernel]
fn narrow_plain<T: PlainWideRule>(value: i32) -> i32 {
    <T as PlainWideRule>::widen(value) as i32
}

#[kernel]
fn narrow_published<T: PublishedWideRule>(value: i32) -> i32 {
    <T as PublishedWideRule>::widen(value) as i32
}

#[kernel]
fn narrow_requires_known<T: PublishedWideRule>(value: i32) -> i32 {
    <T as PublishedWideRule>::widen_known(value) as i32
}

#[kernel]
pub(crate) fn wide_result<T: PublishedWideRule>(
    #[comptime] planned: bool,
    #[comptime] known: i32,
    value: i32,
) -> i32 {
    if planned {
        <T as PublishedWideRule>::widen_known(known) as i32
    } else {
        <T as PublishedWideRule>::widen_known(value) as i32
    }
}

#[kernel]
fn narrow_pure<T: PublishedWideRule>(#[comptime] value: i32) -> i32 {
    comptime!(<T as PublishedWideRule>::pure(value) as i32)
}

#[test]
fn trait_result_casts_preserve_staging() {
    fn check<B: Codegen<Output: Executable>>() {
        macro_rules! check_calls {
            ($entry:ident, $owner:ty, $expected:expr) => {{
                let expected = $entry::<$owner>(-17);
                assert_eq!(expected, $expected);
                let before = WIDE_CALLS.load(Ordering::Relaxed);
                let ir = $entry::ir::<$owner>().unwrap();
                let direct = $entry::build::<B, $owner>($entry::BuildArgs {}).unwrap();
                let portable = B::compile(&ir).unwrap();
                assert_eq!(WIDE_CALLS.load(Ordering::Relaxed), before);
                for (index, code) in [direct, portable].into_iter().enumerate() {
                    assert_eq!(
                        code.call($entry::CallArgs { value: -17 }).unwrap(),
                        expected
                    );
                    assert_eq!(WIDE_CALLS.load(Ordering::Relaxed), before + index + 1);
                }
            }};
        }

        check_calls!(narrow_plain, WideOwner, -17);
        check_calls!(narrow_published, WideOwner, -51);
        check_calls!(narrow_published, PublishedWideOwner, -10);

        check_required_argument::<B, WideOwner>();
        check_required_argument::<B, PublishedWideOwner>();

        let before = WIDE_CALLS.load(Ordering::Relaxed);
        assert!(narrow_pure::ir::<WideOwner>(-17).is_err());
        assert!(narrow_pure::build::<B, WideOwner>(narrow_pure::BuildArgs { value: -17 }).is_err());
        let ir = narrow_pure::ir::<PublishedWideOwner>(-17).unwrap();
        let direct =
            narrow_pure::build::<B, PublishedWideOwner>(narrow_pure::BuildArgs { value: -17 })
                .unwrap();
        for code in [direct, B::compile(&ir).unwrap()] {
            assert_eq!(code.call(narrow_pure::CallArgs {}).unwrap(), -6);
        }
        assert_eq!(WIDE_CALLS.load(Ordering::Relaxed), before);
    }

    fn check_required_argument<B: Codegen<Output: Executable>, T: PublishedWideRule>() {
        let before = WIDE_CALLS.load(Ordering::Relaxed);
        assert!(narrow_requires_known::ir::<T>().is_err());
        assert!(narrow_requires_known::build::<B, T>(narrow_requires_known::BuildArgs {}).is_err());
        let args = |planned| wide_result::BuildArgs {
            planned,
            known: -17,
        };
        assert!(wide_result::ir::<T>(false, -17).is_err());
        assert!(wide_result::build::<B, T>(args(false)).is_err());
        let ir = wide_result::ir::<T>(true, -17).unwrap();
        let direct = wide_result::build::<B, T>(args(true)).unwrap();
        let portable = B::compile(&ir).unwrap();
        assert_eq!(WIDE_CALLS.load(Ordering::Relaxed), before);
        for (index, code) in [direct, portable].into_iter().enumerate() {
            assert_eq!(code.call(wide_result::CallArgs { value: 23 }).unwrap(), -17,);
            assert_eq!(WIDE_CALLS.load(Ordering::Relaxed), before + index + 1);
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}
