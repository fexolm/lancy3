use splice::{CallError, Codegen, Executable, hir::Instr, kernel};
use splice_lancy::Lancy;
use splice_llvm::Llvm;
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

fn checked(value: i32) -> Result<(), &'static str> {
    (value >= 0).then_some(()).ok_or("negative")
}

#[kernel]
fn write_around_result(out: &mut [i32], value: i32) -> Result<i32, &'static str> {
    out[0] = 7;
    checked(value)?;
    out[1] = 8;
    Ok(value)
}

#[kernel]
fn catches_slice_panic(xs: &[u8], index: i32) -> bool {
    std::panic::catch_unwind(|| xs[index as usize]).is_err()
}

struct Matcher {
    value: i32,
    calls: Cell<usize>,
}

#[kernel]
impl Matcher {
    fn matches(&self, #[comptime] value: i32) -> bool {
        self.calls.set(self.calls.get() + 1);
        self.value == value
    }
}

#[kernel]
fn guarded_match(matcher: &Matcher, matches: bool, value: i32) -> bool {
    if matches {
        matcher.matches(value)
    } else {
        false
    }
}

#[kernel]
fn published(#[comptime] value: i32) -> i32 {
    value.wrapping_add(17)
}

#[kernel]
fn runtime_step(value: i32) -> i32 {
    value.wrapping_mul(2)
}

#[kernel]
fn published_in_region(value: i32) -> Result<i32, &'static str> {
    Ok(published(value).wrapping_add(line!() as i32))
}

#[kernel]
fn matched_stages_in_region(
    #[comptime] planned: bool,
    #[comptime] offset: i32,
    value: i32,
) -> Result<i32, &'static str> {
    checked(value)?;
    Ok(if planned {
        runtime_step(value).wrapping_add(published(offset))
    } else {
        published(value)
    })
}

#[test]
fn rust_regions_keep_result_control_flow_local_panics_and_borrowed_methods() {
    fn check<B: Codegen<Output: Executable>>() {
        let ir = write_around_result::ir().unwrap();
        for code in [
            write_around_result::build::<B>(write_around_result::BuildArgs {}).unwrap(),
            B::compile(&ir).unwrap(),
        ] {
            let mut out = [0, 0];
            assert_eq!(
                code.call(write_around_result::CallArgs {
                    out: &mut out,
                    value: -1
                })
                .unwrap(),
                Err("negative")
            );
            assert_eq!(out, [7, 0]);
            assert_eq!(
                code.call(write_around_result::CallArgs {
                    out: &mut out,
                    value: 2
                })
                .unwrap(),
                Ok(2)
            );
            assert_eq!(out, [7, 8]);

            let mut short = [0];
            assert!(
                code.call(write_around_result::CallArgs {
                    out: &mut short,
                    value: 2
                })
                .is_err()
            );
            assert_eq!(short, [7]);
        }

        let ir = catches_slice_panic::ir().unwrap();
        for code in [
            catches_slice_panic::build::<B>(catches_slice_panic::BuildArgs {}).unwrap(),
            B::compile(&ir).unwrap(),
        ] {
            for (index, caught) in [(0, false), (1, true)] {
                assert_eq!(
                    code.call(catches_slice_panic::CallArgs { xs: &[9], index })
                        .unwrap(),
                    caught
                );
            }
        }

        let matcher = Matcher {
            value: 11,
            calls: Cell::new(0),
        };
        let ir = guarded_match::ir().unwrap();
        let direct = guarded_match::build::<B>(guarded_match::BuildArgs {}).unwrap();
        let portable = B::compile(&ir).unwrap();
        assert_eq!(matcher.calls.get(), 0);
        for code in [direct, portable] {
            let before = matcher.calls.get();
            assert!(
                !code
                    .call(guarded_match::CallArgs {
                        matcher: &matcher,
                        matches: false,
                        value: 11
                    })
                    .unwrap()
            );
            assert_eq!(matcher.calls.get(), before);
            assert!(
                code.call(guarded_match::CallArgs {
                    matcher: &matcher,
                    matches: true,
                    value: 11
                })
                .unwrap()
            );
            assert_eq!(matcher.calls.get(), before + 1);
        }

        assert!(published_in_region::ir().is_err());
        assert!(published_in_region::build::<B>(published_in_region::BuildArgs {}).is_err());
        assert!(matched_stages_in_region::ir(false, 3).is_err());
        let ir = matched_stages_in_region::ir(true, 3).unwrap();
        for code in [
            matched_stages_in_region::build::<B>(matched_stages_in_region::BuildArgs {
                planned: true,
                offset: 3,
            })
            .unwrap(),
            B::compile(&ir).unwrap(),
        ] {
            assert_eq!(
                code.call(matched_stages_in_region::CallArgs { value: 11 })
                    .unwrap(),
                Ok(42)
            );
        }
    }

    check::<Lancy>();
    check::<Llvm>();
}

static RUNTIME_DROPS: AtomicUsize = AtomicUsize::new(0);

struct RuntimeGuard {
    offset: i32,
}

impl Drop for RuntimeGuard {
    fn drop(&mut self) {
        RUNTIME_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

#[kernel]
fn runtime_drop(value: i32) -> i32 {
    if value > 0 {
        let guard = RuntimeGuard { offset: 10 };
        guard.offset.wrapping_add(self::runtime_step(value))
    } else {
        0
    }
}

#[test]
fn runtime_aggregate_drops_once_per_call() {
    fn check<B: Codegen<Output: Executable>>() {
        let before = RUNTIME_DROPS.load(Ordering::Relaxed);
        let ir = runtime_drop::ir().unwrap();
        let direct = runtime_drop::build::<B>(runtime_drop::BuildArgs {}).unwrap();
        let portable = B::compile(&ir).unwrap();
        assert_eq!(RUNTIME_DROPS.load(Ordering::Relaxed), before);
        for (index, code) in [direct, portable].into_iter().enumerate() {
            assert_eq!(code.call(runtime_drop::CallArgs { value: 0 }).unwrap(), 0);
            assert_eq!(RUNTIME_DROPS.load(Ordering::Relaxed), before + index);
            assert_eq!(code.call(runtime_drop::CallArgs { value: 3 }).unwrap(), 16);
            assert_eq!(RUNTIME_DROPS.load(Ordering::Relaxed), before + index + 1);
        }
    }

    check::<Lancy>();
    check::<Llvm>();
}

static SLICE_DROPS: AtomicUsize = AtomicUsize::new(0);

struct SliceGuard {
    value: i32,
}

impl Drop for SliceGuard {
    fn drop(&mut self) {
        SLICE_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

#[kernel]
fn slice_drop(
    #[comptime] require_known: bool,
    output: &mut [i32],
    input: &[i32],
    value: i32,
    run: bool,
) -> i32 {
    if run {
        let guard = SliceGuard { value };
        output[0] = guard.value;
        let result = if require_known {
            published(value)
        } else {
            guard.value.wrapping_add(input[0])
        };
        output[1] = result;
        result
    } else {
        0
    }
}

#[kernel]
fn nested_slice_drop(input: &[i32], output: &mut [i32], value: i32, run: bool) -> i32 {
    let result = slice_drop(false, &mut *output, input, value, run);
    if run {
        output[1] = result.wrapping_add(1);
    }
    result
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
fn slice_regions_preserve_drop_order_and_native_aggregates() {
    fn exercise(call: impl Fn(&mut [i32], &[i32], bool) -> Result<i32, CallError>, stored: i32) {
        let before = SLICE_DROPS.load(Ordering::Relaxed);
        let mut output = [-8, -9];
        assert_eq!(call(&mut output, &[], false).unwrap(), 0);
        assert_eq!(output, [-8, -9]);
        assert_eq!(SLICE_DROPS.load(Ordering::Relaxed), before);

        assert!(call(&mut output, &[], true).is_err());
        assert_eq!(output, [12, -9]);
        assert_eq!(SLICE_DROPS.load(Ordering::Relaxed), before + 1);

        output = [-8, -9];
        assert_eq!(call(&mut output, &[7], true).unwrap(), 19);
        assert_eq!(output, [12, stored]);
        assert_eq!(SLICE_DROPS.load(Ordering::Relaxed), before + 2);
    }

    fn check<B: Codegen<Output: Executable>>() {
        let before = SLICE_DROPS.load(Ordering::Relaxed);
        assert!(slice_drop::ir(true).is_err());
        assert!(
            slice_drop::build::<B>(slice_drop::BuildArgs {
                require_known: true
            })
            .is_err()
        );
        let ir = slice_drop::ir(false).unwrap();
        let nested_ir = nested_slice_drop::ir().unwrap();
        let direct = slice_drop::build::<B>(slice_drop::BuildArgs {
            require_known: false,
        })
        .unwrap();
        let nested = nested_slice_drop::build::<B>(nested_slice_drop::BuildArgs {}).unwrap();
        let portable = B::compile(&ir).unwrap();
        let nested_portable = B::compile(&nested_ir).unwrap();
        assert_eq!(SLICE_DROPS.load(Ordering::Relaxed), before);

        for code in [direct, portable] {
            exercise(
                |output, input, run| {
                    code.call(slice_drop::CallArgs {
                        output,
                        input,
                        value: 12,
                        run,
                    })
                },
                19,
            );
        }
        for code in [nested, nested_portable] {
            exercise(
                |output, input, run| {
                    code.call(nested_slice_drop::CallArgs {
                        input,
                        output,
                        value: 12,
                        run,
                    })
                },
                20,
            );
        }

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

    check::<Lancy>();
    check::<Llvm>();
}

struct Weight(i32);

#[kernel]
fn render_captures(
    #[comptime] weight: Weight,
    #[comptime] number: i32,
    #[comptime] _unused: &Weight,
    value: Option<i32>,
) -> String {
    let explicit = weight.0;
    let closure = [(2, 3)]
        .into_iter()
        .map(|(weight, _unused)| weight + _unused)
        .sum::<i32>();
    let selected = match value {
        Some(weight) if weight > 1 => weight,
        _ => 0,
    };
    let conditional = if let Some(weight) = value
        && weight > 1
    {
        weight
    } else {
        weight.0
    };
    let mut total = 0;
    for weight in [weight.0, 4] {
        total += weight;
    }
    format!("{number}:{explicit}:{closure}:{selected}:{conditional}:{total}")
}

#[test]
fn rust_regions_capture_only_lexically_free_build_arguments() {
    fn check<B: Codegen<Output: Executable>>() {
        let (direct, ir) = {
            let unused = Weight(41);
            let direct = render_captures::build::<B>(render_captures::BuildArgs {
                weight: Weight(7),
                number: 11,
                _unused: &unused,
            })
            .unwrap();
            let ir = render_captures::ir(Weight(7), 11, &unused).unwrap();
            (direct, ir)
        };
        for code in [direct, B::compile(&ir).unwrap()] {
            for value in [None, Some(8)] {
                assert_eq!(
                    code.call(render_captures::CallArgs { value }).unwrap(),
                    render_captures(Weight(7), 11, &Weight(41), value)
                );
            }
        }
    }

    check::<Lancy>();
    check::<Llvm>();
}

trait PlainWideRule {
    fn widen(value: i32) -> u64;
}

#[kernel]
trait PublishedWideRule {
    type Output;
    fn widen(value: i32) -> Self::Output;
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
    type Output = u64;

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
    type Output = u64;

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
fn narrow_published<T: PublishedWideRule<Output = u64>>(value: i32) -> i32 {
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
) -> u64 {
    if planned {
        <T as PublishedWideRule>::widen_known(known)
    } else {
        <T as PublishedWideRule>::widen_known(value)
    }
}

#[kernel]
fn narrow_pure<T: PublishedWideRule>(#[comptime] value: i32) -> i32 {
    comptime!(<T as PublishedWideRule>::pure(value) as i32)
}

#[test]
fn trait_result_casts_run_in_rust_and_preserve_staging() {
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
            assert_eq!(
                code.call(wide_result::CallArgs { value: 23 }).unwrap(),
                0x0000_0100_ffff_ffef_u64,
            );
            assert_eq!(WIDE_CALLS.load(Ordering::Relaxed), before + index + 1);
        }
    }

    check::<Lancy>();
    check::<Llvm>();
}
