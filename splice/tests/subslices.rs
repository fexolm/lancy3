//! Subslices as values: `&xs[a..b]` and its open forms, `split_at`,
//! subslices passed to nested kernels and read by SIMD lanes. Every case
//! runs on each backend, staged directly and from portable IR, in release
//! and debug, against the kernel's own Rust function; a range outside the
//! slice traps where Rust panics, after the writes before it.
// The kernels index and reslice on purpose: those are the operations tested.
#![allow(clippy::needless_range_loop, clippy::redundant_slicing)]
use splice::std::intrinsics::simd::Lane;
use splice::{BuildOptions, CallError, Codegen, Executable, Mode, Trap, kernel};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[kernel]
fn window_sum(xs: &[i32], start: i32, end: i32) -> i32 {
    let window = &xs[start as usize..end as usize];
    let mut sum = 0_i32;
    for &x in window {
        sum = sum.wrapping_add(x);
    }
    sum
}

/// The `i32` loop over a subslice whose checks Lancy drops
/// (`lancy/filetests/passes/subslice-bounds.lancy`), next to a loop to the
/// whole length that indexes the shorter subslice and must trap.
#[kernel]
fn window_scan(xs: &[i32], start: i32, end: i32, whole: bool) -> i32 {
    let window = &xs[start as usize..end as usize];
    let (head, tail) = window.split_at(window.len() / 2);
    let mut sum = 0_i32;
    for i in 0..head.len() as i32 {
        sum = sum.wrapping_add(head[i as usize]);
    }
    for i in 0..tail.len() as i32 {
        sum = sum.wrapping_mul(3).wrapping_add(tail[i as usize]);
    }
    if whole {
        for i in 0..xs.len() as i32 {
            sum = sum.wrapping_add(window[i as usize]);
        }
    }
    sum
}

#[kernel]
fn open_ranges(xs: &[i32], k: i32) -> i32 {
    let tail = &xs[k as usize..];
    let head = &xs[..k as usize];
    let closed = &xs[..=k as usize];
    let all = &xs[..];
    let empty = if tail.is_empty() { 1_000_000 } else { 0 };
    let first = if head.is_empty() { 0 } else { head[0] };
    (tail.len() as i32) * 10_000
        + (head.len() as i32) * 100
        + closed.len() as i32
        + all.len() as i32 * 1_000
        + empty
        + first
}

#[kernel]
fn fill_window(out: &mut [i32], start: i32, end: i32, value: i32) -> i32 {
    out[0] = -1;
    let window = &mut out[start as usize..end as usize];
    for i in 0..window.len() {
        window[i] = value;
    }
    window.len() as i32
}

#[kernel]
fn add_one(xs: &mut [i32]) -> i32 {
    for i in 0..xs.len() {
        xs[i] = xs[i].wrapping_add(1);
    }
    xs.len() as i32
}

#[kernel]
fn total(xs: &[i32]) -> i32 {
    let mut sum = 0_i32;
    for &x in xs {
        sum = sum.wrapping_add(x);
    }
    sum
}

#[kernel]
fn nested_windows(xs: &[i32], out: &mut [i32], start: i32) -> i32 {
    let bumped = add_one(&mut out[start as usize..]);
    let inner = &xs[start as usize..];
    total(&inner[1..]).wrapping_mul(100).wrapping_add(bumped)
}

#[kernel]
fn split_halves(xs: &mut [i32], mid: i32) -> i32 {
    let (head, tail) = xs.split_at_mut(mid as usize);
    for i in 0..tail.len() {
        tail[i] = tail[i].wrapping_mul(2);
    }
    let (left, right) = head.split_at(head.len() / 2);
    (left.len() as i32) * 100 + (right.len() as i32) * 10 + tail.len() as i32
}

/// The case of `docs/splice-api-cases.md`: at `k == len` the first write
/// lands and the second traps.
#[kernel]
fn write_both(xs: &mut [i32], k: i32) -> i32 {
    let (l, r) = xs.split_at_mut(k as usize);
    l[0] = 1;
    r[0] = 2;
    0
}

#[kernel]
fn shifted_lanes(xs: &[i32], out: &mut [i32]) -> i32 {
    let shifted = &xs[1..];
    let v = Lane::load(shifted, 0) + Lane::splat(1);
    v.store(&mut out[1..], 0);
    Lane::<i32>::element_width() as i32
}

/// Every built form of a kernel: backend, path and mode.
macro_rules! builds {
    ($kernel:ident, $b:ident) => {{
        let mut codes = Vec::new();
        for mode in [Mode::Release, Mode::Debug] {
            let options = BuildOptions::host(mode).unwrap();
            codes.push($kernel::build_with_options::<$b>(options, $kernel::BuildArgs {}).unwrap());
            codes.push($b::compile(&$kernel::ir_with_options(options).unwrap()).unwrap());
        }
        codes
    }};
}

/// The Rust result, or the trap that stands for its panic.
fn rust<T>(f: impl FnOnce() -> T) -> Result<T, CallError> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|_| CallError::Trap(Trap::Bounds))
}

fn check<B: Codegen<Output: Executable>>() {
    let xs: Vec<i32> = (1..=9).collect();
    for code in builds!(window_sum, B) {
        for (start, end) in [
            (0, 9),
            (2, 5),
            (4, 4),
            (9, 9),
            (5, 4),
            (0, 10),
            (10, 10),
            (-1, 2),
        ] {
            let actual = code.call(window_sum::CallArgs {
                xs: &xs,
                start,
                end,
            });
            let expected = rust(|| window_sum(&xs, start, end));
            assert_eq!(actual, expected, "window_sum({start}, {end})");
        }
    }
    for code in builds!(window_scan, B) {
        for (start, end, whole) in [
            (0, 9, false),
            (2, 7, false),
            (3, 3, false),
            (0, 9, true),
            (2, 7, true),
            (6, 5, false),
        ] {
            let actual = code.call(window_scan::CallArgs {
                xs: &xs,
                start,
                end,
                whole,
            });
            let expected = rust(|| window_scan(&xs, start, end, whole));
            assert_eq!(actual, expected, "window_scan({start}, {end}, {whole})");
        }
    }
    for code in builds!(open_ranges, B) {
        for k in [0, 3, 8, 9] {
            let actual = code.call(open_ranges::CallArgs { xs: &xs, k });
            assert_eq!(actual, rust(|| open_ranges(&xs, k)), "open_ranges({k})");
        }
    }
    for code in builds!(fill_window, B) {
        for (start, end) in [(1, 4), (0, 0), (2, 9), (3, 2), (5, 10)] {
            let (mut actual, mut expected) = (xs.clone(), xs.clone());
            let result = code.call(fill_window::CallArgs {
                out: &mut actual,
                start,
                end,
                value: 7,
            });
            let reference = rust(|| fill_window(&mut expected, start, end, 7));
            assert_eq!(result, reference, "fill_window({start}, {end})");
            assert_eq!(actual, expected, "writes before the trap stay");
        }
    }
    for code in builds!(nested_windows, B) {
        for start in [0, 4, 8, 9, 10] {
            let (mut actual, mut expected) = (xs.clone(), xs.clone());
            let result = code.call(nested_windows::CallArgs {
                xs: &xs,
                out: &mut actual,
                start,
            });
            let reference = rust(|| nested_windows(&xs, &mut expected, start));
            assert_eq!(result, reference, "nested_windows({start})");
            assert_eq!(actual, expected);
        }
    }
    for code in builds!(split_halves, B) {
        for mid in [0, 5, 9, 10] {
            let (mut actual, mut expected) = (xs.clone(), xs.clone());
            let result = code.call(split_halves::CallArgs {
                xs: &mut actual,
                mid,
            });
            assert_eq!(result, rust(|| split_halves(&mut expected, mid)));
            assert_eq!(actual, expected);
        }
    }
    for code in builds!(write_both, B) {
        for k in [1, 5, 9, 10] {
            let (mut actual, mut expected) = (xs.clone(), xs.clone());
            let result = code.call(write_both::CallArgs { xs: &mut actual, k });
            assert_eq!(
                result,
                rust(|| write_both(&mut expected, k)),
                "write_both({k})"
            );
            assert_eq!(actual, expected);
        }
    }
    let lanes = splice::Target::host().unwrap().vector_bytes() / size_of::<i32>();
    for code in builds!(shifted_lanes, B) {
        for len in [lanes, lanes + 1, lanes + 3] {
            let input: Vec<i32> = (0..len as i32).collect();
            let (mut actual, mut expected) = (vec![0; len], vec![0; len]);
            let result = code.call(shifted_lanes::CallArgs {
                xs: &input,
                out: &mut actual,
            });
            // Lanes run only in a kernel: the load of `lanes` elements
            // from the second traps before anything is stored.
            let reference = if len > lanes {
                for i in 1..=lanes {
                    expected[i] = input[i] + 1;
                }
                Ok(lanes as i32)
            } else {
                Err(CallError::Trap(Trap::Bounds))
            };
            assert_eq!(result, reference, "shifted_lanes over {len} elements");
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn subslices_match_rust_on_every_backend() {
    // The Rust references panic on the ranges that trap.
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let message = info.to_string();
        if !message.contains("index") && !message.contains("mid > len") {
            default(info);
        }
    }));
    splice_testkit::each_backend!(B => check::<B>());
}
