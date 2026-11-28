//! Iteration over runtime slices: `for x in xs` by value and
//! `chunks_exact`. Every case runs on each backend, staged directly and from
//! portable IR, in release and debug, against the kernel's own Rust function;
//! a zero chunk size traps where Rust panics, after the writes before it.
// The kernels index chunks on purpose: writing through an index is tested.
#![allow(clippy::needless_range_loop)]
use splice::std::intrinsics::simd::Lane;
use splice::{BuildOptions, CallError, Codegen, Executable, Mode, Trap, kernel};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[kernel]
fn by_value(xs: &[i32], floor: i32) -> i32 {
    let mut sum = 0_i32;
    for x in xs {
        if *x > floor {
            sum = sum.wrapping_add(x * 2);
        }
    }
    for x in xs.iter() {
        sum = sum.wrapping_sub(*x);
    }
    for x in &xs[1..] {
        sum = sum.wrapping_add(*x);
    }
    sum
}

#[kernel]
fn chunk_sums(xs: &[i32], out: &mut [i32], n: i32) -> i32 {
    out[0] = -1;
    let mut count = 0_i32;
    for chunk in xs.chunks_exact(n as usize) {
        let mut sum = 0_i32;
        for &x in chunk {
            sum = sum.wrapping_add(x);
        }
        out[count as usize] = sum.wrapping_add(chunk.len() as i32 * 1000);
        count += 1;
    }
    let rest = xs.chunks_exact(n as usize).remainder();
    let mut tail = 0_i32;
    for x in rest {
        tail = tail.wrapping_mul(10).wrapping_add(*x);
    }
    count * 100_000 + tail
}

#[kernel]
fn scale_chunks(xs: &mut [i32], n: i32) -> i32 {
    let mut blocks = 0_i32;
    for chunk in xs.chunks_exact_mut(n as usize) {
        for i in 0..chunk.len() {
            chunk[i] = chunk[i].wrapping_mul(blocks + 2);
        }
        blocks += 1;
    }
    let rest = xs.chunks_exact_mut(n as usize).into_remainder();
    for i in 0..rest.len() {
        rest[i] = -rest[i];
    }
    blocks
}

#[kernel]
fn lane_chunks(xs: &[i32], out: &mut [i32]) -> i32 {
    let width = Lane::<i32>::element_width();
    let mut done = 0_usize;
    for chunk in xs.chunks_exact(width) {
        let v = Lane::load(chunk, 0) + Lane::splat(1);
        v.store(&mut out[done..], 0);
        done += width;
    }
    xs.chunks_exact(width).remainder().len() as i32
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

/// The Rust result, or the trap that stands for its panic: a zero chunk
/// size divides by zero.
fn rust<T>(f: impl FnOnce() -> T) -> Result<T, CallError> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|_| CallError::Trap(Trap::DivisionByZero))
}

fn check<B: Codegen<Output: Executable>>() {
    for len in [0, 1, 7, 9] {
        let xs: Vec<i32> = (1..=len).collect();
        for code in builds!(by_value, B) {
            for floor in [0, 4] {
                let actual = code.call(by_value::CallArgs { xs: &xs, floor });
                let expected = if xs.is_empty() {
                    // `&xs[1..]` of an empty slice.
                    Err(CallError::Trap(Trap::Bounds))
                } else {
                    Ok(by_value(&xs, floor))
                };
                assert_eq!(actual, expected, "by_value over {len}, floor {floor}");
            }
        }
        for code in builds!(chunk_sums, B) {
            for n in [0, 1, 2, 3, 4, 10] {
                let (mut actual, mut expected) = (vec![0; 10], vec![0; 10]);
                let result = code.call(chunk_sums::CallArgs {
                    xs: &xs,
                    out: &mut actual,
                    n,
                });
                let reference = rust(|| chunk_sums(&xs, &mut expected, n));
                assert_eq!(result, reference, "chunk_sums over {len} by {n}");
                assert_eq!(actual, expected, "writes before the trap stay");
            }
        }
        for code in builds!(scale_chunks, B) {
            for n in [0, 1, 2, 4, 10] {
                let (mut actual, mut expected) = (xs.clone(), xs.clone());
                let result = code.call(scale_chunks::CallArgs { xs: &mut actual, n });
                let reference = rust(|| scale_chunks(&mut expected, n));
                assert_eq!(result, reference, "scale_chunks over {len} by {n}");
                assert_eq!(actual, expected);
            }
        }
    }
    let lanes = splice::Target::host().unwrap().vector_bytes() / size_of::<i32>();
    for code in builds!(lane_chunks, B) {
        for len in [0, lanes - 1, lanes, 3 * lanes + 2] {
            let input: Vec<i32> = (0..len as i32).collect();
            let mut actual = vec![0; len];
            let result = code.call(lane_chunks::CallArgs {
                xs: &input,
                out: &mut actual,
            });
            let whole = len - len % lanes;
            let expected: Vec<i32> = (0..len as i32)
                .map(|x| if (x as usize) < whole { x + 1 } else { 0 })
                .collect();
            assert_eq!(result, Ok((len % lanes) as i32), "lane_chunks over {len}");
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn slice_iteration_matches_rust_on_every_backend() {
    // The Rust references panic on a zero chunk size.
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !info.to_string().contains("chunk size must be non-zero") {
            default(info);
        }
    }));
    splice_testkit::each_backend!(B => check::<B>());
}
