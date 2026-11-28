//! `i8`, `u8`, `i16`, `u16` and `char` kernels follow Rust on every backend,
//! both build routes and both build modes: arithmetic wraps, saturates or
//! traps as the Rust function does, `as` converts like Rust, and narrow
//! SIMD lanes compute each lane as the scalar operation.
use splice::std::intrinsics::simd::Lane;
use splice::{BuildOptions, CallError, Codegen, Executable, Mode, Trap, kernel};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[derive(Clone, Copy, Debug)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    LeftShift,
    RightShift,
    WrappingAdd,
    WrappingMultiply,
    WrappingNegate,
    SaturatingAdd,
    SaturatingSubtract,
    SaturatingMultiply,
    Less,
    Minimum,
    ThroughI64,
    ThroughF32,
    ThroughF64,
    FromU8,
}

fn options() -> [BuildOptions; 2] {
    [
        BuildOptions::host(Mode::Release).unwrap(),
        BuildOptions::host(Mode::Debug).unwrap(),
    ]
}

/// The kernels and the check of one integer type.
macro_rules! integer_type {
    ($($module:ident: $ty:ident),*) => {$(
        // The check uses the build routes it needs, not every generated one.
        #[allow(dead_code)]
        mod $module {
            use super::*;

            #[kernel]
            pub fn arithmetic(#[comptime] op: Operation, out: &mut [$ty], a: $ty, b: $ty) -> $ty {
                out[0] = 7;
                let result = match op {
                    Operation::Add => a + b,
                    Operation::Subtract => a - b,
                    Operation::Multiply => a * b,
                    Operation::Divide => a / b,
                    Operation::Remainder => a % b,
                    Operation::LeftShift => a << b,
                    Operation::RightShift => a >> b,
                    Operation::WrappingAdd => a.wrapping_add(b),
                    Operation::WrappingMultiply => a.wrapping_mul(b),
                    Operation::WrappingNegate => a.wrapping_neg(),
                    Operation::SaturatingAdd => a.saturating_add(b),
                    Operation::SaturatingSubtract => a.saturating_sub(b),
                    Operation::SaturatingMultiply => a.saturating_mul(b),
                    Operation::Less => (a < b) as $ty,
                    Operation::Minimum => {
                        if a <= b { a } else { b }
                    }
                    Operation::ThroughI64 => (a as i64 * 3 + b as i64) as $ty,
                    Operation::ThroughF32 => (a as f32 * 1.5 - b as f32) as $ty,
                    Operation::ThroughF64 => (a as f64 * -300.25) as $ty,
                    Operation::FromU8 => (b as u8) as $ty,
                };
                out[1] = result;
                result
            }

            pub fn check<B: Codegen<Output: Executable>>(options: BuildOptions) {
                use Operation::*;
                let values = [
                    0,
                    1,
                    2,
                    7,
                    $ty::BITS as $ty - 1,
                    $ty::BITS as $ty,
                    $ty::MIN,
                    $ty::MAX,
                    $ty::MAX / 2 + 1,
                    (0 as $ty).wrapping_sub(1),
                ];
                for op in [
                    Add, Subtract, Multiply, Divide, Remainder, LeftShift, RightShift,
                    WrappingAdd, WrappingMultiply, WrappingNegate, SaturatingAdd,
                    SaturatingSubtract, SaturatingMultiply, Less, Minimum, ThroughI64,
                    ThroughF32, ThroughF64, FromU8,
                ] {
                    let direct = arithmetic::build_with_options::<B>(options, arithmetic::BuildArgs { op })
                        .expect("build arithmetic");
                    let portable = B::compile(&arithmetic::ir_with_options(options, op).unwrap())
                        .expect("compile arithmetic IR");
                    for a in values {
                        for b in values {
                            let mut expected_out = [0; 2];
                            let expected = catch_unwind(AssertUnwindSafe(|| {
                                arithmetic(op, &mut expected_out, a, b)
                            }))
                            .map_err(|_| {
                                let division = matches!(op, Divide | Remainder);
                                CallError::Trap(match (division, b) {
                                    (true, 0) => Trap::DivisionByZero,
                                    (true, _) => Trap::DivisionOverflow,
                                    (false, _) => Trap::Overflow,
                                })
                            });
                            for code in [&direct, &portable] {
                                let mut out = [0; 2];
                                let actual = code.call(arithmetic::CallArgs { out: &mut out, a, b });
                                let at = format!("{} {op:?} {a}, {b} ({options:?})", stringify!($ty));
                                assert_eq!(actual, expected, "{at}");
                                assert_eq!(out, expected_out, "{at}");
                            }
                        }
                    }
                }
            }
        }
    )*};
}
integer_type!(signed_byte: i8, byte: u8, signed_word: i16, word: u16);

#[test]
fn narrow_integers_compute_like_rust_in_both_paths_and_modes() {
    for options in options() {
        splice_testkit::each_backend!(B => {
            signed_byte::check::<B>(options);
            byte::check::<B>(options);
            signed_word::check::<B>(options);
            word::check::<B>(options);
        });
    }
}

/// Count the letters of `text` that are `needle` or lowercase ASCII, and
/// write the last one's code point plus `shift` into `codes`.
// The kernel compares `char`s; it does not lower range patterns.
#[allow(clippy::manual_range_contains)]
#[kernel]
fn letters(text: &[char], needle: char, shift: u8, codes: &mut [u32]) -> usize {
    let mut count: usize = 0;
    let mut i = 0;
    while i < text.len() {
        let c = text[i];
        if c == needle || (c >= 'a' && c <= 'z') {
            count += 1;
            codes[0] = c as u32 + (shift as char) as u32;
        }
        i += 1;
    }
    count
}

/// The larger of two characters, as `char` and through `u8` and `i64`.
#[kernel]
fn widest(out: &mut [char], a: char, b: char) -> char {
    out[0] = if a > b { a } else { b };
    out[1] = ((a as u8) as char).max(b);
    out[2] = (b as i64 as u8) as char;
    out[0]
}

#[test]
fn chars_compare_and_convert_like_rust_in_both_paths_and_modes() {
    let texts: [&[char]; 3] = [&['a', 'Z', 'é', '😀', 'q'], &[], &['\0', char::MAX, 'z']];
    for options in options() {
        splice_testkit::each_backend!(B => {
            let direct = letters::build_with_options::<B>(options, letters::BuildArgs {}).unwrap();
            let portable = B::compile(&letters::ir_with_options(options).unwrap()).unwrap();
            for text in texts {
                for (needle, shift) in [('é', 1), ('😀', 200), (char::MAX, 0)] {
                    let mut want = [0];
                    let expected = letters(text, needle, shift, &mut want);
                    for code in [&direct, &portable] {
                        let mut codes = [0];
                        let seen = code.call(letters::CallArgs { text, needle, shift, codes: &mut codes });
                        assert_eq!(seen, Ok(expected), "{text:?} {needle:?}");
                        assert_eq!(codes, want, "{text:?} {needle:?}");
                    }
                }
            }
            let direct = widest::build_with_options::<B>(options, widest::BuildArgs {}).unwrap();
            let portable = B::compile(&widest::ir_with_options(options).unwrap()).unwrap();
            for (a, b) in [('a', 'b'), ('é', 'A'), ('😀', char::MAX), ('\0', 'ÿ')] {
                let mut want = ['?'; 3];
                let expected = widest(&mut want, a, b);
                for code in [&direct, &portable] {
                    let mut out = ['?'; 3];
                    assert_eq!(code.call(widest::CallArgs { out: &mut out, a, b }), Ok(expected));
                    assert_eq!(out, want, "{a:?} {b:?}");
                }
            }
        });
    }
}

/// Brighten `src` into `dst` by `amount`, saturating, a vector at a time;
/// return the wrapping sum, the maximum and the count of bright pixels of
/// the first vector.
#[kernel]
fn brighten(src: &[u8], dst: &mut [u8], amount: u8, stats: &mut [u8]) {
    let n = Lane::<u8>::element_width();
    let add = Lane::splat(amount);
    let bright = Lane::<u8>::splat(200);
    let mut i = 0;
    while i + n <= src.len() {
        let v = Lane::<u8>::load(src, i).saturating_add(add);
        v.store(dst, i);
        if i == 0 {
            stats[0] = v.reduce_sum();
            stats[1] = v.simd_max(bright).extract_wrapping(3);
            stats[2] = v.simd_gt(bright).count() as u8;
            stats[3] = v.simd_lt(bright).select(v, bright).last();
        }
        i += n;
    }
    while i < src.len() {
        dst[i] = src[i].saturating_add(amount);
        i += 1;
    }
}

/// The same with signed 16-bit lanes and saturating subtraction.
#[kernel]
fn darken(src: &[i16], dst: &mut [i16], amount: i16) -> i16 {
    let n = Lane::<i16>::element_width();
    let sub = Lane::splat(amount);
    let mut i = 0;
    let mut low = Lane::<i16>::splat(i16::MAX);
    while i + n <= src.len() {
        let v = Lane::<i16>::load(src, i).saturating_sub(sub);
        v.store(dst, i);
        low = low.simd_min(v);
        i += n;
    }
    while i < src.len() {
        dst[i] = src[i].saturating_sub(amount);
        i += 1;
    }
    low.replace_wrapping(1, i16::MIN + 1)
        .simd_min(low)
        .extract_wrapping(1)
}

/// The reference of `brighten` with vectors of `n` lanes, lane by lane.
fn brighten_reference(n: usize, src: &[u8], dst: &mut [u8], amount: u8, stats: &mut [u8]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = s.saturating_add(amount);
    }
    if src.len() >= n {
        let v = &dst[..n];
        stats[0] = v.iter().fold(0u8, |sum, &x| sum.wrapping_add(x));
        stats[1] = v[3 % n].max(200);
        stats[2] = v.iter().filter(|&&x| x > 200).count() as u8;
        stats[3] = if v[n - 1] < 200 { v[n - 1] } else { 200 };
    }
}

/// The reference of `darken` with vectors of `n` lanes, lane by lane.
fn darken_reference(n: usize, src: &[i16], dst: &mut [i16], amount: i16) -> i16 {
    for (d, s) in dst.iter_mut().zip(src) {
        *d = s.saturating_sub(amount);
    }
    let mut low = vec![i16::MAX; n];
    for block in dst.chunks_exact(n) {
        for (l, &x) in low.iter_mut().zip(block) {
            *l = (*l).min(x);
        }
    }
    (i16::MIN + 1).min(low[1 % n])
}

#[test]
fn narrow_lanes_compute_each_lane_like_rust_in_both_paths_and_modes() {
    let bytes: Vec<u8> = (0..300u32).map(|i| (i * 37 % 256) as u8).collect();
    let words: Vec<i16> = (0..300i32)
        .map(|i| (i * 7919 % 65536 - 32768) as i16)
        .collect();
    let (bytes_n, words_n) = (
        (splice::Target::host().unwrap().vector_bytes() / size_of::<u8>()),
        (splice::Target::host().unwrap().vector_bytes() / size_of::<i16>()),
    );
    for options in options() {
        splice_testkit::each_backend!(B => {
            let direct = brighten::build_with_options::<B>(options, brighten::BuildArgs {}).unwrap();
            let portable = B::compile(&brighten::ir_with_options(options).unwrap()).unwrap();
            for len in [0, 1, 63, 64, 65, 300] {
                for amount in [0, 1, 100, 255] {
                    let (mut want, mut want_stats) = (vec![0; len], [0; 4]);
                    brighten_reference(bytes_n, &bytes[..len], &mut want, amount, &mut want_stats);
                    for code in [&direct, &portable] {
                        let (mut dst, mut stats) = (vec![0; len], [0; 4]);
                        let args = brighten::CallArgs { src: &bytes[..len], dst: &mut dst, amount, stats: &mut stats };
                        assert_eq!(code.call(args), Ok(()));
                        assert_eq!(dst, want, "{len} {amount}");
                        assert_eq!(stats, want_stats, "{len} {amount}");
                    }
                }
            }
            let direct = darken::build_with_options::<B>(options, darken::BuildArgs {}).unwrap();
            let portable = B::compile(&darken::ir_with_options(options).unwrap()).unwrap();
            for len in [0, 31, 32, 33, 300] {
                for amount in [0, 1, i16::MAX, i16::MIN, -5] {
                    let mut want = vec![0; len];
                    let expected = darken_reference(words_n, &words[..len], &mut want, amount);
                    for code in [&direct, &portable] {
                        let mut dst = vec![0; len];
                        let args = darken::CallArgs { src: &words[..len], dst: &mut dst, amount };
                        assert_eq!(code.call(args), Ok(expected));
                        assert_eq!(dst, want, "{len} {amount}");
                    }
                }
            }
        });
    }
}

/// A specialization on narrow types and char, rendered as Rust.
#[kernel(print)]
fn tally(#[comptime] limit: u8, text: &[char], bytes: &[u8], small: i8, half: i16) -> u16 {
    let mut count: u16 = 0;
    for &c in text {
        if c == 'é' || c > 'z' {
            count += 1;
        }
    }
    let n = Lane::<u8>::element_width();
    if bytes.len() >= n {
        let top = Lane::<u8>::load(bytes, 0).simd_min(Lane::splat(limit));
        count = count.wrapping_add(top.reduce_sum() as u16);
    }
    count.wrapping_add((small as i16 + half) as u16)
}

#[test]
fn narrow_kernels_print_as_rust_and_run_their_specialization() {
    let printed = tally::print(200).unwrap();
    for part in [
        "'é'",
        "'z'",
        "char",
        "u8",
        "i8",
        "i16",
        "u16",
        "Lane :: < u8 >",
        "splat(200)",
        "reduce_sum",
    ] {
        assert!(printed.contains(part), "{part}: {printed}");
    }
    let text = ['a', 'é', '{', 'z'];
    let bytes: Vec<u8> = (0..128u32).map(|i| (i * 53 % 256) as u8).collect();
    let n = splice::Target::host().unwrap().vector_bytes() / size_of::<u8>();
    let sum = bytes[..n]
        .iter()
        .fold(0u8, |s, &b| s.wrapping_add(b.min(200)));
    let expected = 2u16
        .wrapping_add(u16::from(sum))
        .wrapping_add((-3i16 + 1000) as u16);
    for options in options() {
        splice_testkit::each_backend!(B => {
            let direct = tally::build_with_options::<B>(options, tally::BuildArgs { limit: 200 }).unwrap();
            let portable = B::compile(&tally::ir_with_options(options, 200).unwrap()).unwrap();
            for code in [&direct, &portable] {
                let args = tally::CallArgs { text: &text, bytes: &bytes, small: -3, half: 1000 };
                assert_eq!(code.call(args), Ok(expected));
            }
        });
    }
}
