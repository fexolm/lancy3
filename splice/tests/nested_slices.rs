//! Slices of slices, `&[&[T]]` and `&[&str]`: each element is Rust's slice
//! reference, read whole as a slice (`xs[i]`, `for s in xs`), resliced,
//! split and passed to nested kernels. Every case runs on each backend,
//! staged directly and from portable IR, in release and debug, against the
//! kernel's own Rust function; an index outside traps where Rust panics,
//! after the writes before it.
#![allow(
    clippy::needless_range_loop,
    clippy::redundant_slicing,
    clippy::manual_range_contains
)]
use splice::{BuildOptions, CallError, Codegen, Executable, Mode, Trap, kernel};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[kernel]
fn total(rows: &[&[i32]]) -> i32 {
    let mut sum = 0_i32;
    for row in rows {
        for &x in row.iter() {
            sum = sum.wrapping_add(x);
        }
    }
    sum
}

/// `out[i]` is the sum of row `i`; `rows[k][j]` is read last, so a bad `k`
/// or `j` traps after every write.
#[kernel]
fn row_sums(rows: &[&[i32]], out: &mut [i32], k: i32, j: i32) -> i32 {
    for i in 0..rows.len() {
        let row = rows[i];
        let mut sum = 0_i32;
        for c in 0..row.len() {
            sum = sum.wrapping_add(row[c]);
        }
        out[i] = sum;
    }
    let tail = &rows[1..];
    let (head, rest) = rows.split_at(1);
    total(tail) * 1000
        + rows[k as usize][j as usize]
        + head.len() as i32 * 100
        + rest.len() as i32 * 10
}

#[kernel]
fn ascii_digits(text: &str) -> i32 {
    let mut count = 0_i32;
    for b in text.bytes() {
        if b >= b'0' && b <= b'9' {
            count += 1;
        }
    }
    count
}

/// Each element of a `&[&str]` is a `&str`: its slices check character
/// boundaries, and a nested kernel takes it as a `&str`.
#[kernel]
fn words(texts: &[&str], cut: i32) -> i32 {
    let mut score = 0_i32;
    for &text in texts {
        score += ascii_digits(text) * 100 + text.len() as i32;
    }
    let first = texts[0];
    let rest = &first[cut as usize..];
    score * 1000
        + rest.len() as i32 * 10
        + if rest.is_empty() {
            0
        } else {
            rest.as_bytes()[0] as i32 % 10
        }
}

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

fn rust<T>(f: impl FnOnce() -> T) -> Result<T, CallError> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|_| CallError::Trap(Trap::Bounds))
}

fn check<B: Codegen<Output: Executable>>() {
    let (a, b, c, empty) = ([1, 2, 3], [10, 20], [-7], []);
    let rows: [&[i32]; 4] = [&a, &b, &empty, &c];
    for code in builds!(total, B) {
        for n in 0..=rows.len() {
            let actual = code.call(total::CallArgs { rows: &rows[..n] });
            assert_eq!(actual, Ok(total(&rows[..n])), "{n} rows");
        }
    }
    for code in builds!(row_sums, B) {
        for n in 1..=rows.len() {
            for (k, j) in [(0, 0), (1, 1), (3, 0), (0, 3), (2, 0), (n as i32, 0)] {
                let (mut actual, mut expected) = ([0; 4], [0; 4]);
                let call = row_sums::CallArgs {
                    rows: &rows[..n],
                    out: &mut actual,
                    k,
                    j,
                };
                let result = code.call(call);
                let wanted = rust(|| row_sums(&rows[..n], &mut expected, k, j));
                assert_eq!((result, actual), (wanted, expected), "{n} rows at {k}, {j}");
            }
        }
    }
    let texts = ["h€llo 42", "", "7"];
    for code in builds!(words, B) {
        for cut in 0..=9 {
            let actual = code.call(words::CallArgs { texts: &texts, cut });
            assert_eq!(actual, rust(|| words(&texts, cut)), "cut {cut}");
        }
    }
}

#[test]
fn slices_of_slices_match_rust_on_every_backend() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let message = info.to_string();
        if !message.contains("byte index") && !message.contains("out of") {
            default(info);
        }
    }));
    splice_testkit::each_backend!(B => check::<B>());
}
