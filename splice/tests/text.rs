//! `&str` parameters: read-only UTF-8 bytes with `len`, `as_bytes`,
//! `bytes`, `is_char_boundary` and `&text[a..b]`, which traps where Rust
//! panics on a range outside the text or off a character boundary. Each
//! case runs on every backend, staged directly and from portable IR, in
//! release and debug, against the kernel's own Rust function.
// Kernels spell comparisons out: a range `contains` is not kernel code.
#![allow(clippy::manual_range_contains)]
use splice::{BuildOptions, CallError, Codegen, Executable, Mode, Trap, kernel};
use std::panic::{AssertUnwindSafe, catch_unwind};

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

#[kernel]
fn boundaries(text: &str) -> i32 {
    let mut count = 0_i32;
    for i in 0..text.len() + 2 {
        if text.is_char_boundary(i) {
            count += 1;
        }
    }
    count * 1000 + text.len() as i32 + if text.is_empty() { 500 } else { 0 }
}

#[kernel]
fn byte_sum(bytes: &[u8]) -> i32 {
    let mut sum = 0_i32;
    for &b in bytes {
        sum += b as i32;
    }
    sum
}

#[kernel]
fn middle(text: &str, start: i32, end: i32) -> i32 {
    let part = &text[start as usize..end as usize];
    let first = if part.is_empty() {
        0
    } else {
        part.as_bytes()[0] as i32
    };
    byte_sum(part.as_bytes()) * 1000 + part.len() as i32 * 10 + first % 10
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

const TEXTS: [&str; 4] = ["", "a1b22", "héllo 9€", "€€"];

fn check<B: Codegen<Output: Executable>>() {
    for code in builds!(ascii_digits, B) {
        for text in TEXTS {
            let actual = code.call(ascii_digits::CallArgs { text });
            assert_eq!(actual, Ok(ascii_digits(text)), "{text:?}");
        }
    }
    for code in builds!(boundaries, B) {
        for text in TEXTS {
            let actual = code.call(boundaries::CallArgs { text });
            assert_eq!(actual, Ok(boundaries(text)), "{text:?}");
        }
    }
    for code in builds!(middle, B) {
        for text in TEXTS {
            for start in 0..=text.len() as i32 + 1 {
                for end in [0, 1, 2, 3, 6, text.len() as i32, text.len() as i32 + 1] {
                    let actual = code.call(middle::CallArgs { text, start, end });
                    let expected = rust(|| middle(text, start, end));
                    assert_eq!(actual, expected, "{text:?}[{start}..{end}]");
                }
            }
        }
    }
}

#[test]
fn str_parameters_match_rust_on_every_backend() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let message = info.to_string();
        if !message.contains("byte index") && !message.contains("out of bounds") {
            default(info);
        }
    }));
    splice_testkit::each_backend!(B => check::<B>());
}
