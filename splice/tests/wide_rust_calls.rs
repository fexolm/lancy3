//! Rust calls pass and return scalars wider than `i32` as 8-byte words on
//! every backend, through a staged build and through portable IR.
use splice::{Codegen, kernel};

/// A free Rust function: every operand type and a value outside `i32`.
fn mix(a: i64, b: u64, c: usize, x: f64, flag: bool, y: f32, z: u32) -> f64 {
    let ints = a as f64 + b as f64 - c as f64 + f64::from(z);
    if flag { ints * x + f64::from(y) } else { -ints }
}

fn sub_i64(a: i64, b: i64) -> i64 {
    a - b
}

fn sub_u64(a: u64, b: u64) -> u64 {
    a - b
}

fn sub_usize(a: usize, b: usize) -> usize {
    a - b
}

fn less(a: f64, b: f64) -> bool {
    a < b
}

#[kernel]
fn free_calls(a: i64, b: u64, c: usize, x: f64, flag: bool, y: f32, z: u32) -> f64 {
    let mixed = mix(a, b, c, x, flag, y, z);
    let wide = sub_i64(a, -a) as f64 + sub_u64(b, 1) as f64 + sub_usize(c, 1) as f64;
    if less(mixed, wide) {
        mixed - wide
    } else {
        wide - mixed
    }
}

#[kernel]
trait Scale {
    fn scale(value: u64, by: i64) -> i64;
}

/// Its trait method is a Rust call from the kernel.
struct Plain;

impl Scale for Plain {
    fn scale(value: u64, by: i64) -> i64 {
        value as i64 * by
    }
}

/// Its trait method is staged; the body calls Rust again.
struct Published;

#[kernel]
impl Scale for Published {
    fn scale(value: u64, by: i64) -> i64 {
        sub_i64(value as i64 * by, 1)
    }
}

#[kernel]
fn scaled<T: Scale>(value: u64, by: i64) -> i64 {
    <T as Scale>::scale(value, by)
}

struct Ledger {
    total: i64,
}

impl Ledger {
    fn add(&mut self, amount: i64, count: usize) -> i64 {
        self.total += amount * count as i64;
        self.total
    }

    fn even(&self) -> bool {
        self.total % 2 == 0
    }
}

#[kernel]
fn ledger(book: &mut Ledger, amount: i64, count: usize) -> i64 {
    let total = book.add(amount, count);
    if book.even() { total } else { -total }
}

#[test]
fn wide_operands_and_results_cross_every_rust_call() -> Result<(), Box<dyn std::error::Error>> {
    let big = 3_i64 << 40;
    let free = free_calls::CallArgs {
        a: big,
        b: u64::MAX - 7,
        c: 1 << 35,
        x: -0.5,
        flag: true,
        y: 1.25,
        z: u32::MAX,
    };
    let expected = free_calls(free.a, free.b, free.c, free.x, free.flag, free.y, free.z);
    let unflagged = free_calls(free.a, free.b, free.c, free.x, false, free.y, free.z);
    let (value, by) = (1_u64 << 33, -3_i64);
    splice_testkit::each_backend!(B => {
        for code in [
            free_calls::build::<B>(free_calls::BuildArgs {})?,
            B::compile(&free_calls::ir()?)?,
        ] {
            assert_eq!(code.call(free_calls::CallArgs { ..free })?, expected);
            assert_eq!(code.call(free_calls::CallArgs { flag: false, ..free })?, unflagged);
        }
        for code in [
            scaled::build::<B, Plain>(scaled::BuildArgs {})?,
            B::compile(&scaled::ir::<Plain>()?)?,
        ] {
            assert_eq!(code.call(scaled::CallArgs { value, by })?, scaled::<Plain>(value, by));
        }
        for code in [
            scaled::build::<B, Published>(scaled::BuildArgs {})?,
            B::compile(&scaled::ir::<Published>()?)?,
        ] {
            assert_eq!(
                code.call(scaled::CallArgs { value, by })?,
                scaled::<Published>(value, by)
            );
        }
        for code in [
            ledger::build::<B>(ledger::BuildArgs {})?,
            B::compile(&ledger::ir()?)?,
        ] {
            let mut book = Ledger { total: 1 };
            let mut rust = Ledger { total: 1 };
            for (amount, count) in [(big, 3), (5, 1 << 33)] {
                assert_eq!(
                    code.call(ledger::CallArgs { book: &mut book, amount, count })?,
                    ledger(&mut rust, amount, count)
                );
            }
            assert_eq!(book.total, rust.total);
        }
    });
    Ok(())
}
