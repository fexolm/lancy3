// MAC-01: a body Splice cannot lower is a compile error, not a Rust fallback.
use splice::kernel;

pub struct Plan {
    pub min: i32,
}

#[kernel]
fn labeled(#[comptime] plan: &Plan, x: i32) -> i32 {
    let mut n = 0;
    'outer: loop {
        n += 1;
        if n > 3 {
            break 'outer;
        }
    }
    plan.min + x + n
}

#[kernel]
fn wide(x: i64) -> i32 {
    x as i32
}

// A function value is Rust data, not a kernel the frontend can resolve.
#[kernel]
fn use_callback(#[comptime] callback: fn(i32) -> i32, x: i32) -> i32 {
    callback(x)
}

fn main() {}
