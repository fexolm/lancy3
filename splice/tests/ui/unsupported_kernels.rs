// MAC-01: a body Splice cannot lower is a compile error, not a Rust fallback.
use splice::kernel;

#[kernel]
fn wide(x: i128) -> i32 {
    x as i32
}

// A function value is Rust data, not a kernel the frontend can resolve.
#[kernel]
fn use_callback(#[comptime] callback: fn(i32) -> i32, x: i32) -> i32 {
    callback(x)
}

// The listing shows an unrolled loop as its iterations.
#[kernel(print)]
fn printed_steps(#[comptime] plan: &[i32], xs: &[i32]) -> i32 {
    let mut hits = 0;
    'steps: for &step in plan {
        for &x in xs {
            if x == step {
                hits += 1;
                continue 'steps;
            }
        }
    }
    hits
}

// `char::from_u32` returns an `Option`, which a kernel register cannot hold.
#[kernel]
fn decode(code: u32) -> char {
    char::from_u32(code).unwrap_or('?')
}

fn main() {}
