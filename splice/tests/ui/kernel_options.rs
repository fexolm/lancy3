use splice::kernel;

#[kernel(fast)]
fn unknown(x: i32) -> i32 {
    x
}

#[kernel(rust, rust)]
fn repeated(x: i32) -> i32 {
    x
}

#[kernel(rust)]
impl Unit {
    fn one() -> i32 {
        1
    }
}

pub struct Unit;

// A Rust kernel keeps its specialization arguments, so they cannot borrow.
#[kernel(rust)]
fn borrowed(#[comptime] plan: &[i32], x: i32) -> i32 {
    plan[0] + x
}

fn main() {}
