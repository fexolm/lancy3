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

// `print` renders Splice source, which a Rust kernel does not have.
#[kernel(rust, print)]
fn printed_rust(x: i32) -> i32 {
    x
}

#[kernel(print)]
impl Unit {
    fn two() -> i32 {
        2
    }
}

// Without `print`, a kernel publishes no `print`.
#[kernel]
fn unprinted(x: i32) -> i32 {
    x
}

fn main() {
    let _ = unprinted::print();
}
