use splice::kernel;

#[kernel(fast)]
fn unknown(x: i32) -> i32 {
    x
}

#[kernel(print, print)]
fn repeated(x: i32) -> i32 {
    x
}

// Splice has no Rust kernels: an ordinary Rust function is called as one.
#[kernel(rust)]
fn rust_kernel(x: i32) -> i32 {
    x
}

pub struct Unit;

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
