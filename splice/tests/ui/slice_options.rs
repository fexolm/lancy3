// Slice methods that return an Option wait for kernel Option support;
// `split_at` binds its halves with a tuple pattern.
use splice::kernel;

#[kernel]
fn first(xs: &[i32]) -> i32 {
    let head = xs.first();
    0
}

#[kernel]
fn get(xs: &[i32], i: i32) -> i32 {
    let x = xs.get(i as usize);
    0
}

#[kernel]
fn split(xs: &[i32]) -> i32 {
    let halves = xs.split_at(1);
    0
}

#[kernel]
fn mutable(xs: &[i32]) -> i32 {
    let tail = &mut xs[1..];
    0
}

fn main() {}
