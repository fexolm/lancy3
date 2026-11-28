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

#[kernel]
fn chars(text: &str) -> i32 {
    let mut n = 0;
    for c in text.chars() {
        n += 1;
    }
    n
}

#[kernel]
fn index_str(text: &str) -> i32 {
    let first = text.as_bytes()[0];
    let other = text.find('a');
    0
}

#[kernel]
fn bound_chunks(xs: &[i32]) -> i32 {
    let chunks = xs.chunks_exact(2);
    0
}

#[kernel]
fn shared_chunks(xs: &[i32]) -> i32 {
    for chunk in xs.chunks_exact_mut(2) {}
    0
}

#[kernel]
fn mutable_elements(xs: &mut [i32]) -> i32 {
    for x in xs {
        *x += 1;
    }
    0
}

fn main() {}
