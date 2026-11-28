// A slice of slices is shared at both levels: `&[&[T]]` or `&[&str]`.
use splice::kernel;

#[kernel]
fn inner(rows: &[&mut [i32]]) -> i32 {
    0
}

#[kernel]
fn outer(rows: &mut [&[i32]]) -> i32 {
    0
}

fn main() {}
