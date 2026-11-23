// Diagnostics of kernel bodies: host expressions, macros and SIMD typing.
use splice::kernel;
use splice::simd::Lane;

#[kernel]
fn runtime_tuple(x: i32) -> i32 {
    let pair = (x, 1);
    pair.1
}

#[kernel]
fn nested_macro() -> i32 {
    let v = vec![format!("a")];
    v.len() as i32
}

#[kernel]
fn untyped_splat(out: &mut [i32]) -> i32 {
    let v = Lane::splat(1.5);
    v.store(out, 0);
    0
}

fn main() {}
