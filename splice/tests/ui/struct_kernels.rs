// Struct values stay inside a kernel: they are not returned, passed to
// calls, updated with `..`, or used with SIMD operations yet.
use splice::kernel;
use splice::std::intrinsics::simd::Lane;

#[derive(Clone, Copy, splice::KernelType)]
struct Point {
    x: f32,
    y: f32,
}

#[kernel]
fn fixed(mut p: &Point) -> f32 {
    p.x
}

#[kernel]
fn returned(xs: &[Point]) -> Point {
    xs[0]
}

#[kernel]
fn x_of(p: Point) -> f32 {
    p.x
}

#[kernel]
fn passed(xs: &[Point]) -> f32 {
    x_of(xs[0])
}

#[kernel]
fn updated(xs: &mut [Point], p: Point) -> i32 {
    xs[0] = Point { x: 1.0, ..p };
    0
}

#[kernel]
fn vector(xs: &[Point]) -> i32 {
    let v = Lane::load(xs, 0);
    0
}

#[kernel]
fn vector_store(xs: &mut [Point], v: Lane<f32>) -> i32 {
    v.store(xs, 0);
    0
}

fn main() {}
