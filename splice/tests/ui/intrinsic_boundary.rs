use splice::kernel;

#[splice::intrinsic]
pub struct Atomic<T>;

#[splice::intrinsic]
impl<T> Atomic<T> {
    pub fn load(&self) -> T;
}

#[kernel]
fn by_reference(#[comptime] _atomic: &Atomic<i32>, x: i32) -> i32 {
    x
}

#[kernel]
fn by_value(#[comptime] _atomic: Atomic<u64>) -> i32 {
    0
}

fn main() {}
