#[splice::intrinsic]
pub trait Bits {
    fn ones(self) -> u32 {
        0
    }
}

#[splice::intrinsic]
pub trait Shapes {
    fn generic<T>(self) -> u32;
    fn no_receiver() -> u32;
    fn mutable(&mut self) -> u32;
    fn no_result(self);
}

#[splice::intrinsic]
pub trait Plain {
    fn plain(self) -> u32;
}

#[splice::intrinsic]
impl Plain for u32 {
    fn plain(self) -> u32 {
        self
    }
}

#[splice::intrinsic]
impl<T> Plain for Vec<T> {
    fn plain(self) -> u32;
}

#[splice::intrinsic]
pub struct Fields {
    value: i32,
}

#[splice::intrinsic]
pub struct Bounded<T: Copy>;

#[splice::intrinsic]
pub fn free(x: i32) -> i32;

fn main() {}
