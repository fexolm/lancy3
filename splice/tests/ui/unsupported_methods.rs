// MAC-08: every method of a #[kernel] block lowers; a Rust method belongs
// in an ordinary impl.
use splice::kernel;

pub struct Foo {
    pub n: i32,
}

#[kernel]
impl Foo {
    fn name(&self) -> String {
        format!("{}", self.n)
    }

    fn wide(x: isize) -> i32 {
        x as i32
    }
}

#[kernel]
trait Rule {
    type Output;
    fn widen(value: i32) -> Self::Output;
}

fn main() {}
