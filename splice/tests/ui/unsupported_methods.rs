// MAC-08: every method of a #[kernel] block is lowered or marked rust.
use splice::kernel;

pub struct Foo {
    pub n: i32,
}

#[kernel]
impl Foo {
    fn name(&self) -> String {
        format!("{}", self.n)
    }

    fn wide(x: i64) -> i32 {
        x as i32
    }

    #[kernel(rust)]
    fn label(&self) -> String {
        format!("{}", self.n)
    }
}

#[kernel]
trait Rule {
    type Output;
    fn widen(value: i32) -> Self::Output;
}

fn main() {}
