// Specialization runs no user code: a type that would need it is a
// compile error at the kernel, not a build error.
use splice::kernel;

struct Opaque(i32);

#[kernel]
fn undescribed_field(#[comptime] plan: Opaque, x: i32) -> i32 {
    x + plan.0
}

#[derive(splice::KernelType)]
struct Steps {
    first: i32,
}

struct Indirect(Steps);

impl std::ops::Deref for Indirect {
    type Target = Steps;
    fn deref(&self) -> &Steps {
        &self.0
    }
}

#[kernel]
fn user_deref(#[comptime] plan: &Indirect) -> i32 {
    let steps = &**plan;
    steps.first
}

#[derive(Clone, Copy, PartialEq)]
struct Meters(i32);

impl std::ops::Add for Meters {
    type Output = Meters;
    fn add(self, other: Meters) -> Meters {
        Meters(self.0 + other.0)
    }
}

#[kernel]
fn user_operator(#[comptime] a: Meters, #[comptime] b: Meters, x: i32) -> i32 {
    if a + b == a { x } else { 0 }
}

#[kernel]
fn user_operator_in_comptime(#[comptime] a: Meters) -> i32 {
    comptime!((a + a).0)
}

fn main() {}
