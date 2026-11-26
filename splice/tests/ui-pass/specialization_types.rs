// The counterparts of `ui/specialization_types.rs` that specialization runs
// without user code.
use splice::kernel;

#[derive(splice::KernelType)]
struct Plan(i32);

#[kernel]
fn described_field(#[comptime] plan: &Plan, x: i32) -> i32 {
    let plan = &*plan;
    x + plan.0
}

#[kernel]
fn primitive_operator(#[comptime] a: i32, #[comptime] b: i32, x: i32) -> i32 {
    if a + b == a { x } else { 0 }
}

fn main() {
    assert!(described_field::ir(&Plan(1)).is_ok());
    assert!(primitive_operator::ir(1, 0).is_ok());
}
