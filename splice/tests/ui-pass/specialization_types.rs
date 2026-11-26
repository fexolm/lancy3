// The counterparts of `ui/specialization_types.rs` and `ui/comptime_drops.rs`
// that specialization runs without user code.
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

struct Guard {
    value: i32,
}

impl Drop for Guard {
    fn drop(&mut self) {}
}

// An `i32` kernel with a destructor runs as a Rust call.
#[kernel]
fn guarded(x: i32) -> i32 {
    let guard = Guard { value: x };
    guard.value
}

fn main() {
    assert!(described_field::ir(&Plan(1)).is_ok());
    assert!(primitive_operator::ir(1, 0).is_ok());
    assert!(guarded::ir().is_ok());
}
