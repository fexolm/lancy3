// A value with drop glue that specialization would construct and drop is a
// compile error when the kernel is instantiated (a post-monomorphization
// error, so `cargo check` does not report it).
use splice::kernel;

const NAME: String = String::new();

#[kernel]
fn comptime_string(x: i32) -> i32 {
    x + comptime!(NAME.len() as i32)
}

struct Probe(i32);

impl Drop for Probe {
    fn drop(&mut self) {}
}

const PROBE: Probe = Probe(10);

#[kernel]
fn comptime_local(x: i32) -> i32 {
    comptime!({
        let Probe(value) = PROBE;
        value + x
    })
}

struct Guard {
    value: i32,
}

impl Drop for Guard {
    fn drop(&mut self) {}
}

// A `usize` result cannot travel through a Rust call, so the destructor
// of `guard` could not run.
#[kernel]
fn guarded_index(x: i32) -> usize {
    let guard = Guard { value: x };
    guard.value as usize
}

fn main() {
    let _ = comptime_string::ir();
    let _ = comptime_local::ir();
    let _ = guarded_index::ir();
}
