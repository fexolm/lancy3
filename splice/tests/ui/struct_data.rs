// Kernel data are kernel scalars and `Copy` structs of kernel data:
// references, `Option`, `Vec` and non-`Copy` structs are not supported yet.
use splice::kernel;

#[derive(Clone, Copy, splice::KernelType)]
struct WithReference {
    x: &'static i32,
}

#[derive(splice::KernelType)]
struct NotCopy {
    x: i32,
}

#[derive(Clone, Copy, splice::KernelType)]
struct WithOption {
    x: Option<u8>,
}

#[derive(splice::KernelType)]
#[kernel_type(pod)]
struct Owned {
    values: Vec<i32>,
}

#[kernel]
fn reference(p: WithReference) -> i32 {
    0
}

#[kernel]
fn not_copy(xs: &[NotCopy]) -> i32 {
    0
}

#[kernel]
fn option(xs: &mut [WithOption], y: Option<i32>) -> i32 {
    0
}

fn main() {}
