// Layouts a kernel cannot address yet: a packed struct, a transparent
// newtype over a struct, and a slice of bools.
use splice::kernel;

#[derive(Clone, Copy, splice::KernelType)]
#[repr(C, packed)]
struct Packed {
    a: u8,
    b: u32,
}

#[derive(Clone, Copy, splice::KernelType)]
struct Inner {
    x: f32,
}

#[derive(Clone, Copy, splice::KernelType)]
#[repr(transparent)]
struct Wrapper(Inner);

#[derive(Clone, Copy, splice::KernelType)]
#[repr(transparent)]
struct Flag(bool);

#[kernel]
fn packed(xs: &[Packed]) -> i32 {
    0
}

#[kernel]
fn wrapper(w: Wrapper) -> i32 {
    0
}

#[kernel]
fn flags(xs: &[Flag]) -> i32 {
    0
}

fn main() {
    let _ = packed::build::<splice::lancy::Lancy>(packed::BuildArgs {});
    let _ = wrapper::build::<splice::lancy::Lancy>(wrapper::BuildArgs {});
    let _ = flags::build::<splice::lancy::Lancy>(flags::BuildArgs {});
}
