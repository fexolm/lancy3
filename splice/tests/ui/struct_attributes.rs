// `#[kernel_type(...)]` options other than `transparent` and `pod` are not
// supported yet; a transparent kernel type has one field.
#[derive(Clone, Copy, splice::KernelType)]
#[kernel_type(soa)]
struct Columns {
    x: f32,
}

#[derive(Clone, Copy, splice::KernelType)]
#[kernel_type(align(16))]
struct Aligned {
    x: f32,
}

#[derive(Clone, Copy, splice::KernelType)]
#[kernel_type(transparent)]
struct Two(u32, u32);

#[derive(Clone, Copy, splice::KernelType)]
#[kernel_type(packed_simd)]
struct Unknown {
    x: f32,
}

#[derive(Clone, Copy, splice::KernelType)]
#[kernel_type(pod)]
struct Borrowed<'a> {
    x: &'a f32,
}

fn main() {}
