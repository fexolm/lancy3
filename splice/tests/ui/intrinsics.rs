use splice::kernel;

#[allow(non_camel_case_types)]
type block = splice::std::intrinsics::Lane<i32>;
struct Hook;
impl Hook {
    fn record(&mut self, x: usize) -> usize { x }
}

#[kernel]
fn ambiguous_width(h: &mut Hook) -> usize { h.record(block::byte_width()) }

#[kernel]
fn ambiguous_vector(h: &mut Hook) -> usize { h.record(block::splat(3).last() as usize) }

fn main() {}
