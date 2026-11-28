use splice::kernel;

trait Extra { fn extra(self) -> u32; }
impl Extra for u32 {
    splice::std::intrinsics::intrinsic!(fn extra(self) -> u32);
}

#[kernel]
fn unknown(x: u32) -> u32 { x.extra() }

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

use splice::std::intrinsics::Math as bits;

#[kernel]
fn ambiguous_math(h: &mut Hook, x: u32) -> usize { h.record(bits::count_ones(x) as usize) }

fn main() {}
