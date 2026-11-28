// Every signature diagnostic of #[kernel] points at the offending item.
use splice::kernel;
use splice::std::intrinsics::simd::Lane;
use splice::BufferArg;

#[kernel]
async fn asynchronous() {}

#[kernel]
fn wide_result() -> i128 {
    0
}

#[kernel]
fn unsupported_vector() -> Lane<char> {
    Lane::splat('a')
}

#[kernel]
fn mutable_slice(mut out: &[i32]) -> i32 {
    out[0]
}

#[kernel]
fn unknown_marker(#[inline] x: i32) -> i32 {
    x
}

#[kernel]
fn two_tables(
    #[columns(4)] a: &[BufferArg<'_>],
    #[columns(4)] b: &[BufferArg<'_>],
) -> i32 {
    0
}

#[kernel]
fn table_before_buffer(#[columns(4)] cols: &[BufferArg<'_>], xs: &[i32]) -> i32 {
    xs[0]
}

#[kernel]
fn table_type(#[columns(4)] cols: &[i32]) -> i32 {
    0
}

pub struct State;

#[kernel]
fn two_contexts(a: &mut State, b: &mut State) -> i32 {
    0
}

#[kernel]
fn context_pattern(ref a: &mut State) -> i32 {
    0
}

#[kernel]
fn context_vector(a: &mut State) -> Lane<i32> {
    Lane::splat(0)
}

#[kernel]
fn table_vector(#[columns(4)] cols: &[BufferArg<'_>]) -> Lane<i32> {
    Lane::splat(0)
}

#[kernel]
fn trait_object(#[comptime] f: &dyn Fn(i32) -> i32) -> i32 {
    0
}

#[kernel]
#[kernel]
fn duplicated() -> i32 {
    0
}

#[kernel]
#[comptime]
fn misplaced() -> i32 {
    0
}

#[kernel]
fn twice_comptime(#[comptime] #[comptime] n: i32, x: i32) -> i32 {
    n + x
}

#[derive(splice::KernelType)]
pub struct Pair {
    pub first: i32,
}

#[kernel]
fn missing_aggregate_field(x: i32) -> i32 {
    let pair = Pair { first: x };
    comptime!(pair.second)
}

fn main() {}

#[kernel]
fn bool_slice(flags: &[bool]) -> i32 {
    flags.len() as i32
}

#[kernel]
fn pointer_sized(x: isize) -> isize {
    x
}
