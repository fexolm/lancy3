// Diagnostics of #[kernel] on impls, traits, methods and derives.
use splice::{ComptimeKey, KernelType, kernel};

pub struct Nested;
#[kernel]
impl Nested {
    #[kernel]
    fn twice(x: i32) -> i32 {
        x * 2
    }
}

pub struct Wide;
#[kernel]
impl Wide {
    fn method_param(x: i128) -> i32 {
        x as i32
    }
}

pub struct Tuple;
#[kernel]
impl Tuple {
    fn tuple_param(#[comptime] (a, b): (i32, i32)) -> i32 {
        a + b
    }
}

pub struct State;
pub struct Contextual;
#[kernel]
impl Contextual {
    fn context_param(state: &mut State) -> i32 {
        0
    }
}

pub struct Asynchronous;
#[kernel]
impl Asynchronous {
    async fn later() -> i32 {
        0
    }
}

pub struct Plain;
impl Plain {
    #[kernel]
    fn get(&self) -> i32 {
        0
    }
}

pub trait PlainTrait {
    #[kernel]
    fn get(&self) -> i32;
}

#[kernel]
pub trait UnsupportedArgument {
    fn sum(x: &mut State) -> i32;
}

pub struct Marked;
#[kernel]
impl Marked {
    fn marked(#[inline] x: i32) -> i32 {
        x
    }
}

pub struct Doubly;
#[kernel]
impl Doubly {
    fn doubly(#[comptime] #[comptime] n: i32, x: i32) -> i32 {
        n + x
    }
}

pub struct Argument;
#[kernel]
impl Argument {
    fn with_argument(#[comptime(1)] n: i32, x: i32) -> i32 {
        n + x
    }
}

#[kernel]
pub trait Shape {
    fn area(x: u128) -> i32;
}

#[kernel]
struct NotAFunction;

#[derive(KernelType)]
enum NotAStruct {
    A,
}

#[derive(KernelType)]
#[splice(root = splice)]
struct UnknownOption {
    a: i32,
}

#[derive(ComptimeKey)]
#[splice(crate = splice, crate = splice)]
enum RepeatedRoot {
    A,
}

#[derive(ComptimeKey)]
union Bits {
    a: u32,
}

// MAC-23: each condition doubles the expanded variants.
#[splice::kernel]
fn many_conditions(
    #[cfg(test)] a: i32,
    #[cfg(test)] b: i32,
    #[cfg(test)] c: i32,
    #[cfg(test)] d: i32,
    #[cfg(test)] e: i32,
) -> i32 {
    0
}

fn main() {}
