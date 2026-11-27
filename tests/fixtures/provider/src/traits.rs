use splice::kernel;

#[kernel]
pub trait Transform {
    fn shift(value: i32, #[comptime] offset: i32) -> i32;
}

pub trait OrdinaryTransform<Tag = ()> {
    fn ordinary(value: i32) -> i32;
}

#[kernel]
pub trait ReverseTransform: Transform {
    fn shift(value: i32, #[comptime] offset: i32) -> i32;
}

#[kernel]
pub trait IndexedTransform {
    fn adjust(value: i32, #[comptime] offset: usize) -> i32;
}

#[kernel]
pub trait PlanTransform {
    fn shift(#[comptime] &self, value: i32) -> i32;
}

#[kernel]
pub trait OtherPlanTransform {
    fn shift(#[comptime] &self, value: i32) -> i32;
}

#[kernel]
pub trait OwnedPlanTransform {
    fn shift(#[comptime] self, value: i32) -> i32;
}

#[kernel]
pub trait ParameterizedTransform<Tag: Copy> {
    fn shift(value: i32, #[comptime] offset: i32) -> i32;
}

#[kernel]
pub trait WideTransform {
    fn combine(value: i32, #[comptime] offset: i32, subtract: bool) -> i32;
}
