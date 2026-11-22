use splice::kernel;

pub trait Transform {
    #[kernel]
    fn shift(value: i32, #[comptime] offset: i32) -> i32;

    fn ordinary(value: i32) -> i32;
}

#[kernel]
pub trait WholeTransform {
    fn shift(value: i32, #[comptime] offset: i32) -> i32;
}

#[kernel]
pub trait IndexedTransform {
    fn adjust(value: i32, #[comptime] offset: usize) -> i32;
}

pub trait PlanTransform {
    #[kernel]
    fn shift(#[comptime] &self, value: i32) -> i32;
}

pub trait OtherPlanTransform {
    #[kernel]
    fn shift(#[comptime] &self, value: i32) -> i32;
}

pub trait OwnedPlanTransform {
    #[kernel]
    fn shift(#[comptime] self, value: i32) -> i32;
}

pub trait ParameterizedTransform<Tag: Copy> {
    #[kernel]
    fn shift(value: i32, #[comptime] offset: i32) -> i32;

    fn ordinary(value: i32) -> i32;
}

#[kernel]
pub trait WideTransform {
    fn combine(value: i32, #[comptime] offset: i32, subtract: bool) -> i32;
}
