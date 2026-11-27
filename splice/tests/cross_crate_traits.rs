use splice::{Codegen, Executable, hir::Instr, kernel};
use splice_test_traits::IndexedTransform as ImportedIndexed;
use splice_test_traits::ReverseTransform as ImportedReverse;
use splice_test_traits::Transform as Imported;
use splice_test_traits::{
    OrdinaryTransform, OtherPlanTransform, OwnedPlanTransform, ParameterizedTransform,
    PlanTransform, WideTransform,
};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

static ORDINARY_CALLS: AtomicUsize = AtomicUsize::new(0);
static OPAQUE_SHIFT_CALLS: AtomicUsize = AtomicUsize::new(0);
static OWNED_PLAN_DROPS: AtomicUsize = AtomicUsize::new(0);
static PARAMETERIZED_CALLS: AtomicUsize = AtomicUsize::new(0);
static LIFETIMED_DROPS: AtomicUsize = AtomicUsize::new(0);
static OTHER_RECEIVER_CALLS: AtomicUsize = AtomicUsize::new(0);

struct Local<const EXTRA: i32>;

#[derive(splice::KernelType)]
struct ReceiverPlan<'a, Tag> {
    base: &'a i32,
    tag: PhantomData<Tag>,
}

macro_rules! publish_plan_transform {
    ($plan:ident) => {
        #[kernel]
        impl<Tag> PlanTransform for $plan<'_, Tag> {
            fn shift(#[comptime] &self, value: i32) -> i32 {
                (*self.base).wrapping_add(value)
            }
        }
    };
}

publish_plan_transform!(ReceiverPlan);

#[derive(splice::KernelType)]
struct LifetimedPlan<'p> {
    value: &'p i32,
    invariant: PhantomData<std::cell::Cell<&'p i32>>,
}

impl Drop for LifetimedPlan<'_> {
    fn drop(&mut self) {
        LIFETIMED_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

#[kernel]
impl<'p> PlanTransform for LifetimedPlan<'p> {
    fn shift(#[comptime] &self, value: i32) -> i32 {
        (*self.value).wrapping_add(value)
    }
}

#[kernel]
impl<'p> OwnedPlanTransform for LifetimedPlan<'p> {
    fn shift(#[comptime] self, value: i32) -> i32 {
        (*self.value).wrapping_sub(value)
    }
}

impl OtherPlanTransform for LifetimedPlan<'_> {
    fn shift(&self, value: i32) -> i32 {
        OTHER_RECEIVER_CALLS.fetch_add(1, Ordering::Relaxed);
        value
    }
}

struct Parameterized;

#[kernel]
impl ParameterizedTransform<i32> for Parameterized {
    fn shift(value: i32, #[comptime] offset: i32) -> i32 {
        value.wrapping_add(offset)
    }
}

#[kernel]
impl ParameterizedTransform<bool> for Parameterized {
    fn shift(value: i32, #[comptime] offset: i32) -> i32 {
        value.wrapping_sub(offset)
    }
}

impl<Tag> OrdinaryTransform<Tag> for Parameterized {
    fn ordinary(value: i32) -> i32 {
        PARAMETERIZED_CALLS.fetch_add(1, Ordering::Relaxed);
        value
    }
}

#[derive(splice::KernelType)]
struct OwnedPlan {
    base: i32,
}

impl Drop for OwnedPlan {
    fn drop(&mut self) {
        OWNED_PLAN_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

#[kernel]
impl OwnedPlanTransform for OwnedPlan {
    fn shift(#[comptime] self, value: i32) -> i32 {
        self.base.wrapping_add(value)
    }
}

#[kernel]
impl<const EXTRA: i32> Imported for Local<EXTRA> {
    fn shift(value: i32, #[comptime] offset: i32) -> i32 {
        value.wrapping_add(offset).wrapping_add(EXTRA)
    }
}

impl<const EXTRA: i32> OrdinaryTransform for Local<EXTRA> {
    fn ordinary(value: i32) -> i32 {
        ORDINARY_CALLS.fetch_add(1, Ordering::Relaxed);
        value.wrapping_mul(2)
    }
}

#[kernel]
impl<const EXTRA: i32> ImportedReverse for Local<EXTRA> {
    fn shift(value: i32, #[comptime] offset: i32) -> i32 {
        value.wrapping_sub(offset).wrapping_add(EXTRA)
    }
}

struct Subtracting;

#[kernel]
impl Imported for Subtracting {
    fn shift(value: i32, #[comptime] offset: i32) -> i32 {
        value.wrapping_sub(offset)
    }
}

impl OrdinaryTransform for Subtracting {
    fn ordinary(value: i32) -> i32 {
        ORDINARY_CALLS.fetch_add(1, Ordering::Relaxed);
        value.wrapping_mul(2)
    }
}

#[kernel]
impl ImportedIndexed for Subtracting {
    fn adjust(value: i32, #[comptime] offset: usize) -> i32 {
        value.wrapping_add(offset as i32)
    }
}

struct OpaqueImplementation;

impl Imported for OpaqueImplementation {
    fn shift(value: i32, offset: i32) -> i32 {
        OPAQUE_SHIFT_CALLS.fetch_add(1, Ordering::Relaxed);
        value.wrapping_add(offset)
    }
}

impl OrdinaryTransform for OpaqueImplementation {
    fn ordinary(value: i32) -> i32 {
        value
    }
}

impl WideTransform for OpaqueImplementation {
    fn combine(value: i32, offset: i32, subtract: bool) -> i32 {
        OPAQUE_SHIFT_CALLS.fetch_add(1, Ordering::Relaxed);
        let scaled = offset.wrapping_mul(7);
        if subtract {
            value.wrapping_sub(scaled)
        } else {
            value.wrapping_add(scaled)
        }
    }
}

#[kernel]
fn route<T: Imported + OrdinaryTransform>(#[comptime] offset: i32, value: i32) -> i32 {
    let folded = comptime!(<T as Imported>::shift(2, offset));
    folded
        .wrapping_add(<T as Imported>::shift(value, offset))
        .wrapping_add(<T as OrdinaryTransform>::ordinary(value))
}

#[kernel]
fn require_ordinary(#[comptime] value: i32) -> i32 {
    comptime!(<Local<0> as OrdinaryTransform>::ordinary(value))
}

#[kernel]
fn route_reverse<T: ImportedReverse>(#[comptime] offset: i32, value: i32) -> i32 {
    let folded = comptime!(<T as ImportedReverse>::shift(2, offset));
    let base = comptime!(<T as Imported>::shift(2, offset));
    folded
        .wrapping_add(<T as ImportedReverse>::shift(value, offset))
        .wrapping_add(base)
        .wrapping_add(<T as Imported>::shift(value, offset))
}

#[kernel]
fn route_indexed(#[comptime] offset: usize, value: i32) -> i32 {
    let folded = comptime!(<Subtracting as ImportedIndexed>::adjust(2, offset));
    folded.wrapping_add(<Subtracting as ImportedIndexed>::adjust(
        <Subtracting as OrdinaryTransform>::ordinary(value),
        offset,
    ))
}

#[kernel]
fn runtime_only<T: Imported>(#[comptime] offset: i32, value: i32) -> i32 {
    <T as Imported>::shift(value, offset)
}

#[kernel]
fn runtime_wide(#[comptime] offset: i32, value: i32, subtract: bool) -> i32 {
    <OpaqueImplementation as WideTransform>::combine(value, offset, subtract)
}

#[kernel]
fn mismatched_stage<T: Imported>(value: i32, offset: i32) -> i32 {
    <T as Imported>::shift(value, offset)
}

#[kernel]
fn route_receiver<T: PlanTransform>(#[comptime] plan: &T, value: i32) -> i32 {
    let folded = comptime!(<T as PlanTransform>::shift(plan, 2));
    folded.wrapping_add(<T as PlanTransform>::shift(plan, value))
}

#[kernel]
fn route_lifetimed<'p>(#[comptime] plan: &LifetimedPlan<'p>, value: i32) -> i32 {
    let folded = comptime!(<LifetimedPlan<'p> as PlanTransform>::shift(plan, 2));
    folded.wrapping_add(<LifetimedPlan as PlanTransform>::shift(plan, value))
}

#[kernel]
fn route_lifetimed_owned(#[comptime] plan: LifetimedPlan<'_>, value: i32) -> i32 {
    <LifetimedPlan<'_> as OwnedPlanTransform>::shift(plan, value)
}

#[kernel]
fn require_other_receiver(#[comptime] plan: &LifetimedPlan<'_>) -> i32 {
    comptime!(<LifetimedPlan<'_> as OtherPlanTransform>::shift(plan, 2))
}

#[test]
fn lifetimed_receivers_are_consumed_or_borrowed_only_at_build() {
    fn check<B: Codegen<Output: Executable>>() {
        let before = LIFETIMED_DROPS.load(Ordering::Relaxed);
        let (borrowed, borrowed_ir, owned, owned_ir) = {
            let storage = 5;
            let plan = || LifetimedPlan {
                value: &storage,
                invariant: PhantomData,
            };
            let borrowed_plan = plan();
            let borrowed = route_lifetimed::build::<B>(route_lifetimed::BuildArgs {
                plan: &borrowed_plan,
            })
            .unwrap();
            let borrowed_ir = route_lifetimed::ir(&borrowed_plan).unwrap();
            let owned = route_lifetimed_owned::build::<B>(route_lifetimed_owned::BuildArgs {
                plan: plan(),
            })
            .unwrap();
            let owned_ir = route_lifetimed_owned::ir(plan()).unwrap();
            assert!(require_other_receiver::ir(&borrowed_plan).is_err());
            assert!(
                require_other_receiver::build::<B>(require_other_receiver::BuildArgs {
                    plan: &borrowed_plan
                })
                .is_err()
            );
            (borrowed, borrowed_ir, owned, owned_ir)
        };
        assert_eq!(LIFETIMED_DROPS.load(Ordering::Relaxed), before + 3);
        assert_eq!(OTHER_RECEIVER_CALLS.load(Ordering::Relaxed), 0);
        for code in [borrowed, B::compile(&borrowed_ir).unwrap()] {
            assert_eq!(
                code.call(route_lifetimed::CallArgs { value: 4 }).unwrap(),
                16
            );
        }
        for code in [owned, B::compile(&owned_ir).unwrap()] {
            assert_eq!(
                code.call(route_lifetimed_owned::CallArgs { value: 4 })
                    .unwrap(),
                1
            );
        }
        assert_eq!(LIFETIMED_DROPS.load(Ordering::Relaxed), before + 3);
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
fn route_concrete_trait_argument(value: i32) -> i32 {
    let folded = comptime!(<Parameterized as ParameterizedTransform<i32>>::shift(2, 3));
    folded.wrapping_add(<Parameterized as ParameterizedTransform<i32>>::shift(
        value, 3,
    ))
}

#[kernel]
fn route_generic_trait_argument<T, U>(value: i32) -> i32
where
    T: ParameterizedTransform<U> + OrdinaryTransform<U>,
    U: Copy,
{
    let folded = comptime!(<T as ParameterizedTransform<U>>::shift(2, 3));
    folded
        .wrapping_add(<T as ParameterizedTransform<U>>::shift(value, 3))
        .wrapping_add(<T as OrdinaryTransform<U>>::ordinary(value))
}

#[kernel]
fn require_generic_ordinary<T: OrdinaryTransform<U>, U: Copy>(#[comptime] value: i32) -> i32 {
    comptime!(<T as OrdinaryTransform<U>>::ordinary(value))
}

#[test]
fn generic_arguments_select_the_exact_trait_instance() {
    fn check<B: Codegen<Output: Executable>>() {
        let ir = route_concrete_trait_argument::ir().unwrap();
        let direct =
            route_concrete_trait_argument::build::<B>(route_concrete_trait_argument::BuildArgs {})
                .unwrap();
        for code in [direct, B::compile(&ir).unwrap()] {
            assert_eq!(
                code.call(route_concrete_trait_argument::CallArgs { value: 4 })
                    .unwrap(),
                12
            );
        }

        check_instance::<B, i32>(16);
        check_instance::<B, bool>(4);
    }

    fn check_instance<B: Codegen<Output: Executable>, U: Copy>(expected: i32)
    where
        Parameterized: ParameterizedTransform<U>,
    {
        let before = PARAMETERIZED_CALLS.load(Ordering::Relaxed);
        assert!(require_generic_ordinary::ir::<Parameterized, U>(4).is_err());
        assert!(
            require_generic_ordinary::build::<B, Parameterized, U>(
                require_generic_ordinary::BuildArgs { value: 4 },
            )
            .is_err()
        );
        let ir = route_generic_trait_argument::ir::<Parameterized, U>().unwrap();
        let direct = route_generic_trait_argument::build::<B, Parameterized, U>(
            route_generic_trait_argument::BuildArgs {},
        )
        .unwrap();
        let portable = B::compile(&ir).unwrap();
        assert_eq!(PARAMETERIZED_CALLS.load(Ordering::Relaxed), before);
        for code in [direct, portable] {
            assert_eq!(
                code.call(route_generic_trait_argument::CallArgs { value: 4 })
                    .unwrap(),
                expected
            );
        }
        assert_eq!(PARAMETERIZED_CALLS.load(Ordering::Relaxed), before + 2);
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
fn route_owned_receiver(#[comptime] plan: OwnedPlan, value: i32) -> i32 {
    <OwnedPlan as OwnedPlanTransform>::shift(plan, value)
}

#[kernel]
fn fold_owned_receiver(#[comptime] plan: OwnedPlan) -> i32 {
    comptime!(<OwnedPlan as OwnedPlanTransform>::shift(plan, 4))
}

#[test]
fn owned_receiver_drops_once_in_each_build() {
    fn check<B: Codegen<Output: Executable>>() {
        let before = OWNED_PLAN_DROPS.load(Ordering::Relaxed);
        let direct = route_owned_receiver::build::<B>(route_owned_receiver::BuildArgs {
            plan: OwnedPlan { base: 5 },
        })
        .unwrap();
        let ir = route_owned_receiver::ir(OwnedPlan { base: 7 }).unwrap();
        let folded = fold_owned_receiver::build::<B>(fold_owned_receiver::BuildArgs {
            plan: OwnedPlan { base: 9 },
        })
        .unwrap();
        let folded_ir = fold_owned_receiver::ir(OwnedPlan { base: 13 }).unwrap();
        assert_eq!(OWNED_PLAN_DROPS.load(Ordering::Relaxed), before + 4);

        let portable = B::compile(&ir).unwrap();
        assert_eq!(
            direct
                .call(route_owned_receiver::CallArgs { value: 4 })
                .unwrap(),
            9
        );
        assert_eq!(
            portable
                .call(route_owned_receiver::CallArgs { value: 4 })
                .unwrap(),
            11
        );
        assert_eq!(folded.call(fold_owned_receiver::CallArgs {}).unwrap(), 13);
        assert_eq!(
            B::compile(&folded_ir)
                .unwrap()
                .call(fold_owned_receiver::CallArgs {})
                .unwrap(),
            17
        );
        assert_eq!(OWNED_PLAN_DROPS.load(Ordering::Relaxed), before + 4);
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[test]
fn macro_generated_impl_specializes_a_generic_borrow_without_a_static_bound() {
    fn check<B: Codegen<Output: Executable>>() {
        let (direct, ir) = {
            let storage = 5;
            let plan = ReceiverPlan::<u8> {
                base: &storage,
                tag: PhantomData,
            };
            (
                route_receiver::build::<B, ReceiverPlan<'_, u8>>(route_receiver::BuildArgs {
                    plan: &plan,
                })
                .unwrap(),
                route_receiver::ir(&plan).unwrap(),
            )
        };
        let portable = B::compile(&ir).unwrap();
        for code in [direct, portable] {
            assert_eq!(
                code.call(route_receiver::CallArgs { value: 4 }).unwrap(),
                16
            );
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[test]
fn trait_and_impl_publication_follow_an_imported_trait_alias() {
    fn check<B: Codegen<Output: Executable>>() {
        let before = ORDINARY_CALLS.load(Ordering::Relaxed);
        assert!(require_ordinary::ir(8).is_err());
        assert!(require_ordinary::build::<B>(require_ordinary::BuildArgs { value: 8 }).is_err());
        assert_eq!(ORDINARY_CALLS.load(Ordering::Relaxed), before);
        check_instance::<B, Local<0>>(20);
        check_instance::<B, Local<2>>(24);
        check_instance::<B, Subtracting>(8);

        let ir = route_reverse::ir::<Local<5>>(3).unwrap();
        let direct =
            route_reverse::build::<B, Local<5>>(route_reverse::BuildArgs { offset: 3 }).unwrap();
        let portable = B::compile(&ir).unwrap();
        for code in [direct, portable] {
            assert_eq!(code.call(route_reverse::CallArgs { value: 4 }).unwrap(), 32);
        }

        let expected = route_indexed(3, 4);
        let before = ORDINARY_CALLS.load(Ordering::Relaxed);
        let ir = route_indexed::ir(3).unwrap();
        let direct = route_indexed::build::<B>(route_indexed::BuildArgs { offset: 3 }).unwrap();
        let portable = B::compile(&ir).unwrap();
        assert_eq!(ORDINARY_CALLS.load(Ordering::Relaxed), before);
        for (index, code) in [direct, portable].into_iter().enumerate() {
            assert_eq!(
                code.call(route_indexed::CallArgs { value: 4 }).unwrap(),
                expected
            );
            assert_eq!(ORDINARY_CALLS.load(Ordering::Relaxed), before + index + 1);
        }

        let before = OPAQUE_SHIFT_CALLS.load(Ordering::Relaxed);
        assert!(route::ir::<OpaqueImplementation>(3).is_err());
        assert!(route::build::<B, OpaqueImplementation>(route::BuildArgs { offset: 3 }).is_err());
        assert!(mismatched_stage::ir::<OpaqueImplementation>().is_err());
        assert!(mismatched_stage::ir::<Local<0>>().is_err());
        assert!(
            mismatched_stage::build::<B, OpaqueImplementation>(mismatched_stage::BuildArgs {})
                .is_err()
        );
        assert!(mismatched_stage::build::<B, Local<0>>(mismatched_stage::BuildArgs {}).is_err());

        let ir = runtime_only::ir::<OpaqueImplementation>(3).unwrap();
        assert!(
            ir.instructions()
                .iter()
                .any(|op| matches!(op, Instr::RustCall { args, .. } if args.len() == 2))
        );
        let direct =
            runtime_only::build::<B, OpaqueImplementation>(runtime_only::BuildArgs { offset: 3 })
                .unwrap();
        let portable = B::compile(&ir).unwrap();
        assert_eq!(OPAQUE_SHIFT_CALLS.load(Ordering::Relaxed), before);
        for code in [direct, portable] {
            assert_eq!(code.call(runtime_only::CallArgs { value: 4 }).unwrap(), 7);
        }
        assert_eq!(OPAQUE_SHIFT_CALLS.load(Ordering::Relaxed), before + 2);

        let expected = <OpaqueImplementation as WideTransform>::combine(4, 3, true);
        let before = OPAQUE_SHIFT_CALLS.load(Ordering::Relaxed);
        let ir = runtime_wide::ir(3).unwrap();
        assert!(
            ir.instructions()
                .iter()
                .any(|op| matches!(op, Instr::RustCall { args, .. } if args.len() == 3))
        );
        let direct = runtime_wide::build::<B>(runtime_wide::BuildArgs { offset: 3 }).unwrap();
        let portable = B::compile(&ir).unwrap();
        assert_eq!(OPAQUE_SHIFT_CALLS.load(Ordering::Relaxed), before);
        for code in [direct, portable] {
            assert_eq!(
                code.call(runtime_wide::CallArgs {
                    value: 4,
                    subtract: true
                })
                .unwrap(),
                expected
            );
        }
        assert_eq!(OPAQUE_SHIFT_CALLS.load(Ordering::Relaxed), before + 2);
    }

    fn check_instance<B: Codegen<Output: Executable>, T: Imported + OrdinaryTransform>(
        expected: i32,
    ) {
        let before = ORDINARY_CALLS.load(Ordering::Relaxed);
        let ir = route::ir::<T>(3).unwrap();
        let direct = route::build::<B, T>(route::BuildArgs { offset: 3 }).unwrap();
        let portable = B::compile(&ir).unwrap();
        assert_eq!(ORDINARY_CALLS.load(Ordering::Relaxed), before);

        for code in [direct, portable] {
            assert_eq!(code.call(route::CallArgs { value: 4 }).unwrap(), expected);
        }
        assert_eq!(ORDINARY_CALLS.load(Ordering::Relaxed), before + 2);
    }

    splice_testkit::each_backend!(B => check::<B>());
}
