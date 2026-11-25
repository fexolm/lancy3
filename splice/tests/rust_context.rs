use splice::hir::{Buffer, Instr, RustTarget};
use splice::{CallError, Codegen, Executable, Trap, TypedIr, kernel};
use std::collections::HashSet;
use std::ops::AddAssign;

#[derive(Default)]
struct Counter {
    total: i32,
    operands: Vec<i32>,
}

impl AddAssign<i32> for Counter {
    fn add_assign(&mut self, operand: i32) {
        self.total = self.total.wrapping_add(operand);
        self.operands.push(operand);
    }
}

#[derive(Default)]
struct Totals {
    count: Counter,
    completed: i32,
    fail_completion: bool,
}

impl Totals {
    fn complete(&mut self) {
        self.completed += 1;
        assert!(!self.fail_completion, "completion failed");
    }

    fn record_values(&mut self, xs: &[i32]) -> i32 {
        let sum = xs.iter().copied().fold(0, i32::wrapping_add);
        self.count += sum;
        sum
    }

    fn observe_pair(&mut self, first: &[i32], second: &[i32]) -> i32 {
        let first = first[0];
        self.count += first;
        let second = second[1];
        self.count += second;
        first.wrapping_mul(10).wrapping_add(second)
    }

    fn modify_slice(
        &mut self,
        output: &mut [i32],
        codes: &[u16],
        delta: i32,
        enabled: bool,
    ) -> i32 {
        self.count += output[0];
        if enabled {
            output[0] = output[0].wrapping_add(delta);
        }
        output[1] = i32::from(codes[0]);
        output[0]
    }

    fn record_element_count<T>(&mut self, values: &[T]) -> i32 {
        let len = values.len() as i32;
        self.count += len;
        len
    }

    fn record_words(&mut self, a: i32, enabled: bool, b: i32, c: i32) -> i32 {
        let result = a
            .wrapping_mul(100)
            .wrapping_add(b.wrapping_mul(10))
            .wrapping_add(c);
        let result = if enabled {
            result
        } else {
            result.wrapping_neg()
        };
        self.count += result;
        result
    }
}

#[kernel]
impl Totals {
    fn update(&mut self, xs: &[i32]) -> i32 {
        let mut sum = 0_i32;
        for &value in xs {
            sum = sum.wrapping_add(value);
        }
        self.count += sum;
        sum
    }

    fn read_pair(&mut self, first: &[i32], second: &[i32]) -> i32 {
        self.observe_pair(first, second)
    }

    fn update_slice(
        &mut self,
        output: &mut [i32],
        codes: &[u16],
        delta: i32,
        enabled: bool,
    ) -> i32 {
        self.modify_slice(output, codes, delta, enabled)
    }

    fn generic_slice<T: splice::BufferElement>(&mut self, values: &[T]) -> i32 {
        self.record_element_count(values)
    }

    fn record_len(&mut self, xs: &[i32]) -> i32 {
        let length = xs.len() as i32;
        self.completed += length;
        length
    }
}

#[kernel]
fn run(totals: &mut Totals, xs: &[i32]) -> i32 {
    totals.complete();
    let first = Totals::update(totals, xs);
    let selected = xs[0];
    let second = Totals::update(totals, xs);
    let length = Totals::record_len(totals, xs);
    first
        .wrapping_add(second)
        .wrapping_add(selected)
        .wrapping_add(length)
}

#[kernel]
fn single_published(totals: &mut Totals, xs: &[i32]) -> i32 {
    Totals::record_len(totals, xs)
}

#[kernel]
fn single_ordinary(totals: &mut Totals, xs: &[i32]) -> i32 {
    Totals::record_values(totals, xs)
}

mod nested_helpers {
    use super::{Totals, kernel};

    #[kernel]
    pub(super) fn add_scaled(scale: i32, totals: &mut Totals, xs: &[i32]) -> i32 {
        let mut sum = 0_i32;
        for &value in xs {
            sum = sum.wrapping_add(value);
        }
        let amount = sum.wrapping_mul(scale);
        totals.count += amount;
        amount
    }

    #[kernel]
    pub(super) fn double(totals: &mut Totals, xs: &[i32]) -> i32 {
        add_scaled(2, &mut *totals, xs)
    }

    #[kernel]
    #[expect(
        clippy::borrow_deref_ref,
        reason = "exercise remapping through a shared reborrow"
    )]
    pub(super) fn read_reordered(totals: &mut Totals, left: &[i32], right: &[i32]) -> i32 {
        let alias = &*right;
        Totals::read_pair(totals, alias, left)
    }

    #[kernel]
    pub(super) fn generic_slice<T: splice::BufferElement>(
        totals: &mut Totals,
        values: &[T],
    ) -> i32 {
        Totals::generic_slice(totals, values)
    }
}

#[kernel]
fn remap_opaque_slices(totals: &mut Totals, left: &[i32], right: &[i32], out: &mut [i32]) -> i32 {
    out[0] = 101;
    let first = nested_helpers::read_reordered(totals, left, right);
    out[1] = first;
    let second = nested_helpers::read_reordered(totals, left, left);
    first.wrapping_add(second)
}

#[kernel]
fn writable_opaque_slice(
    totals: &mut Totals,
    source: &[i32],
    codes: &[u16],
    out: &mut [i32],
    enabled: bool,
) -> i32 {
    let mut sum = 0_i32;
    for &value in source {
        sum = sum.wrapping_add(value);
    }
    out[0] = sum;
    let changed = Totals::update_slice(totals, &mut *out, codes, sum.wrapping_add(3), enabled);
    let shared = &*out;
    let observed = Totals::read_pair(totals, shared, shared);
    out[1] = observed;
    changed.wrapping_add(out[0]).wrapping_add(observed)
}

#[kernel]
fn generic_opaque_slice(totals: &mut Totals, first: &[u16], second: &[f32]) -> i32 {
    let left = nested_helpers::generic_slice(totals, first);
    let right = nested_helpers::generic_slice(totals, second);
    left.wrapping_add(right)
}

#[kernel]
fn nested_context_call(totals: &mut Totals, xs: &[i32]) -> i32 {
    totals.complete();
    let first = nested_helpers::add_scaled(1, totals, xs);
    let selected = xs[0];
    let second = nested_helpers::add_scaled(2, totals, xs);
    first.wrapping_add(second).wrapping_add(selected)
}

#[kernel]
fn forward_context(totals: &mut Totals, xs: &[i32]) -> i32 {
    nested_helpers::double(totals, xs)
}

fn record_into_slice(totals: &mut Totals, xs: &mut [i32], a: i32, b: i32, c: i32) -> i32 {
    let sum = a.wrapping_add(b).wrapping_add(c);
    totals.count += sum;
    xs[0] = sum;
    sum
}

#[kernel]
fn ordinary_free_call(totals: &mut Totals, xs: &mut [i32], a: i32, b: i32, c: i32) -> i32 {
    record_into_slice(totals, xs, a, b, c)
}

#[kernel]
fn wide_context_pipeline(
    totals: &mut Totals,
    output: &mut [i32],
    a: i32,
    enabled: bool,
    b: i32,
    c: i32,
) -> i32 {
    let logged = totals.record_words(a, enabled, b, c);
    let stored = record_into_slice(totals, output, a, b, c);
    logged.wrapping_add(stored)
}

fn has_backedge(code: &[Instr]) -> bool {
    let mut labels = HashSet::new();
    code.iter().any(|op| match op {
        Instr::Label(label) => {
            labels.insert(*label);
            false
        }
        Instr::Jump(label) => labels.contains(label),
        _ => false,
    })
}

#[test]
fn nested_opaque_operations_use_the_callers_reordered_shared_slices() {
    let ir = remap_opaque_slices::ir().unwrap();
    let instructions = ir.instructions();
    assert!(
        instructions
            .iter()
            .any(|op| matches!(op, Instr::RustCall { .. }))
    );
    assert!(
        instructions
            .iter()
            .any(|op| matches!(op, Instr::Store { .. }))
    );

    fn check<B: Codegen<Output: Executable>>(ir: &TypedIr<remap_opaque_slices::Signature>) {
        for code in [
            remap_opaque_slices::build::<B>(remap_opaque_slices::BuildArgs {}).unwrap(),
            B::compile(ir).unwrap(),
        ] {
            for left in [&[4, 7][..], &[4][..]] {
                let right = [11, 13];
                let mut expected = Totals::default();
                let mut expected_out = [0, 98];
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    remap_opaque_slices(&mut expected, left, &right, &mut expected_out)
                }))
                .ok();

                let actual_left = left.to_vec();
                let actual_right = right;
                let mut actual = Totals::default();
                let mut out = [0, 98];
                let actual_result = code.call(remap_opaque_slices::CallArgs {
                    totals: &mut actual,
                    left: &actual_left,
                    right: &actual_right,
                    out: &mut out,
                });
                assert_eq!(actual_result.ok(), result);
                assert_eq!(out, expected_out);
                assert_eq!(actual.count.operands, expected.count.operands);
                assert_eq!(actual_left, left);
                assert_eq!(actual_right, right);
            }
        }
    }

    splice_testkit::each_backend!(B => check::<B>(&ir));
}

#[test]
fn native_and_rust_operations_reborrow_the_same_writable_slice_in_order() {
    let ir = writable_opaque_slice::ir().unwrap();
    let instructions = ir.instructions();
    assert!(has_backedge(instructions));
    let borrows = instructions
        .iter()
        .filter_map(|instruction| match instruction {
            Instr::RustCall { slices, .. } => Some(slices.as_slice()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [[first, tag], [left, right]] = borrows[..] else {
        panic!("expected a mutable view followed by repeated shared views: {borrows:?}");
    };
    assert!(first.writable && !tag.writable && !left.writable && !right.writable);
    assert_eq!(first.buffer, left.buffer);
    assert_eq!(left, right);
    assert_eq!((first.buffer, tag.buffer), (Buffer(3), Buffer(2)));

    fn check<B: Codegen<Output: Executable>>(ir: &TypedIr<writable_opaque_slice::Signature>) {
        for code in [
            writable_opaque_slice::build::<B>(writable_opaque_slice::BuildArgs {}).unwrap(),
            B::compile(ir).unwrap(),
        ] {
            for (codes, enabled) in [(&[29_u16][..], true), (&[13][..], false), (&[][..], true)] {
                let source = [7, 11];
                let mut expected_totals = Totals::default();
                let mut expected_out = [0, -1];
                let expected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    writable_opaque_slice(
                        &mut expected_totals,
                        &source,
                        codes,
                        &mut expected_out,
                        enabled,
                    )
                }))
                .ok();
                let mut actual_totals = Totals::default();
                let mut out = [0, -1];
                let actual = code
                    .call(writable_opaque_slice::CallArgs {
                        totals: &mut actual_totals,
                        source: &source,
                        codes,
                        out: &mut out,
                        enabled,
                    })
                    .ok();
                assert_eq!(actual, expected);
                assert_eq!(actual_totals.count.operands, expected_totals.count.operands);
                assert_eq!(out, expected_out);
            }
        }
    }

    splice_testkit::each_backend!(B => check::<B>(&ir));
}

#[test]
fn nested_slice_descriptors_preserve_inferred_element_types() {
    let ir = generic_opaque_slice::ir().unwrap();
    assert_eq!(
        ir.instructions()
            .iter()
            .filter(|op| matches!(op, Instr::RustCall { slices, .. } if slices.len() == 1))
            .count(),
        2
    );
    fn check<B: Codegen<Output: Executable>>(ir: &TypedIr<generic_opaque_slice::Signature>) {
        for code in [
            generic_opaque_slice::build::<B>(generic_opaque_slice::BuildArgs {}).unwrap(),
            B::compile(ir).unwrap(),
        ] {
            let mut totals = Totals::default();
            let first = [5, 3];
            let second = [3.5, 4.0, 9.0];
            assert_eq!(
                code.call(generic_opaque_slice::CallArgs {
                    totals: &mut totals,
                    first: &first,
                    second: &second
                })
                .unwrap(),
                5
            );
            assert_eq!(totals.count.operands, [2, 3]);
        }
    }
    splice_testkit::each_backend!(B => check::<B>(&ir));
}

#[test]
fn opaque_context_operations_receive_all_scalar_words() {
    let ir = wide_context_pipeline::ir().unwrap();
    let shapes = ir
        .instructions()
        .iter()
        .filter_map(|instruction| match instruction {
            Instr::RustCall { args, slices, .. } => Some((args.len(), slices.len())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(shapes, [(4, 0), (3, 1)]);
    #[cfg(feature = "llvm")]
    {
        let llvm = splice_llvm::to_llvm_ir(ir.as_inner(), &ir.as_inner().options().target).unwrap();
        // Each call has a fixed-size packet slot in the entry block.
        let packets = llvm
            .lines()
            .filter(|line| line.contains("%rust_packet") && line.contains(" = alloca ["))
            .count();
        assert_eq!(packets, 2);
        assert!(!llvm.contains("alloca i64, i64"));
    }

    fn check<B: Codegen<Output: Executable>>(ir: &TypedIr<wide_context_pipeline::Signature>) {
        for code in [
            wide_context_pipeline::build::<B>(wide_context_pipeline::BuildArgs {}).unwrap(),
            B::compile(ir).unwrap(),
        ] {
            for (len, enabled) in [(1, false), (1, true), (0, true)] {
                let mut expected_totals = Totals::default();
                let mut expected_out = [0];
                let expected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    wide_context_pipeline(
                        &mut expected_totals,
                        &mut expected_out[..len],
                        2,
                        enabled,
                        3,
                        7,
                    )
                }))
                .ok();
                let mut actual_totals = Totals::default();
                let mut actual_out = [0];
                let actual = code
                    .call(wide_context_pipeline::CallArgs {
                        totals: &mut actual_totals,
                        output: &mut actual_out[..len],
                        a: 2,
                        enabled,
                        b: 3,
                        c: 7,
                    })
                    .ok();
                assert_eq!(actual, expected);
                assert_eq!(actual_totals.count.operands, expected_totals.count.operands);
                assert_eq!(actual_out, expected_out);
            }
        }
    }
    splice_testkit::each_backend!(B => check::<B>(&ir));
}

#[kernel]
fn update(totals: &mut Totals, xs: &[i32], index: i32) -> i32 {
    let mut sum = 0_i32;
    for &value in xs {
        sum = sum.wrapping_add(value);
    }
    totals.count += sum;
    let selected = xs[index as usize];
    totals.complete();
    selected.wrapping_add(sum)
}

#[test]
fn rust_context_operations_surround_a_native_fault_and_keep_their_effects() {
    let ir = update::ir().unwrap();
    let code = ir.instructions();
    let calls: Vec<_> = code
        .iter()
        .enumerate()
        .filter_map(|(index, op)| matches!(op, Instr::RustCall { .. }).then_some(index))
        .collect();
    let [add_count, complete] = calls.as_slice() else {
        panic!("expected two Rust context calls, got {calls:?}");
    };
    assert!(
        code[add_count + 1..*complete]
            .iter()
            .any(|op| matches!(op, Instr::Load { .. }))
    );
    assert!(has_backedge(&code[..*add_count]));

    fn check<B: Codegen<Output: Executable>>(ir: &TypedIr<update::Signature>) {
        for code in [
            update::build::<B>(update::BuildArgs {}).unwrap(),
            B::compile(ir).unwrap(),
        ] {
            let mut totals = Totals::default();
            totals.count.total = 7;
            let xs = [2, 3];

            assert_eq!(
                code.call(update::CallArgs {
                    totals: &mut totals,
                    xs: &xs,
                    index: 2
                }),
                Err(CallError::Trap(Trap::Bounds))
            );
            assert_eq!(totals.count.total, 12);
            assert_eq!(totals.count.operands, [5]);
            assert_eq!(totals.completed, 0);

            assert_eq!(
                code.call(update::CallArgs {
                    totals: &mut totals,
                    xs: &xs,
                    index: 1
                })
                .unwrap(),
                8
            );
            assert_eq!(totals.count.total, 17);
            assert_eq!(totals.count.operands, [5, 5]);
            assert_eq!(totals.completed, 1);

            let mut other = Totals {
                fail_completion: true,
                ..Totals::default()
            };
            // The context operation's own panic message reaches the caller.
            assert_eq!(
                code.call(update::CallArgs {
                    totals: &mut other,
                    xs: &xs,
                    index: 0
                }),
                Err(CallError::RustPanic("completion failed".into()))
            );
            assert_eq!(other.count.operands, [5]);
            assert_eq!(other.completed, 1);

            other.fail_completion = false;
            assert_eq!(
                code.call(update::CallArgs {
                    totals: &mut other,
                    xs: &xs,
                    index: 0
                })
                .unwrap(),
                7
            );
            assert_eq!(other.count.total, 10);
            assert_eq!(other.count.operands, [5, 5]);
            assert_eq!(other.completed, 2);
            assert_eq!(totals.count.operands, [5, 5]);
        }
    }

    splice_testkit::each_backend!(B => check::<B>(&ir));
}

#[test]
fn published_methods_share_the_receivers_context_without_confusing_their_operations() {
    let ir = run::ir().unwrap();
    let instructions = ir.instructions();
    assert!(has_backedge(instructions));
    assert!(
        instructions
            .iter()
            .any(|op| matches!(op, Instr::Load { .. }))
    );
    let operations: HashSet<_> = instructions
        .iter()
        .filter_map(|op| match op {
            Instr::RustCall {
                target: RustTarget::Context { operation, .. },
                ..
            } => Some(*operation),
            _ => None,
        })
        .collect();
    assert!(
        operations.len() >= 3,
        "local and method effects need distinct operations: {operations:?}"
    );

    fn check<B: Codegen<Output: Executable>>(ir: &TypedIr<run::Signature>) {
        for code in [
            run::build::<B>(run::BuildArgs {}).unwrap(),
            B::compile(ir).unwrap(),
        ] {
            let mut actual = Totals::default();
            let mut expected = Totals::default();
            for xs in [&[][..], &[2, 3], &[i32::MAX, 1, -5]] {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(&mut expected, xs)
                }))
                .ok();
                assert_eq!(
                    code.call(run::CallArgs {
                        totals: &mut actual,
                        xs
                    })
                    .ok(),
                    result
                );
                assert_eq!(actual.count.total, expected.count.total);
                assert_eq!(actual.count.operands, expected.count.operands);
                assert_eq!(actual.completed, expected.completed);
            }
        }
    }

    splice_testkit::each_backend!(B => check::<B>(&ir));
}

#[test]
fn published_free_functions_share_the_enclosing_context() {
    let ir = nested_context_call::ir().unwrap();
    let instructions = ir.instructions();
    let mut seen = HashSet::new();
    let operations = instructions
        .iter()
        .enumerate()
        .filter_map(|(index, op)| match op {
            Instr::RustCall {
                target: RustTarget::Context { operation, .. },
                ..
            } => {
                assert!(seen.insert(*operation), "calls need distinct operation IDs");
                Some(index)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let [complete, first, second] = operations[..] else {
        panic!("expected three context calls, got {operations:?}");
    };
    assert!(has_backedge(&instructions[complete + 1..first]));
    assert!(has_backedge(&instructions[first + 1..second]));
    assert!(
        instructions[first + 1..second]
            .iter()
            .any(|op| matches!(op, Instr::Load { .. }))
    );

    let forwarded = forward_context::ir().unwrap();
    assert!(has_backedge(forwarded.instructions()));

    fn check<B: Codegen<Output: Executable>>(
        ir: &TypedIr<nested_context_call::Signature>,
        forwarded: &TypedIr<forward_context::Signature>,
    ) {
        for code in [
            nested_context_call::build::<B>(nested_context_call::BuildArgs {}).unwrap(),
            B::compile(ir).unwrap(),
        ] {
            let mut actual = Totals::default();
            assert!(
                code.call(nested_context_call::CallArgs {
                    totals: &mut actual,
                    xs: &[]
                })
                .is_err()
            );
            assert_eq!(actual.completed, 1);
            assert_eq!(actual.count.operands, [0]);

            let mut expected = Totals::default();
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    nested_context_call(&mut expected, &[])
                }))
                .is_err()
            );
            let xs = [2, 3];
            assert_eq!(
                code.call(nested_context_call::CallArgs {
                    totals: &mut actual,
                    xs: &xs
                })
                .unwrap(),
                nested_context_call(&mut expected, &xs)
            );
            assert_eq!(actual.count.total, expected.count.total);
            assert_eq!(actual.count.operands, expected.count.operands);
            assert_eq!(actual.completed, expected.completed);
        }

        for code in [
            forward_context::build::<B>(forward_context::BuildArgs {}).unwrap(),
            B::compile(forwarded).unwrap(),
        ] {
            let xs = [2, 3];
            let mut actual = Totals::default();
            let mut expected = Totals::default();
            assert_eq!(
                code.call(forward_context::CallArgs {
                    totals: &mut actual,
                    xs: &xs
                })
                .unwrap(),
                forward_context(&mut expected, &xs)
            );
            assert_eq!(actual.count.operands, expected.count.operands);
        }

        let ordinary = ordinary_free_call::ir().unwrap();
        for code in [
            ordinary_free_call::build::<B>(ordinary_free_call::BuildArgs {}).unwrap(),
            B::compile(&ordinary).unwrap(),
        ] {
            let mut xs = [0];
            let mut actual = Totals::default();
            assert_eq!(
                code.call(ordinary_free_call::CallArgs {
                    totals: &mut actual,
                    xs: &mut xs,
                    a: 1,
                    b: 2,
                    c: 3
                })
                .unwrap(),
                6
            );
            assert_eq!(actual.count.operands, [6]);
            assert_eq!(xs, [6]);
        }
    }

    splice_testkit::each_backend!(B => check::<B>(&ir, &forwarded));
}

#[test]
fn published_and_ordinary_methods_preserve_shared_slice_arguments() {
    let ir = single_published::ir().unwrap();
    let instructions = ir.instructions();
    assert!(
        instructions
            .iter()
            .any(|op| matches!(op, Instr::BufferLen { .. }))
    );
    assert!(
        instructions
            .iter()
            .any(|op| matches!(op, Instr::RustCall { .. }))
    );

    fn check<B: Codegen<Output: Executable>>(ir: &TypedIr<single_published::Signature>) {
        let xs = [4, -2, 5];
        for code in [
            single_published::build::<B>(single_published::BuildArgs {}).unwrap(),
            B::compile(ir).unwrap(),
        ] {
            let mut actual = Totals::default();
            let mut expected = Totals::default();
            assert_eq!(
                code.call(single_published::CallArgs {
                    totals: &mut actual,
                    xs: &xs
                })
                .unwrap(),
                single_published(&mut expected, &xs)
            );
            assert_eq!(actual.completed, expected.completed);
        }

        let ordinary = single_ordinary::ir().unwrap();
        for code in [
            single_ordinary::build::<B>(single_ordinary::BuildArgs {}).unwrap(),
            B::compile(&ordinary).unwrap(),
        ] {
            let mut actual = Totals::default();
            let mut expected = Totals::default();
            assert_eq!(
                code.call(single_ordinary::CallArgs {
                    totals: &mut actual,
                    xs: &xs
                })
                .unwrap(),
                single_ordinary(&mut expected, &xs)
            );
            assert_eq!(actual.count.total, expected.count.total);
            assert_eq!(actual.count.operands, expected.count.operands);
        }
    }

    splice_testkit::each_backend!(B => check::<B>(&ir));
}
