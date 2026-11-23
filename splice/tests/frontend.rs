use splice::hir::{Binary, BufferKind, FunctionBuilder, Instr};
use splice::{BufferArg, Codegen, Executable, Function, kernel};

// A backend used only to inspect the IR handed off by build().
struct Inspect(Function);

impl Codegen for Inspect {
    type Output = Self;

    fn compile_raw(f: &Function) -> Result<Self, String> {
        Ok(Self(f.clone()))
    }
}

#[test]
fn oversized_signature_is_rejected_by_the_portable_builder() {
    let result = Inspect::build(
        usize::MAX,
        &[],
        Box::new(|b| {
            b.ret(b.arg(0));
            Ok(())
        }),
    );
    assert!(matches!(result, Err(message) if message.contains("too many")));
}

mod pricing {
    pub struct Lane;
    impl Lane {
        pub fn surcharge(zone: i32) -> i32 {
            zone + 2
        }
    }
}

#[kernel]
fn delivery(#[comptime] zone: i32, price: i32) -> i32 {
    price + pricing::Lane::surcharge(zone)
}

#[test]
fn host_associated_functions_can_belong_to_a_type_named_lane() {
    let expected = delivery(3, 8);
    let ir = delivery::ir(3).unwrap();
    for kernel in [
        delivery::build::<splice_lancy::Lancy>(delivery::BuildArgs { zone: 3 }).unwrap(),
        splice_lancy::Lancy::compile(&ir).unwrap(),
    ] {
        assert_eq!(
            kernel.call(delivery::CallArgs { price: 8 }).unwrap(),
            expected
        );
    }
}

#[kernel]
fn staged_weights(#[comptime] bias: i32, value: i32) -> i32 {
    value + bias
}

#[test]
fn collection_computed_by_the_application_can_specialize_a_kernel() {
    let mut weights = vec![1; 3];
    weights.push(4);
    let bias = weights.iter().sum::<i32>();
    let ir = staged_weights::ir(bias).unwrap();
    assert!(
        ir.instructions()
            .iter()
            .any(|i| matches!(i, Instr::Const(_, 7)))
    );
    // Builds follow Rust's profile: debug builds check overflow.
    let add = if cfg!(debug_assertions) {
        Binary::CheckedAdd
    } else {
        Binary::Add
    };
    assert!(ir.instructions().iter().all(|i| matches!(
        i,
        Instr::Debug(_)
            | Instr::Const(..)
            | Instr::Copy(..)
            | Instr::Label(_)
            | Instr::Jump(_)
            | Instr::Return(_)
    ) || matches!(i, Instr::Bin(_, op, ..) if *op == add)));
    let source = staged_weights::print(bias).unwrap();
    assert!(
        source.contains("fn staged_weights(value: i32) -> i32 {\n    value + 7\n}"),
        "{source}"
    );

    fn check<B: Codegen<Output: Executable>>(bias: i32) {
        let ir = staged_weights::ir(bias).unwrap();
        for kernel in [
            staged_weights::build::<B>(staged_weights::BuildArgs { bias }).unwrap(),
            B::compile(&ir).unwrap(),
        ] {
            assert_eq!(
                kernel.call(staged_weights::CallArgs { value: 10 }).unwrap(),
                staged_weights(bias, 10)
            );
        }
    }

    splice_testkit::each_backend!(B => check::<B>(bias));
}

trait Factor {
    const SCALE: i32;
}

impl Factor for i32 {
    const SCALE: i32 = 2;
}

#[kernel]
fn scaled<T, const OFFSET: i32>(#[comptime] settings: &i32, row: i32) -> i32
where
    T: Factor,
{
    row.wrapping_mul(T::SCALE * *settings).wrapping_add(OFFSET)
}

#[kernel]
fn adjusted(#[comptime] positive: bool, row: i32) -> i32 {
    if positive {
        scaled::<i32, 3>(&1, row)
    } else {
        row
    }
}

#[kernel]
fn column(#[comptime] positive: bool, rows: &[i32], out: &mut [i32]) -> i32 {
    for i in 0..out.len() as i32 {
        out[i as usize] = adjusted(positive, rows[i as usize]);
    }
    out.len() as i32
}

#[kernel]
fn overlapping_reads(left: &[i32], right: &[i32], out: &mut [i32]) -> i32 {
    out[0] = left[0] + right[0];
    1
}

#[test]
fn slice_parameters_expose_their_alias_contract() {
    struct InspectKinds;
    impl Codegen for InspectKinds {
        type Output = Vec<BufferKind>;

        fn compile_raw(_: &Function) -> Result<Self::Output, String> {
            panic!("expected direct staging")
        }

        fn build(
            _: usize,
            kinds: &[BufferKind],
            _: splice::frontend::Body<'_>,
        ) -> Result<Self::Output, String> {
            Ok(kinds.to_vec())
        }
    }

    let direct = overlapping_reads::build::<InspectKinds>(overlapping_reads::BuildArgs {})
        .unwrap()
        .into_inner();
    let portable = overlapping_reads::ir().unwrap();
    assert_eq!(direct, portable.buffer_kinds());
    assert!(direct[0].may_alias(direct[1]));
    assert!(!direct[0].may_alias(direct[2]));
    assert!(!direct[2].may_alias(direct[1]));
    assert!(!direct[2].may_alias(direct[2]));
    let input = [17, 42];
    splice_testkit::each_backend!(B => {
        for kernel in [
            overlapping_reads::build::<B>(overlapping_reads::BuildArgs {}).unwrap(),
            B::compile(&portable).unwrap(),
        ] {
            let mut output = [0];
            assert_eq!(
                kernel
                    .call(overlapping_reads::CallArgs {
                        left: &input,
                        right: &input[..1],
                        out: &mut output,
                    })
                    .unwrap(),
                1
            );
            assert_eq!(output, [34]);
        }
    });
}

#[kernel]
fn shared_read_after_store(index: i32, src: &[i32], dst: &mut [i32]) -> i32 {
    let before = src[0];
    dst[0] = before;
    dst[index as usize] = 11;
    before + src[0]
}

#[kernel]
fn mutable_read_after_store(buf: &mut [i32]) -> i32 {
    let before = buf[0];
    buf[0] = 11;
    before + buf[0]
}

#[test]
fn repeated_slice_reads_preserve_writes_and_errors() {
    fn check_shared(call: impl Fn(i32, &[i32], &mut [i32]) -> Result<i32, splice::CallError>) {
        for (src, index, output, result) in [
            (&[7][..], 1, [7, 11], Some(14)),
            (&[7][..], 2, [7, 0], None),
            (&[][..], 1, [0, 0], None),
        ] {
            let mut dst = [0, 0];
            let actual = call(index, src, &mut dst);
            assert_eq!(actual.ok(), result);
            assert_eq!(dst, output);
        }
    }

    fn check<B: Codegen<Output: Executable>>() {
        for kernel in [
            shared_read_after_store::build::<B>(shared_read_after_store::BuildArgs {}).unwrap(),
            B::compile(&shared_read_after_store::ir().unwrap()).unwrap(),
        ] {
            check_shared(|index, src, dst| {
                kernel.call(shared_read_after_store::CallArgs { index, src, dst })
            });
        }
        for kernel in [
            mutable_read_after_store::build::<B>(mutable_read_after_store::BuildArgs {}).unwrap(),
            B::compile(&mutable_read_after_store::ir().unwrap()).unwrap(),
        ] {
            let mut buf = [7];
            assert_eq!(
                kernel
                    .call(mutable_read_after_store::CallArgs { buf: &mut buf })
                    .unwrap(),
                18
            );
            assert_eq!(buf, [11]);
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[test]
fn generic_functions_and_nested_calls_stage_before_codegen() {
    let compiled = scaled::build::<Inspect, i32, 3>(scaled::BuildArgs { settings: &1 })
        .unwrap()
        .into_inner();
    assert_eq!(scaled::<i32, 3>(&1, 7), 17);
    assert_eq!(compiled.0.arity(), 1);
    assert!(
        compiled
            .0
            .instructions()
            .iter()
            .any(|op| matches!(op, Instr::Bin(_, Binary::Mul, ..)))
    );
    let positive = adjusted::ir(true).unwrap();
    assert!(
        positive
            .instructions()
            .iter()
            .any(|op| matches!(op, Instr::Call { .. }))
    );
    let negative = adjusted::build::<Inspect>(adjusted::BuildArgs { positive: false })
        .unwrap()
        .into_inner();
    assert!(
        !negative
            .0
            .instructions()
            .iter()
            .any(|op| matches!(op, Instr::Call { .. }))
    );
}

#[test]
fn slice_kernel_compiles_the_complete_loop() {
    let batch = column::build::<Inspect>(column::BuildArgs { positive: true })
        .unwrap()
        .into_inner()
        .0;
    assert_eq!(batch.arity(), 0);
    assert_eq!(batch.buffer_kinds(), [BufferKind::Read, BufferKind::Write]);
    assert!(
        batch
            .instructions()
            .iter()
            .any(|op| matches!(op, Instr::Store { .. }))
    );
    let input = [1, 2];
    assert!(
        splice::validate_buffers(
            batch.buffer_kinds(),
            &[BufferArg::Read(&input), BufferArg::Read(&input)]
        )
        .is_err()
    );
}

#[test]
fn verification_rechecks_a_loop_after_a_delayed_predecessor() {
    let build = |initialize_delayed_arm| {
        let mut b = FunctionBuilder::new(2);
        for _ in 0..64 {
            b.local();
        }
        let value = b.local();
        let fast = b.label();
        let delayed = b.label();
        let delayed_tail = b.label();
        let head = b.label();
        let step = b.label();
        let done = b.label();
        b.branch(b.arg(0), fast, delayed);

        b.mark(fast);
        let seven = b.constant(7);
        b.assign(value, seven);
        b.jump(head);
        b.mark(delayed);
        b.jump(delayed_tail);
        b.mark(delayed_tail);
        if initialize_delayed_arm {
            let eleven = b.constant(11);
            b.assign(value, eleven);
        }
        b.jump(head);

        // The fast arm reaches the header before the delayed predecessor.
        b.mark(head);
        b.branch(b.arg(1), step, done);
        b.mark(step);
        let one = b.constant(1);
        let incremented = b.binary(Binary::Add, value, one);
        let remaining = b.binary(Binary::Sub, b.arg(1), one);
        b.assign(value, incremented);
        b.assign(b.arg(1), remaining);
        b.jump(head);
        b.mark(done);
        b.ret(value);
        b.finish()
    };

    assert!(build(false).unwrap_err().contains("uninitialized"));
    let function = build(true).unwrap();
    splice_testkit::each_backend!(B => {
        let code = B::compile(&function).unwrap();
        for (fast, initial) in [(1, 7), (0, 11)] {
            for count in [0, 2] {
                assert_eq!(code.call(&[fast, count], &mut []).unwrap(), initial + count);
            }
        }
    });
}

#[test]
fn verification_resolves_jumps_even_in_unreachable_code() {
    let body = || {
        let mut b = FunctionBuilder::new(1);
        b.ret(b.arg(0));
        b
    };
    let mut dead_tail = body();
    let unused = dead_tail.label();
    dead_tail.mark(unused);
    assert!(dead_tail.finish().is_ok());

    let mut dangling = body();
    let missing = dangling.label();
    dangling.jump(missing);
    assert!(dangling.finish().unwrap_err().contains("undefined label"));
}

mod directly_staged {
    use super::*;
    use splice::frontend::{Body, KernelBuilder};
    use splice::hir::{Buffer, Label, Reg};
    struct Config<'a>(&'a i32);

    #[kernel]
    fn leaf(#[comptime] config: Config<'_>, x: i32) -> i32 {
        let Config(bias) = config;
        x.wrapping_add(*bias)
    }
    #[kernel]
    fn conditional(#[comptime] enabled: bool, #[comptime] config: Config<'_>, x: i32) -> i32 {
        if enabled { leaf(config, x) } else { x }
    }

    /// A backend that accepts emitted operations only. Each nested body must
    /// get its own argument and label namespace, and is staged synchronously.
    #[derive(Default)]
    struct Trace {
        args: usize,
        next_reg: u32,
        next_label: u32,
        bodies: usize,
        constants: Vec<i32>,
    }
    impl Codegen for Trace {
        type Output = Self;
        fn compile_raw(_: &Function) -> Result<Self, String> {
            panic!("direct build materialized portable IR")
        }
        fn build(arity: usize, kinds: &[BufferKind], body: Body<'_>) -> Result<Self, String> {
            assert!(kinds.is_empty());
            let mut builder = Self {
                args: arity,
                next_reg: arity as u32,
                bodies: 1,
                ..Self::default()
            };
            body(&mut builder)?;
            Ok(builder)
        }
    }
    impl KernelBuilder for Trace {
        fn arg(&self, index: usize) -> Reg {
            assert!(index < self.args);
            Reg(index as u32)
        }
        fn local(&mut self) -> Reg {
            let r = Reg(self.next_reg);
            self.next_reg += 1;
            r
        }
        fn constant(&mut self, n: i32) -> Reg {
            self.constants.push(n);
            self.local()
        }
        fn buffer_len(&mut self, _: Buffer) -> Reg {
            unreachable!()
        }
        fn load(&mut self, _: Buffer, _: Reg) -> Reg {
            unreachable!()
        }
        fn store(&mut self, _: Buffer, _: Reg, _: Reg) {
            unreachable!()
        }
        fn binary(&mut self, _: Binary, _: Reg, _: Reg) -> Reg {
            self.local()
        }
        fn assign(&mut self, _: Reg, _: Reg) {}
        fn label(&mut self) -> Label {
            let l = Label(self.next_label);
            self.next_label += 1;
            l
        }
        fn mark(&mut self, _: Label) {}
        fn jump(&mut self, _: Label) {}
        fn branch(&mut self, _: Reg, _: Label, _: Label) {
            unreachable!("comptime condition must be staged")
        }
        fn ret(&mut self, _: Reg) {}
        fn call_kernel(
            &mut self,
            arity: usize,
            kinds: &[BufferKind],
            args: &[Reg],
            buffers: &[Buffer],
            body: Body<'_>,
        ) -> Result<Reg, String> {
            assert_eq!(args.len(), arity);
            assert_eq!(buffers.len(), kinds.len());
            let child = Self::build(arity, kinds, body)?;
            self.bodies += child.bodies;
            self.constants.extend(child.constants);
            Ok(self.local())
        }
    }
    #[test]
    fn nested_calls_stage_once_into_the_chosen_backend_with_borrowed_host_data() {
        let bias = 17;
        let skipped = conditional::build::<Trace>(conditional::BuildArgs {
            enabled: false,
            config: Config(&bias),
        })
        .unwrap();
        assert_eq!(skipped.bodies, 1);
        assert!(skipped.constants.is_empty());
        let built = conditional::build::<Trace>(conditional::BuildArgs {
            enabled: true,
            config: Config(&bias),
        })
        .unwrap();
        assert_eq!(built.bodies, 2);
        assert_eq!(built.constants, [bias]);
    }
}

#[derive(Clone, Copy)]
enum Arithmetic {
    Affine(i32, i32),
    Negate,
}
#[derive(Clone, Copy)]
enum Step {
    Eval(Arithmetic),
    SkipNegative,
    BreakPositive,
    ContinueOrStop,
    Stop,
    Finish,
    Unreachable,
}
fn unreachable_step() -> i32 {
    panic!("host code after a terminating step was staged")
}

#[kernel]
fn walk(#[comptime] steps: &[Step], mut value: i32) -> i32 {
    for step in steps {
        match step {
            Step::Eval(Arithmetic::Affine(scale, bias)) if *scale != 0 => {
                value = value.wrapping_mul(*scale).wrapping_add(*bias);
            }
            Step::Eval(Arithmetic::Affine(..)) => {}
            Step::Eval(Arithmetic::Negate) => {
                value = 0 - value;
            }
            Step::SkipNegative => {
                if value < 0 {
                    continue;
                }
            }
            Step::BreakPositive => {
                if value > 0 {
                    break;
                }
            }
            Step::ContinueOrStop => {
                if value < 0 {
                    continue;
                }
                break;
            }
            Step::Stop => break,
            Step::Finish => return value.wrapping_add(19),
            Step::Unreachable => {
                value += comptime!(unreachable_step());
            }
        }
        value += 2;
    }
    value.wrapping_mul(3)
}

#[test]
fn unrolled_host_plans_preserve_runtime_exits_and_prune_after_known_stops() {
    use Step::*;
    for plan in [
        &[][..],
        &[
            Eval(Arithmetic::Affine(3, 7)),
            Eval(Arithmetic::Affine(0, 99)),
            Eval(Arithmetic::Affine(-1, 3)),
        ],
        &[
            SkipNegative,
            Eval(Arithmetic::Affine(-2, 1)),
            ContinueOrStop,
            Eval(Arithmetic::Affine(1, 9)),
        ],
        &[Eval(Arithmetic::Negate), BreakPositive, Finish, Unreachable],
        &[Stop, Unreachable],
        &[Finish, Unreachable],
    ] {
        let portable = walk::ir(plan).unwrap();
        splice_testkit::each_backend!(B => {
            let code = walk::build::<B>(walk::BuildArgs { steps: plan }).unwrap();
            for x in [-23, 0, 7] {
                assert_eq!(code.call(walk::CallArgs { value: x }).unwrap(), walk(plan, x));
            }
        });
        // Iterator progress and enum dispatch finish at specialization time.
        let mut labels = std::collections::HashSet::new();
        for op in portable.instructions() {
            match op {
                Instr::Label(l) => {
                    labels.insert(*l);
                }
                Instr::Jump(l) => assert!(!labels.contains(l)),
                Instr::Branch { yes, no, .. } => {
                    assert!(!labels.contains(yes) && !labels.contains(no))
                }
                _ => {}
            }
        }
    }
}

#[kernel]
fn weighted(#[comptime] terms: [(i32, i32); 3], mut x: i32) -> i32 {
    for (scale, bias) in terms {
        x = x.wrapping_mul(scale).wrapping_add(bias);
    }
    x
}
#[test]
fn by_value_array_patterns_bind_comptime_operands() {
    let terms = [(2, 1), (-1, 0), (3, 4)];
    let ir = weighted::ir(terms).unwrap();
    assert_eq!(
        ir.label_count(),
        1,
        "only the kernel return label is needed"
    );
    splice_testkit::each_backend!(B => {
        let code = weighted::build::<B>(weighted::BuildArgs { terms }).unwrap();
        assert_eq!(
            code.call(weighted::CallArgs { x: 7 }).unwrap(),
            weighted(terms, 7)
        );
    });
}

#[test]
fn nested_loop_exits_and_generated_names_preserve_rust_bindings() {
    #[allow(non_upper_case_globals)]
    const builder: i32 = 5;

    #[kernel]
    #[expect(
        clippy::needless_range_loop,
        reason = "exercise the kernel's indexed range loop"
    )]
    fn nested(
        #[comptime] steps: &[i32],
        mut __splice_result: i32,
        __splice_builder: &mut [i32],
    ) -> i32 {
        for &loop_end in steps {
            for loop_next in 0..__splice_builder.len() {
                let value = match __splice_builder[loop_next] {
                    0 => continue,
                    1 if __splice_result < 0 => return __splice_result,
                    _ => loop_end + __splice_result,
                };
                __splice_builder[loop_next] = value;
                __splice_result += value;
                if value > builder {
                    break;
                }
            }
            if __splice_result < 0 {
                break;
            }
        }
        __splice_result
    }

    fn check<B: Codegen<Output: Executable>>() {
        let steps = [2, -3, 4];
        let ir = nested::ir(&steps).unwrap();
        let kernels = [
            nested::build::<B>(nested::BuildArgs { steps: &steps }).unwrap(),
            B::compile(&ir).unwrap(),
        ];
        for seed in [-5, -1, 0, 3] {
            let input = [0, 4, 1, 0, 2];
            let mut expected = input;
            let result = nested(&steps, seed, &mut expected);
            for kernel in &kernels {
                let mut actual = input;
                assert_eq!(
                    kernel
                        .call(nested::CallArgs {
                            __splice_result: seed,
                            __splice_builder: &mut actual,
                        })
                        .unwrap(),
                    result
                );
                assert_eq!(actual, expected);
            }
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

#[kernel]
fn slice_length(xs: &[i32]) -> i32 {
    xs.len() as i32
}

#[kernel]
fn first_lane(xs: &[i32]) -> i32 {
    splice::simd::Lane::<i32>::load(xs, 0).extract_wrapping(0)
}

#[kernel]
fn through_callee(xs: &[i32]) -> i32 {
    first_lane(xs) + 1
}

#[test]
fn only_operations_that_can_fail_need_a_status_channel() {
    assert!(!slice_length::ir().unwrap().needs_status());
    assert!(first_lane::ir().unwrap().needs_status());
    assert!(through_callee::ir().unwrap().needs_status());
    fn check<B: Codegen<Output: Executable>>() {
        let code = through_callee::build::<B>(through_callee::BuildArgs {}).unwrap();
        let short = [7];
        let error = code
            .call(through_callee::CallArgs { xs: &short })
            .unwrap_err();
        assert_eq!(error.to_string(), "kernel slice index out of bounds");
        let lanes = vec![7; 64];
        assert_eq!(code.call(through_callee::CallArgs { xs: &lanes }), Ok(8));
        let code = slice_length::build::<B>(slice_length::BuildArgs {}).unwrap();
        assert_eq!(code.call(slice_length::CallArgs { xs: &short }), Ok(1));
    }
    splice_testkit::each_backend!(B => check::<B>());
}
