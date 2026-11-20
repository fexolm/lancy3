use splice::hir::{Binary, BufferKind, FunctionBuilder, Instr};
use splice::{BufferArg, Codegen, Executable, Function, kernel};

// A backend used only to inspect the IR handed off by build().
struct Inspect(Function);

impl Codegen for Inspect {
    type Output = Self;

    fn compile(f: &Function) -> Result<Self, String> {
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
    assert_eq!(delivery(3, 8), 13);
    assert!(
        delivery::ir(3)
            .unwrap()
            .instructions()
            .iter()
            .any(|inst| matches!(inst, Instr::Const(_, 5)))
    );
}

trait Factor {
    fn factor(&self) -> i32;
}

impl Factor for i32 {
    fn factor(&self) -> i32 {
        *self
    }
}

#[kernel]
fn scaled<T, const OFFSET: i32>(#[comptime] settings: &T, row: i32) -> i32
where
    T: Factor,
{
    row.wrapping_mul(settings.factor()).wrapping_add(OFFSET)
}

#[kernel]
fn adjusted(#[comptime] positive: bool, row: i32) -> i32 {
    if positive {
        scaled::<_, 3>(&2, row)
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

        fn compile(_: &Function) -> Result<Self::Output, String> {
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

    let direct = overlapping_reads::build::<InspectKinds>().unwrap();
    let portable = overlapping_reads::ir().unwrap();
    assert_eq!(direct, portable.buffer_kinds());
    assert!(direct[0].may_alias(direct[1]));
    assert!(!direct[0].may_alias(direct[2]));
    assert!(!direct[2].may_alias(direct[1]));
    assert!(!direct[2].may_alias(direct[2]));
    let input = [17, 42];
    for kernel in [
        overlapping_reads::build::<splice_llvm::LlvmManualSimd>().unwrap(),
        splice_llvm::LlvmManualSimd::compile(&portable).unwrap(),
    ] {
        let mut output = [0];
        assert_eq!(
            kernel
                .call(
                    &[],
                    &mut [
                        BufferArg::Read(&input),
                        BufferArg::Read(&input[..1]),
                        BufferArg::Write(&mut output),
                    ],
                )
                .unwrap(),
            1
        );
        assert_eq!(output, [34]);
    }
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
    fn check_shared(kernel: impl Executable) {
        for (src, index, output, result) in [
            (&[7][..], 1, [7, 11], Some(14)),
            (&[7][..], 2, [7, 0], None),
            (&[][..], 1, [0, 0], None),
        ] {
            let mut dst = [0, 0];
            let actual = kernel.call(
                &[index],
                &mut [BufferArg::Read(src), BufferArg::Write(&mut dst)],
            );
            assert_eq!(actual.ok(), result);
            assert_eq!(dst, output);
        }
    }

    fn check_mutable(kernel: impl Executable) {
        let mut buf = [7];
        assert_eq!(kernel.call(&[], &mut [BufferArg::Write(&mut buf)]), Ok(18));
        assert_eq!(buf, [11]);
    }

    fn check<B: Codegen<Output: Executable>>() {
        check_shared(shared_read_after_store::build::<B>().unwrap());
        check_shared(B::compile(&shared_read_after_store::ir().unwrap()).unwrap());
        check_mutable(mutable_read_after_store::build::<B>().unwrap());
        check_mutable(B::compile(&mutable_read_after_store::ir().unwrap()).unwrap());
    }

    check::<splice_lancy::Lancy>();
    check::<splice_llvm::Llvm>();
}

#[test]
fn generic_functions_and_nested_calls_stage_before_codegen() {
    let compiled = scaled::build::<Inspect, _, 3>(&2).unwrap();
    assert_eq!(scaled::<_, 3>(&2, 7), 17);
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
    let negative = adjusted::build::<Inspect>(false).unwrap();
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
    let batch = column::build::<Inspect>(true).unwrap().0;
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
fn verification_rejects_a_branch_with_an_uninitialized_result() {
    let mut b = FunctionBuilder::new(1);
    let result = b.local();
    let assign = b.label();
    let done = b.label();
    b.branch(b.arg(0), assign, done);
    b.mark(assign);
    b.assign(result, b.arg(0));
    b.mark(done);
    b.ret(result);
    assert!(b.finish().unwrap_err().contains("uninitialized"));
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
    use std::cell::Cell;

    struct Config<'a>(&'a Cell<usize>);
    impl Config<'_> {
        fn bias(self) -> i32 {
            self.0.set(self.0.get() + 1);
            17
        }
    }

    #[kernel]
    fn leaf(#[comptime] config: Config<'_>, x: i32) -> i32 {
        x.wrapping_add(config.bias())
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
        fn compile(_: &Function) -> Result<Self, String> {
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
        fn immediate_condition(&self, _: Reg) -> Option<bool> {
            None
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
        let count = Cell::new(0);
        let skipped = conditional::build::<Trace>(false, Config(&count)).unwrap();
        assert_eq!((count.get(), skipped.bodies), (0, 1));
        let built = conditional::build::<Trace>(true, Config(&count)).unwrap();
        assert_eq!((count.get(), built.bodies), (1, 2));
        assert_eq!(built.constants, [17]);
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
                value += unreachable_step();
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
        let lancy = walk::build::<splice_lancy::Lancy>(plan).unwrap();
        let llvm = walk::build::<splice_llvm::Llvm>(plan).unwrap();
        for x in [-23, 0, 7] {
            let expected = walk(plan, x);
            assert_eq!(lancy.call(&[x], &mut []), Ok(expected));
            assert_eq!(llvm.call(&[x], &mut []), Ok(expected));
        }
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
    let code = weighted::build::<splice_lancy::Lancy>(terms).unwrap();
    assert_eq!(code.call(&[7], &mut []), Ok(weighted(terms, 7)));
    let code = weighted::build::<splice_llvm::Llvm>(terms).unwrap();
    assert_eq!(code.call(&[7], &mut []), Ok(weighted(terms, 7)));
}

#[test]
fn nested_loop_exits_and_generated_names_preserve_rust_bindings() -> splice::Result<()> {
    use splice_lancy::Lancy;
    use splice_llvm::Llvm;

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

    let steps = [2, -3, 4];
    let ir = nested::ir(&steps)?;
    let kernels: [&dyn Executable; 4] = [
        &nested::build::<Lancy>(&steps)?,
        &Lancy::compile(&ir)?,
        &nested::build::<Llvm>(&steps)?,
        &Llvm::compile(&ir)?,
    ];
    for seed in [-5, -1, 0, 3] {
        let input = [0, 4, 1, 0, 2];
        let mut expected = input;
        let result = nested(&steps, seed, &mut expected);
        for kernel in kernels {
            let mut actual = input;
            assert_eq!(
                kernel.call(&[seed], &mut [BufferArg::Write(&mut actual)])?,
                result
            );
            assert_eq!(actual, expected);
        }
    }
    Ok(())
}
