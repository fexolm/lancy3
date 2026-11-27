use splice::__private::{ContextArguments, StaticRustSymbol};
use splice::frontend::{KernelBuilder, Signature, materialize};
use splice::hir::{Buffer, ContextSlice, Instr, RustTarget};
use splice::hir::{Constant, ScalarType};
use splice::{BufferArg, BufferKind, BuildOptions, Codegen, Executable, KernelScalar, Mode};
use std::sync::atomic::{AtomicUsize, Ordering};

fn empty(_: ContextArguments<'_>) -> u64 {
    19
}

static EMPTY: StaticRustSymbol = StaticRustSymbol::new("empty", 0, 0, empty);

#[test]
fn static_packets_can_have_no_operands() {
    fn body(builder: &mut dyn KernelBuilder) -> splice::Result<()> {
        let result = builder.rust_call(RustTarget::Static(&EMPTY), &[], &[], ScalarType::I32)?;
        builder.ret(result)?;
        Ok(())
    }

    fn check<B: Codegen<Output: Executable>>() {
        let ir = materialize(
            Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[]),
            Box::new(body),
            BuildOptions::host(Mode::Release).unwrap(),
        )
        .unwrap();
        for kernel in [
            B::build_with_options(
                Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[]),
                Box::new(body),
                BuildOptions::host(Mode::Release).unwrap(),
            )
            .unwrap(),
            B::compile(&ir).unwrap(),
        ] {
            assert_eq!(splice_testkit::call_i32(&kernel, &[], &mut []).unwrap(), 19);
        }
    }

    splice_testkit::each_backend!(B => check::<B>());
}

static SEEN_INPUT: AtomicUsize = AtomicUsize::new(0);

fn modify(mut args: ContextArguments<'_>) -> u64 {
    let word = i32::from_word(args.word(0));
    let input = args.read::<u16>();
    SEEN_INPUT.store(input.as_ptr() as usize, Ordering::Relaxed);
    let output = args.write::<i32>();
    let value = word.wrapping_add(i32::from(input[0]));
    output[0] = value;
    output[1] = word;
    value.to_word()
}

static MODIFY: StaticRustSymbol = StaticRustSymbol::new("modify", 1, 2, modify);
const READ_U16: BufferKind = BufferKind::new(splice::simd::Element::U16, false);
const PARENT_KINDS: [BufferKind; 2] = [BufferKind::WRITE_I32, READ_U16];
const CHILD_KINDS: [BufferKind; 2] = [READ_U16, BufferKind::WRITE_I32];
const SHARED_INPUT: ContextSlice = view(0, false);
const MUTABLE_OUTPUT: ContextSlice = view(1, true);

const fn view(buffer: u32, writable: bool) -> ContextSlice {
    ContextSlice {
        buffer: Buffer(buffer),
        writable,
    }
}

fn modify_child(builder: &mut dyn KernelBuilder) -> splice::Result<()> {
    let result = builder.rust_call(
        RustTarget::Static(&MODIFY),
        &[builder.arg(0)?],
        &[SHARED_INPUT, MUTABLE_OUTPUT],
        ScalarType::I32,
    )?;
    builder.ret(result)?;
    Ok(())
}

fn forward_slices(builder: &mut dyn KernelBuilder) -> splice::Result<()> {
    let first = builder.constant(Constant::from(0usize))?;
    let marker = builder.constant(Constant::from(91))?;
    builder.store(Buffer(0), first, marker)?;
    let result = builder.call_kernel(
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &CHILD_KINDS),
        &[builder.arg(0)?],
        &[Buffer(1), Buffer(0)],
        Box::new(modify_child),
    )?;
    let one = builder.constant(Constant::from(1))?;
    let next = builder.binary(splice::hir::Binary::Add, result, one)?;
    builder.store(Buffer(0), first, next)?;
    builder.ret(result)?;
    Ok(())
}

#[test]
fn static_packets_keep_nested_slice_borrows_and_effect_order() {
    let ir = materialize(
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &PARENT_KINDS),
        Box::new(forward_slices),
        BuildOptions::host(Mode::Release).unwrap(),
    )
    .unwrap();
    let child = ir
        .instructions()
        .iter()
        .find_map(|op| match op {
            Instr::Call { function, .. } => Some(function),
            _ => None,
        })
        .unwrap();
    assert!(child.instructions().iter().any(|op| matches!(op,
        Instr::RustCall { target: RustTarget::Static(symbol), slices, .. }
        if symbol.name() == "modify" && slices == &[SHARED_INPUT, MUTABLE_OUTPUT]
    )));
    #[cfg(feature = "llvm")]
    {
        let text = splice::llvm::to_llvm_ir(&ir, &ir.options().target).unwrap();
        assert!(!text.contains("nocapture") && !text.contains("captures(none)"));
        assert!(text.contains("readonly") && text.contains("noalias"));
    }

    fn check<B: Codegen<Output: Executable>>(ir: &splice::Function) {
        for kernel in [
            B::build_with_options(
                Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &PARENT_KINDS),
                Box::new(forward_slices),
                BuildOptions::host(Mode::Release).unwrap(),
            )
            .unwrap(),
            B::compile(ir).unwrap(),
        ] {
            let input = [7_u16];
            let mut short = [0];
            assert!(
                splice_testkit::call_i32(
                    &kernel,
                    &[5],
                    &mut [BufferArg::write(&mut short), BufferArg::read::<u16>(&input)]
                )
                .is_err()
            );
            assert_eq!(short, [12]);
            assert_eq!(SEEN_INPUT.load(Ordering::Relaxed), input.as_ptr() as usize);

            let mut output = [0, 0];
            assert_eq!(
                splice_testkit::call_i32(
                    &kernel,
                    &[-10],
                    &mut [
                        BufferArg::write(&mut output),
                        BufferArg::read::<u16>(&input)
                    ]
                )
                .unwrap(),
                -3
            );
            assert_eq!(output, [-2, -10]);
        }
    }

    splice_testkit::each_backend!(B => check::<B>(&ir));
}

#[test]
fn invalid_static_packet_operands_are_rejected_before_native_codegen() {
    use splice::hir::Reg;
    static EMPTY_CONTEXT: splice::__private::ContextSignature =
        splice::__private::ContextSignature::new(&[]);
    let kinds = [
        CHILD_KINDS[0],
        CHILD_KINDS[1],
        BufferKind::context(&EMPTY_CONTEXT),
    ];
    type Invalid = fn(&mut dyn KernelBuilder) -> splice::Result<Reg>;
    let cases: [Invalid; 6] = [
        |b| {
            b.rust_call(
                RustTarget::Static(&MODIFY),
                &[],
                &[SHARED_INPUT, MUTABLE_OUTPUT],
                ScalarType::I32,
            )
        },
        |b| {
            b.rust_call(
                RustTarget::Static(&MODIFY),
                &[b.arg(0)?],
                &[MUTABLE_OUTPUT],
                ScalarType::I32,
            )
        },
        |b| {
            b.rust_call(
                RustTarget::Static(&MODIFY),
                &[b.arg(0)?],
                &[SHARED_INPUT, view(3, true)],
                ScalarType::I32,
            )
        },
        |b| {
            b.rust_call(
                RustTarget::Static(&MODIFY),
                &[b.arg(0)?],
                &[view(2, false), MUTABLE_OUTPUT],
                ScalarType::I32,
            )
        },
        |b| {
            b.rust_call(
                RustTarget::Static(&MODIFY),
                &[b.arg(0)?],
                &[SHARED_INPUT, view(0, true)],
                ScalarType::I32,
            )
        },
        |b| {
            b.rust_call(
                RustTarget::Static(&MODIFY),
                &[b.arg(0)?],
                &[view(1, false), MUTABLE_OUTPUT],
                ScalarType::I32,
            )
        },
    ];
    for bad in cases {
        let body = move |builder: &mut dyn KernelBuilder| {
            let result = bad(builder)?;
            builder.ret(result)?;
            Ok(())
        };
        assert!(
            materialize(
                Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &kinds),
                Box::new(body),
                BuildOptions::host(Mode::Release).unwrap()
            )
            .is_err()
        );
        splice_testkit::each_backend!(B => assert!(
            B::build_with_options(Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &kinds), Box::new(body), BuildOptions::host(Mode::Release).unwrap()).is_err()
        ));
    }
}

#[test]
fn static_rust_adapters_cannot_reinterpret_a_slice_element_type() {
    const KINDS: [BufferKind; 2] = [BufferKind::READ_I32, BufferKind::WRITE_I32];
    let ir = materialize(
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &KINDS),
        Box::new(modify_child),
        BuildOptions::host(Mode::Release).unwrap(),
    )
    .unwrap();

    fn check<B: Codegen<Output: Executable>>(ir: &splice::Function) {
        for kernel in [
            B::build_with_options(
                Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &KINDS),
                Box::new(modify_child),
                BuildOptions::host(Mode::Release).unwrap(),
            )
            .unwrap(),
            B::compile(ir).unwrap(),
        ] {
            let input = [12_i32];
            let mut output = [-1, -2];
            assert!(
                splice_testkit::call_i32(
                    &kernel,
                    &[3],
                    &mut [BufferArg::read(&input), BufferArg::write(&mut output)]
                )
                .is_err()
            );
            assert_eq!(output, [-1, -2]);
        }
    }

    splice_testkit::each_backend!(B => check::<B>(&ir));
}
