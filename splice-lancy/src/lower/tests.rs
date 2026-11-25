use super::*;
use splice_core::KernelScalar;
use splice_core::hir::{Constant, ScalarType};
mod simd;

/// Stage `body` directly with the host vector width, as `Lancy::stage` does.
fn build(signature: Signature<'_>, body: Body<'_>) -> splice_core::Result<Function> {
    let body = splice_core::validate::validated(signature, body);
    stage(signature, body, options().target.vector_bytes(), None).map(|(function, _)| function)
}

/// This machine's `SPLICE_SIMD` target in the mode of this test build.
fn options() -> splice_core::BuildOptions {
    splice_core::BuildOptions::host(splice_core::Mode::for_profile(cfg!(debug_assertions))).unwrap()
}

#[test]
fn context_descriptors_reject_other_signatures_and_slice_operations() {
    use splice_core::{
        BufferArg, Codegen, ContextFrame, ContextSignature, Executable, RuntimeContext,
    };
    static CHILD: ContextSignature = ContextSignature::new(&[1]);
    static FIRST: ContextSignature = ContextSignature::with_children(&[2], &[&CHILD]);
    static SECOND: ContextSignature = ContextSignature::with_children(&[2], &[&CHILD]);
    const KINDS: [BufferKind; 1] = [BufferKind::context(&FIRST)];

    fn good(b: &mut dyn KernelBuilder) -> splice_core::Result<()> {
        let operation = ContextFrame::root(&FIRST).child(0)?.operation(0)?;
        let result = b.rust_call(
            RustTarget::Context {
                buffer: Buffer(0),
                operation,
            },
            &[b.arg(0)?],
            &[],
        )?;
        b.ret(result)?;
        Ok(())
    }

    use splice_core::hir::check_forwarding;
    assert!(check_forwarding(&KINDS, &[BufferKind::Read], &[Buffer(0)]).is_err());
    assert!(check_forwarding(&[BufferKind::Write], &KINDS, &[Buffer(0)]).is_err());
    assert!(check_forwarding(&KINDS, &[KINDS[0]; 2], &[Buffer(0); 2]).is_err());
    assert!(check_forwarding(&KINDS, &KINDS, &[Buffer(0)]).is_ok());

    type Invalid = fn(&mut dyn KernelBuilder) -> splice_core::Result<Reg>;
    let invalid: [Invalid; 5] = [
        |b| {
            b.rust_call(
                RustTarget::Context {
                    buffer: Buffer(0),
                    operation: 2,
                },
                &[b.arg(0)?],
                &[],
            )
        },
        |b| {
            b.rust_call(
                RustTarget::Context {
                    buffer: Buffer(0),
                    operation: 0,
                },
                &[],
                &[],
            )
        },
        |b| {
            b.rust_call(
                RustTarget::Context {
                    buffer: Buffer(1),
                    operation: 0,
                },
                &[b.arg(0)?],
                &[],
            )
        },
        |b| b.load(Buffer(0), b.arg(0)?),
        |b| b.buffer_len(Buffer(0)),
    ];
    for bad in invalid {
        let body = |b: &mut dyn KernelBuilder| {
            let result = bad(b)?;
            b.ret(result)?;
            Ok(())
        };
        assert!(
            build(
                Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &KINDS),
                Box::new(body)
            )
            .is_err()
        );
        assert!(
            splice_core::frontend::materialize(
                Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &KINDS),
                Box::new(body),
                options()
            )
            .is_err()
        );
    }

    let direct = crate::Lancy::build_with_options(
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &KINDS),
        Box::new(good),
        splice_core::BuildOptions::host(splice_core::Mode::Release).unwrap(),
    )
    .unwrap();
    let ir = splice_core::frontend::materialize(
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &KINDS),
        Box::new(good),
        options(),
    )
    .unwrap();
    let portable = crate::Lancy::compile(&ir).unwrap();
    for kernel in [direct, portable] {
        let mut unreachable =
            |_, _: splice_core::ContextArguments<'_>| panic!("wrong context callback was called");
        let mut wrong = RuntimeContext::new(&SECOND, &mut unreachable);
        assert!(
            kernel
                .call(&[7], &mut [BufferArg::context(&mut wrong)])
                .is_err()
        );
        let mut slice = [0];
        assert!(
            kernel
                .call(&[7], &mut [BufferArg::write(&mut slice)])
                .is_err()
        );

        let (mut value, mut calls) = (3, 0);
        let mut invoke = |ordinal, args: splice_core::ContextArguments<'_>| {
            assert_eq!(ordinal, 1);
            calls += 1;
            value += args.word(0);
            value
        };
        let mut context = RuntimeContext::new(&FIRST, &mut invoke);
        assert_eq!(
            kernel
                .call(&[7], &mut [BufferArg::context(&mut context)])
                .unwrap(),
            10
        );
        assert_eq!((value, calls), (10, 1));
    }
}

#[test]
fn context_slice_operands_require_valid_nonaliasing_descriptors() {
    use splice_core::ContextSignature;
    static SIGNATURE: ContextSignature = ContextSignature::with_borrows(&[0], &[2], &[]);
    let kinds = [
        BufferKind::context(&SIGNATURE),
        BufferKind::Read,
        BufferKind::Write,
    ];
    let view = |buffer, writable| ContextSlice {
        buffer: Buffer(buffer),
        writable,
    };
    let cases: &[(&[ContextSlice], bool)] = &[
        (&[view(1, false)], false),
        (&[view(0, false), view(1, false)], false),
        (&[view(1, false), view(3, false)], false),
        (&[view(1, true), view(2, false)], false),
        (&[view(2, true), view(2, false)], false),
        (&[view(2, false), view(2, true)], false),
        (&[view(2, false), view(2, false)], true),
        (&[view(1, false), view(2, true)], true),
    ];
    for &(slices, valid) in cases {
        let body = |builder: &mut dyn KernelBuilder| {
            let result = builder.rust_call(
                RustTarget::Context {
                    buffer: Buffer(0),
                    operation: 0,
                },
                &[],
                slices,
            )?;
            builder.ret(result)?;
            Ok(())
        };
        assert_eq!(
            build(
                Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &kinds),
                Box::new(body)
            )
            .is_ok(),
            valid,
            "{slices:?}"
        );
        assert_eq!(
            splice_core::frontend::materialize(
                Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &kinds),
                Box::new(body),
                options()
            )
            .is_ok(),
            valid,
            "{slices:?}"
        );
    }
}

#[test]
fn static_rust_packets_carry_words_in_order_in_both_staging_paths() {
    use splice_core::StaticRustSymbol;
    static THREE: StaticRustSymbol = StaticRustSymbol::new("three", 0, 0, |_| 3);
    static DIGITS: StaticRustSymbol = StaticRustSymbol::new("digits", 3, 0, |args| {
        (0..3).fold(0, |n, i| n * 10 + args.word(i))
    });
    fn body(b: &mut dyn KernelBuilder) -> splice_core::Result<()> {
        let three = b.rust_call(RustTarget::Static(&THREE), &[], &[])?;
        let (one, two) = (
            b.constant(Constant::from(1))?,
            b.constant(Constant::from(2))?,
        );
        let digits = b.rust_call(RustTarget::Static(&DIGITS), &[one, two, b.arg(0)?], &[])?;
        let total = b.binary(Binary::Add, three, digits)?;
        b.ret(total)?;
        Ok(())
    }

    use splice_core::{Codegen, Executable};
    let signature = Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]);
    let direct = crate::Lancy::build_with_options(signature, Box::new(body), options()).unwrap();
    let ir = splice_core::frontend::materialize(signature, Box::new(body), options()).unwrap();
    let portable = crate::Lancy::compile(&ir).unwrap();
    for kernel in [direct, portable] {
        assert_eq!(kernel.call(&[5], &mut []), Ok(128));
        assert_eq!(kernel.call(&[-5_i32 as u32 as u64], &mut []), Ok(118));
    }
}

#[test]
fn reuses_shared_loads_across_writes_in_both_staging_paths() {
    fn body(b: &mut dyn KernelBuilder) -> splice_core::Result<()> {
        let zero = b.constant(Constant::from(0usize))?;
        let before = b.load(Buffer(0), zero)?;
        let at = b.cast(b.arg(0)?, ScalarType::Index)?;
        b.store(Buffer(1), at, before)?;
        let zero = b.constant(Constant::from(0usize))?;
        let after = b.load(Buffer(0), zero)?;
        let sum = b.binary(Binary::Add, before, after)?;
        b.ret(sum)?;
        Ok(())
    }

    for input in [BufferKind::Read, BufferKind::Write] {
        let kinds = [input, BufferKind::Write];
        let direct = build(
            Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &kinds),
            Box::new(body),
        )
        .unwrap();
        let portable = splice_core::frontend::materialize(
            Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &kinds),
            Box::new(body),
            options(),
        )
        .unwrap();
        for graph in [direct, lower(&portable, &options().target).unwrap()] {
            let count = graph
                .instructions(graph.entry())
                .filter(|&i| matches!(graph.op(i), Op::Load(..)))
                .count();
            assert_eq!(count, if input.writable() { 2 } else { 1 });
        }
    }
}

#[test]
fn portable_calls_share_one_lowered_body_and_preserve_argument_reassignment() {
    use splice_core::hir::FunctionBuilder;

    let mut leaf = FunctionBuilder::new(
        options(),
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
    );
    let one = leaf.constant(Constant::from(1)).unwrap();
    let value = leaf.binary(Binary::Add, leaf.arg(0).unwrap(), one).unwrap();
    leaf.ret(value).unwrap();
    let leaf = Arc::new(leaf.finish().unwrap());

    let mut caller = FunctionBuilder::new(
        options(),
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
    );
    let arg = caller.arg(0).unwrap();
    let first = caller.call_function(&leaf, &[arg], &[]).unwrap();
    caller.assign(arg, first).unwrap();
    let second = caller.call_function(&leaf, &[arg], &[]).unwrap();
    caller.ret(second).unwrap();
    let graph = lower(&caller.finish().unwrap(), &options().target).unwrap();
    let calls: Vec<_> = graph
        .instructions(graph.entry())
        .filter_map(|i| match graph.op(i) {
            Op::Call(id) => Some(graph.call(*id).0),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2);
    assert!(Arc::ptr_eq(calls[0], calls[1]));
    let kernel =
        lancy::ScalarKernel::compile(&graph, &lancy::CompileOptions::new(options().target))
            .unwrap();
    assert_eq!(kernel.call(&[22]), Ok(24));
}

#[test]
fn backedge_swaps_live_names_on_the_selected_edge() {
    let graph = build(
        Signature::new(&[ScalarType::I32; 3], ScalarType::I32, &[]),
        Box::new(|b| {
            let (n, x, y) = (b.arg(0)?, b.arg(1)?, b.arg(2)?);
            let (head, body, exit) = (b.label()?, b.label()?, b.label()?);
            b.mark(head)?;
            let zero = b.constant(Constant::from(0))?;
            let more = b.binary(Binary::Ne, n, zero)?;
            b.branch(more, body, exit)?;
            b.mark(body)?;
            let saved = b.local(ValueType::Scalar(ScalarType::I32))?;
            b.assign(saved, x)?;
            b.assign(x, y)?;
            b.assign(y, saved)?;
            let one = b.constant(Constant::from(1))?;
            let next = b.binary(Binary::Sub, n, one)?;
            b.assign(n, next)?;
            b.jump(head)?;
            b.mark(exit)?;
            b.ret(x)?;
            Ok(())
        }),
    )
    .unwrap();
    let kernel =
        lancy::ScalarKernel::compile(&graph, &lancy::CompileOptions::new(options().target))
            .unwrap();
    for n in 0..10 {
        assert_eq!(
            kernel.call(&[n, 20, 91]),
            Ok(if n % 2 == 0 { 20 } else { 91 })
        );
    }
}

#[test]
fn backward_jump_into_previously_unreferenced_label_is_reachable() {
    let f = build(
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
        Box::new(|b| {
            let (early, late, dead) = (b.label()?, b.label()?, b.label()?);
            let _unused = b.label()?;
            b.jump(late)?;
            b.mark(early)?;
            let one = b.constant(Constant::from(1))?;
            let v = b.binary(Binary::Add, one, b.arg(0)?)?;
            b.ret(v)?;
            b.mark(dead)?;
            let uninitialized = b.local(ValueType::Scalar(ScalarType::I32))?;
            b.ret(uninitialized)?;
            b.mark(late)?;
            b.jump(early)?;
            Ok(())
        }),
    )
    .unwrap();
    assert_eq!(
        lancy::ScalarKernel::compile(&f, &lancy::CompileOptions::new(options().target))
            .unwrap()
            .call(&[41]),
        Ok(42)
    );
}

#[test]
fn reads_require_a_definition_on_every_reachable_edge() {
    let result = build(
        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
        Box::new(|b| {
            let (yes, join) = (b.label()?, b.label()?);
            let local = b.local(ValueType::Scalar(ScalarType::I32))?;
            let zero = b.constant(Constant::from(0))?;
            let taken = b.binary(Binary::Ne, b.arg(0)?, zero)?;
            b.branch(taken, yes, join)?;
            b.mark(yes)?;
            let val = b.constant(Constant::from(7))?;
            b.assign(local, val)?;
            b.mark(join)?;
            b.ret(local)?;
            Ok(())
        }),
    );
    assert!(
        matches!(result.unwrap_err(), splice_core::BuildError::Verify { reason, .. } if reason.contains("uninitialized"))
    );
}

#[test]
fn checked_nested_call_preserves_writes_before_failure() {
    let graph = build(
        Signature::new(
            &[ScalarType::I32; 1],
            ScalarType::I32,
            &[BufferKind::Read, BufferKind::Write],
        ),
        Box::new(|b| {
            let x = b.call_kernel(
                Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[BufferKind::Write]),
                &[b.arg(0)?],
                &[Buffer(1)],
                Box::new(|c| {
                    let first = c.constant(Constant::from(0usize))?;
                    c.store(Buffer(0), first, c.arg(0)?)?;
                    let zero = c.constant(Constant::from(0))?;
                    let x = c.binary(Binary::Div, c.arg(0)?, zero)?;
                    c.ret(x)?;
                    Ok(())
                }),
            )?;
            b.ret(x)?;
            Ok(())
        }),
    )
    .unwrap();
    use splice_core::Executable;
    let f = crate::LancyKernel::compile(
        Signature::new(
            &[ScalarType::I32; 1],
            ScalarType::I32,
            &[BufferKind::Read, BufferKind::Write],
        ),
        splice_core::BuildOptions::host(splice_core::Mode::Release).unwrap(),
        |_, _| Ok((graph, None)),
    )
    .unwrap();
    let mut out = [0];
    let error = f
        .call(
            &[27],
            &mut [
                splice_core::BufferArg::read::<i32>(&[]),
                splice_core::BufferArg::write(&mut out),
            ],
        )
        .unwrap_err();
    assert_eq!(
        error,
        splice_core::CallError::Trap(splice_core::Trap::DivisionByZero)
    );
    assert_eq!(out, [27]);
}

#[test]
fn lowered_call_graphs_retain_slice_contracts_for_out_of_line_calls() {
    use lancy::{Inlining, ScalarKernel};
    use splice_core::{BufferArg, Executable, kernel};

    #[kernel(crate = splice_core)]
    fn read_first(#[comptime] guard: bool, n: i32, xs: &[i32]) -> i32 {
        if guard && xs.is_empty() {
            return -1;
        }
        let mut sum = 0;
        let mut i = 0;
        while i < n {
            sum += xs[0];
            i += 1;
        }
        sum
    }
    #[kernel(crate = splice_core)]
    fn reader(#[comptime] guard: bool, n: i32, xs: &[i32]) -> i32 {
        read_first(guard, n, xs)
    }
    #[kernel(crate = splice_core)]
    fn write_first(n: i32, xs: &mut [i32]) -> i32 {
        let mut sum = 0;
        let mut i = 0;
        while i < n {
            sum += xs[0];
            xs[0] += 1;
            i += 1;
        }
        sum
    }
    #[kernel(crate = splice_core)]
    fn writer(n: i32, xs: &mut [i32]) -> i32 {
        write_first(n, xs)
    }

    fn out_of_line(ir: splice_core::Function, writable: bool) -> crate::LancyKernel {
        let root = crate::lower(&ir, &options().target).unwrap();
        let child = root
            .blocks()
            .flat_map(|block| root.instructions(block))
            .find_map(|inst| {
                let Op::Call(call) = root.op(inst) else {
                    return None;
                };
                Some(root.call(*call).0)
            })
            .unwrap();
        for function in [&root, child] {
            let [scalar, ptr, len] = function.params(function.entry()).try_into().unwrap();
            assert_eq!(function.slice_param(scalar), None);
            assert_eq!(
                function.slice_param(ptr),
                Some(hir::SliceParam {
                    ptr,
                    len,
                    element: 4,
                    writable
                })
            );
        }
        let code = ScalarKernel::compile(
            &root,
            &lancy::CompileOptions::host()
                .unwrap()
                .with_inlining(Inlining::Never),
        )
        .unwrap();
        crate::LancyKernel {
            code: splice_jit::Publish::new(false).finish(false, code).unwrap(),
            interface: splice_core::call::Interface::new(ir.signature()),
        }
    }
    for guard in [false, true] {
        for kernel in [
            out_of_line(reader::ir(guard).unwrap().into_inner(), false),
            reader::build::<crate::Lancy>(reader::BuildArgs { guard })
                .unwrap()
                .into_inner(),
        ] {
            assert_eq!(
                kernel.call(&[3], &mut [BufferArg::read::<i32>(&[7])]),
                Ok(21)
            );
            assert_eq!(
                kernel.call(&[0], &mut [BufferArg::read::<i32>(&[])]),
                Ok(if guard { (-1i32).to_word() } else { 0 })
            );
            let empty = kernel.call(&[1], &mut [BufferArg::read::<i32>(&[])]);
            if guard {
                assert_eq!(empty, Ok((-1i32).to_word()));
            } else {
                assert!(empty.is_err());
            }
        }
    }
    for kernel in [
        out_of_line(writer::ir().unwrap().into_inner(), true),
        writer::build::<crate::Lancy>(writer::BuildArgs {})
            .unwrap()
            .into_inner(),
    ] {
        let mut xs = [7];
        assert_eq!(kernel.call(&[2], &mut [BufferArg::write(&mut xs)]), Ok(15));
        assert_eq!(xs, [9]);
    }
}

#[test]
fn rejects_undefined_labels_and_aliased_writable_arguments() {
    let undefined = build(
        Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[]),
        Box::new(|b| {
            let missing = b.label()?;
            b.jump(missing)?;
            Ok(())
        }),
    );
    assert!(
        matches!(undefined.unwrap_err(), splice_core::BuildError::Verify { reason, .. } if reason.contains("undefined"))
    );
    let alias = build(
        Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[BufferKind::Write]),
        Box::new(|b| {
            let value = b.call_kernel(
                Signature::new(
                    &[ScalarType::I32; 0],
                    ScalarType::I32,
                    &[BufferKind::Write, BufferKind::Read],
                ),
                &[],
                &[Buffer(0), Buffer(0)],
                Box::new(|c| {
                    let n = c.constant(Constant::from(0))?;
                    c.ret(n)?;
                    Ok(())
                }),
            )?;
            b.ret(value)?;
            Ok(())
        }),
    );
    assert!(
        matches!(alias.unwrap_err(), splice_core::BuildError::Verify { reason, .. } if reason.contains("aliases"))
    );
}

/// The `Index` side of a comparison: an argument cast from `i32`, the slice
/// length or a constant.
#[derive(Clone, Copy, Debug)]
enum IndexOperand {
    Arg(usize),
    Len,
    Constant(u64),
}

fn index_operand(b: &mut dyn KernelBuilder, operand: IndexOperand) -> splice_core::Result<Reg> {
    match operand {
        IndexOperand::Arg(n) => b.cast(b.arg(n)?, ScalarType::Index),
        IndexOperand::Len => b.buffer_len(Buffer(0)),
        IndexOperand::Constant(bits) => b.constant(Constant::from(bits as usize)),
    }
}

#[test]
fn index_comparisons_on_i32_forms_match_unsigned_64_bit_order() {
    use splice_core::{BufferArg, Codegen, Executable};
    type Order = fn(u64, u64) -> bool;
    const COMPARISONS: [(Binary, Order); 6] = [
        (Binary::Eq, |a, b| a == b),
        (Binary::Ne, |a, b| a != b),
        (Binary::Lt, |a, b| a < b),
        (Binary::Le, |a, b| a <= b),
        (Binary::Gt, |a, b| a > b),
        (Binary::Ge, |a, b| a >= b),
    ];
    let constants = [0, 3, i32::MAX as u64, 1 << 31, u64::MAX];
    let operands = |arg| {
        [IndexOperand::Arg(arg), IndexOperand::Len]
            .into_iter()
            .chain(constants.map(IndexOperand::Constant))
    };
    let args = [i32::MIN, -1, 0, 2, 3, 4, i32::MAX];
    let xs = [7; 3];
    let value = |operand, arg: i32| match operand {
        IndexOperand::Arg(_) => arg as usize as u64,
        IndexOperand::Len => xs.len() as u64,
        IndexOperand::Constant(bits) => bits,
    };
    for (op, expected) in COMPARISONS {
        for lhs in operands(0) {
            for rhs in operands(1) {
                let body = move |b: &mut dyn KernelBuilder| {
                    let (x, y) = (index_operand(b, lhs)?, index_operand(b, rhs)?);
                    let compared = b.binary(op, x, y)?;
                    let result = b.cast(compared, ScalarType::I32)?;
                    b.ret(result)
                };
                let kernel = crate::Lancy::build_with_options(
                    Signature::new(&[ScalarType::I32; 2], ScalarType::I32, &[BufferKind::Read]),
                    Box::new(body),
                    options(),
                )
                .unwrap();
                for a in args {
                    for b in args {
                        let actual =
                            kernel.call(&[a, b].map(i32::to_word), &mut [BufferArg::read(&xs)]);
                        let want = expected(value(lhs, a), value(rhs, b));
                        assert_eq!(
                            actual,
                            Ok(want.to_word()),
                            "{lhs:?} {op:?} {rhs:?} at {a}, {b}"
                        );
                    }
                }
            }
        }
    }
}

/// `xs.len() < N` must stay a guard on the `i32` length parameter: Lancy's
/// bounds analysis removes checks of constant indices below `N` only then
/// (a 64-bit comparison cost `sql_plan` 13-17% in Ф4.1).
#[test]
fn index_guard_on_a_slice_length_compares_the_i32_length() {
    fn body(b: &mut dyn KernelBuilder) -> splice_core::Result<()> {
        let len = b.buffer_len(Buffer(0))?;
        let four = b.constant(Constant::from(4usize))?;
        let short = b.binary(Binary::Lt, len, four)?;
        let (fail, read) = (b.label()?, b.label()?);
        b.branch(short, fail, read)?;
        b.mark(fail)?;
        let minus_one = b.constant(Constant::from(-1))?;
        b.ret(minus_one)?;
        b.mark(read)?;
        let three = b.constant(Constant::from(3usize))?;
        let value = b.load(Buffer(0), three)?;
        b.ret(value)
    }
    let graph = build(
        Signature::new(&[], ScalarType::I32, &[BufferKind::Read]),
        Box::new(body),
    )
    .unwrap();
    let [_, len] = graph.params(graph.entry()).try_into().unwrap();
    let hir::Terminator::Branch { cond, .. } = *graph.terminator(graph.entry()) else {
        panic!("the entry ends in the guard");
    };
    let guard = graph
        .instructions(graph.entry())
        .find(|&i| graph.result(i) == Some(cond))
        .unwrap();
    let Op::Bin(hir::Binary::Lt, lhs, rhs) = *graph.op(guard) else {
        panic!("the guard is a comparison");
    };
    assert_eq!(lhs, len);
    assert_eq!(graph.value_type(rhs), hir::Type::I32);
}

/// An `Index` that reaches a block through a parameter keeps its `i32`
/// form: the load in the join indexes with that form instead of clamping
/// the 64-bit parameter again (`simd_matmul` paid a `min` + `cast` per use).
#[test]
fn index_parameters_carry_their_i32_form_across_blocks() {
    fn body(b: &mut dyn KernelBuilder) -> splice_core::Result<()> {
        let index = b.cast(b.arg(0)?, ScalarType::Index)?;
        let zero = b.constant(Constant::from(0))?;
        let positive = b.binary(Binary::Gt, b.arg(0)?, zero)?;
        let (left, right, join) = (b.label()?, b.label()?, b.label()?);
        b.branch(positive, left, right)?;
        b.mark(left)?;
        b.jump(join)?;
        b.mark(right)?;
        b.jump(join)?;
        b.mark(join)?;
        let value = b.load(Buffer(0), index)?;
        b.ret(value)
    }
    let graph = build(
        Signature::new(&[ScalarType::I32], ScalarType::I32, &[BufferKind::Read]),
        Box::new(body),
    )
    .unwrap();
    let clamps = graph
        .blocks()
        .flat_map(|b| graph.instructions(b))
        .filter(|&i| matches!(graph.op(i), Op::Bin(hir::Binary::Min, ..)))
        .count();
    assert_eq!(clamps, 0, "{graph}");
}
