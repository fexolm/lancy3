use super::*;
mod simd;

#[test]
fn context_descriptors_reject_other_signatures_and_slice_operations() {
    use splice::{BufferArg, Codegen, ContextFrame, ContextSignature, Executable, RuntimeContext};
    static CHILD: ContextSignature = ContextSignature::new(&[1]);
    static FIRST: ContextSignature = ContextSignature::with_children(&[2], &[&CHILD]);
    static SECOND: ContextSignature = ContextSignature::with_children(&[2], &[&CHILD]);
    const KINDS: [BufferKind; 1] = [BufferKind::context(&FIRST)];

    fn good(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        let operation = ContextFrame::root(&FIRST)?.child(0)?.operation(0)?;
        let result = b.rust_context_call(Buffer(0), operation, &[b.arg(0)])?;
        b.ret(result);
        Ok(())
    }

    use splice::hir::check_forwarding;
    assert!(check_forwarding(&KINDS, &[BufferKind::Read], &[Buffer(0)]).is_err());
    assert!(check_forwarding(&[BufferKind::Write], &KINDS, &[Buffer(0)]).is_err());
    assert!(check_forwarding(&KINDS, &[KINDS[0]; 2], &[Buffer(0); 2]).is_err());
    assert!(check_forwarding(&KINDS, &KINDS, &[Buffer(0)]).is_ok());

    type Invalid = fn(&mut dyn KernelBuilder) -> splice::Result<Reg>;
    let invalid: [Invalid; 5] = [
        |b| b.rust_context_call(Buffer(0), 2, &[b.arg(0)]),
        |b| b.rust_context_call(Buffer(0), 0, &[]),
        |b| b.rust_context_call(Buffer(1), 0, &[b.arg(0)]),
        |b| Ok(b.load(Buffer(0), b.arg(0))),
        |b| Ok(b.buffer_len(Buffer(0))),
    ];
    for bad in invalid {
        let body = |b: &mut dyn KernelBuilder| {
            let result = bad(b)?;
            b.ret(result);
            Ok(())
        };
        assert!(build(1, &KINDS, Box::new(body)).is_err());
        assert!(splice::frontend::materialize(1, &KINDS, Box::new(body)).is_err());
    }

    let direct =
        crate::Lancy::build_with_options(1, &KINDS, Box::new(good), splice::BuildOptions::Release)
            .unwrap();
    let ir = splice::frontend::materialize(1, &KINDS, Box::new(good)).unwrap();
    let portable = crate::Lancy::compile(&ir).unwrap();
    for kernel in [direct, portable] {
        let mut unreachable = |_, _, _| panic!("wrong context callback was called");
        let mut wrong = RuntimeContext::new(&SECOND, &mut unreachable);
        assert!(
            kernel
                .call(&[7], &mut [BufferArg::Context(&mut wrong)])
                .is_err()
        );
        let mut slice = [0];
        assert!(
            kernel
                .call(&[7], &mut [BufferArg::Write(&mut slice)])
                .is_err()
        );

        let (mut value, mut calls) = (3, 0);
        let mut invoke = |ordinal, word: i32, padding| {
            assert_eq!((ordinal, padding), (1, 0));
            calls += 1;
            value += word;
            value
        };
        let mut context = RuntimeContext::new(&FIRST, &mut invoke);
        assert_eq!(
            kernel
                .call(&[7], &mut [BufferArg::Context(&mut context)])
                .unwrap(),
            10
        );
        assert_eq!((value, calls), (10, 1));
    }
}

#[test]
fn context_slice_operands_require_valid_nonaliasing_descriptors() {
    use splice::ContextSignature;
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
            let result = builder.rust_context_call_with_slices(Buffer(0), 0, &[], slices)?;
            builder.ret(result);
            Ok(())
        };
        assert_eq!(
            build(0, &kinds, Box::new(body)).is_ok(),
            valid,
            "{slices:?}"
        );
        assert_eq!(
            splice::frontend::materialize(0, &kinds, Box::new(body)).is_ok(),
            valid,
            "{slices:?}"
        );
    }
}

#[test]
fn rust_imports_fill_unused_abi_arguments_with_zero_in_both_staging_paths() {
    extern "C" fn three(a: i32, b: i32) -> i64 {
        if (a, b) == (0, 0) { 3 } else { 5_i64 << 32 }
    }
    extern "C" fn double(a: i32, b: i32) -> i64 {
        if b == 0 {
            i64::from(a.wrapping_mul(2) as u32)
        } else {
            5_i64 << 32
        }
    }
    fn body(b: &mut dyn KernelBuilder) -> splice::Result<()> {
        let three = b.rust_call(
            RustCall {
                symbol: &splice::hir::RustSymbol {
                    name: "three",
                    resolve: || three,
                },
                arity: 0,
            },
            &[],
        )?;
        let double = b.rust_call(
            RustCall {
                symbol: &splice::hir::RustSymbol {
                    name: "double",
                    resolve: || double,
                },
                arity: 1,
            },
            &[b.arg(0)],
        )?;
        let total = b.binary(Binary::Add, three, double);
        b.ret(total);
        Ok(())
    }

    let direct = build(1, &[], Box::new(body)).unwrap();
    let portable = splice::frontend::materialize(1, &[], Box::new(body)).unwrap();
    for graph in [direct, lower(&portable).unwrap()] {
        let kernel = lancy::ScalarKernel::compile(&graph).unwrap();
        assert_eq!(kernel.call(&[5]), Ok(13));
        assert_eq!(kernel.call(&[-5]), Ok(-7));
    }
}

#[test]
fn reuses_shared_loads_across_writes_in_both_staging_paths() {
    fn body(b: &mut dyn KernelBuilder) -> Result<(), String> {
        let zero = b.constant(0);
        let before = b.load(Buffer(0), zero);
        b.store(Buffer(1), b.arg(0), before);
        let zero = b.constant(0);
        let after = b.load(Buffer(0), zero);
        let sum = b.binary(Binary::Add, before, after);
        b.ret(sum);
        Ok(())
    }

    for input in [BufferKind::Read, BufferKind::Write] {
        let kinds = [input, BufferKind::Write];
        let direct = build(1, &kinds, Box::new(body)).unwrap();
        let portable = splice::frontend::materialize(1, &kinds, Box::new(body)).unwrap();
        for graph in [direct, lower(&portable).unwrap()] {
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
    use splice::hir::FunctionBuilder;

    let mut leaf = FunctionBuilder::new(1);
    let one = leaf.constant(1);
    let value = leaf.binary(Binary::Add, leaf.arg(0), one);
    leaf.ret(value);
    let leaf = Arc::new(leaf.finish().unwrap());

    let mut caller = FunctionBuilder::new(1);
    let arg = caller.arg(0);
    let first = caller.call_shared(Arc::clone(&leaf), &[arg]);
    caller.assign(arg, first);
    let second = caller.call_shared(leaf, &[arg]);
    caller.ret(second);
    let graph = lower(&caller.finish().unwrap()).unwrap();
    let calls: Vec<_> = graph
        .instructions(graph.entry())
        .filter_map(|i| match graph.op(i) {
            Op::Call(id) => Some(graph.call(*id).0),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 2);
    assert!(Arc::ptr_eq(calls[0], calls[1]));
    let kernel = lancy::ScalarKernel::compile(&graph).unwrap();
    assert_eq!(kernel.call(&[22]), Ok(24));
}

#[test]
fn backedge_swaps_live_names_on_the_selected_edge() {
    let graph = build(
        3,
        &[],
        Box::new(|b| {
            let (n, x, y) = (b.arg(0), b.arg(1), b.arg(2));
            let (head, body, exit) = (b.label(), b.label(), b.label());
            b.mark(head);
            b.branch(n, body, exit);
            b.mark(body);
            let saved = b.local();
            b.assign(saved, x);
            b.assign(x, y);
            b.assign(y, saved);
            let one = b.constant(1);
            let next = b.binary(Binary::Sub, n, one);
            b.assign(n, next);
            b.jump(head);
            b.mark(exit);
            b.ret(x);
            Ok(())
        }),
    )
    .unwrap();
    let kernel = lancy::ScalarKernel::compile(&graph).unwrap();
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
        1,
        &[],
        Box::new(|b| {
            let (early, late, dead) = (b.label(), b.label(), b.label());
            let _unused = b.label();
            b.jump(late);
            b.mark(early);
            let one = b.constant(1);
            let v = b.binary(Binary::Add, one, b.arg(0));
            b.ret(v);
            b.mark(dead);
            let uninitialized = b.local();
            b.ret(uninitialized);
            b.mark(late);
            b.jump(early);
            Ok(())
        }),
    )
    .unwrap();
    assert_eq!(
        lancy::ScalarKernel::compile(&f).unwrap().call(&[41]),
        Ok(42)
    );
}

#[test]
fn reads_require_a_definition_on_every_reachable_edge() {
    let result = build(
        1,
        &[],
        Box::new(|b| {
            let (yes, join) = (b.label(), b.label());
            let local = b.local();
            b.branch(b.arg(0), yes, join);
            b.mark(yes);
            let val = b.constant(7);
            b.assign(local, val);
            b.mark(join);
            b.ret(local);
            Ok(())
        }),
    );
    assert!(result.unwrap_err().contains("uninitialized"));
}

#[test]
fn checked_nested_call_preserves_writes_before_failure() {
    let graph = build(
        1,
        &[BufferKind::Read, BufferKind::Write],
        Box::new(|b| {
            let x = b.call_kernel(
                1,
                &[BufferKind::Write],
                &[b.arg(0)],
                &[Buffer(1)],
                Box::new(|c| {
                    let zero = c.constant(0);
                    c.store(Buffer(0), zero, c.arg(0));
                    let x = c.binary(Binary::Div, c.arg(0), zero);
                    c.ret(x);
                    Ok(())
                }),
            )?;
            b.ret(x);
            Ok(())
        }),
    )
    .unwrap();
    use splice::Executable;
    let f =
        crate::LancyKernel::new(graph, 1, &[BufferKind::Read, BufferKind::Write], None).unwrap();
    let mut out = [0];
    let error = f
        .call(
            &[27],
            &mut [
                splice::BufferArg::Read(&[]),
                splice::BufferArg::Write(&mut out),
            ],
        )
        .unwrap_err();
    assert_eq!(error, "kernel integer division by zero");
    assert_eq!(out, [27]);
}

#[test]
fn lowered_call_graphs_retain_slice_contracts_for_out_of_line_calls() {
    use lancy::{Inlining, ScalarKernel};
    use splice::{BufferArg, Executable, kernel};

    #[kernel]
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
    #[kernel]
    fn reader(#[comptime] guard: bool, n: i32, xs: &[i32]) -> i32 {
        read_first(guard, n, xs)
    }
    #[kernel]
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
    #[kernel]
    fn writer(n: i32, xs: &mut [i32]) -> i32 {
        write_first(n, xs)
    }

    fn out_of_line(ir: splice::Function, writable: bool) -> crate::LancyKernel {
        let root = crate::lower(&ir).unwrap();
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
            assert_eq!(function.slice_param(ptr), Some((len, 4, writable)));
        }
        crate::LancyKernel {
            _debug: None,
            code: ScalarKernel::compile_with_inlining(&root, Inlining::Never).unwrap(),
            arity: ir.arity(),
            slices: ir.buffer_kinds().into(),
        }
    }
    for guard in [false, true] {
        for kernel in [
            out_of_line(reader::ir(guard).unwrap().into_inner(), false),
            reader::build::<crate::Lancy>(reader::BuildArgs { guard })
                .unwrap()
                .into_inner(),
        ] {
            assert_eq!(kernel.call(&[3], &mut [BufferArg::Read(&[7])]), Ok(21));
            assert_eq!(
                kernel.call(&[0], &mut [BufferArg::Read(&[])]),
                Ok(if guard { -1 } else { 0 })
            );
            let empty = kernel.call(&[1], &mut [BufferArg::Read(&[])]);
            if guard {
                assert_eq!(empty, Ok(-1));
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
        assert_eq!(kernel.call(&[2], &mut [BufferArg::Write(&mut xs)]), Ok(15));
        assert_eq!(xs, [9]);
    }
}

#[test]
fn rejects_undefined_labels_and_aliased_writable_arguments() {
    let undefined = build(
        0,
        &[],
        Box::new(|b| {
            let missing = b.label();
            b.jump(missing);
            Ok(())
        }),
    );
    assert!(undefined.unwrap_err().contains("undefined"));
    let alias = build(
        0,
        &[BufferKind::Write],
        Box::new(|b| {
            let value = b.call_kernel(
                0,
                &[BufferKind::Write, BufferKind::Read],
                &[],
                &[Buffer(0), Buffer(0)],
                Box::new(|c| {
                    let n = c.constant(0);
                    c.ret(n);
                    Ok(())
                }),
            )?;
            b.ret(value);
            Ok(())
        }),
    );
    assert!(alias.unwrap_err().contains("aliases"));
}
