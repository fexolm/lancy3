use super::*;

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
    let f = lancy::ScalarKernel::compile(&graph).unwrap();
    let mut out = [0];
    let error = f
        .call_with_buffers(
            &[27],
            &mut [
                lancy::BufferArg::Read(&[]),
                lancy::BufferArg::Write(&mut out),
            ],
        )
        .unwrap_err();
    assert_eq!(error, "kernel integer division by zero");
    assert_eq!(out, [27]);
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
