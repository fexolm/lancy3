use splice::frontend::Body;
use splice::hir::{Binary, Buffer, Label, Reg};
use splice::{BufferArg, BufferKind, Codegen, Executable};
use splice_llvm::Llvm;

#[test]
fn failed_nested_specialization_can_be_replaced_before_installation() {
    let kernel = Llvm::build(
        0,
        &[],
        Box::new(|root| {
            let failed = root.call_kernel(
                0,
                &[],
                &[],
                &[],
                Box::new(|child| {
                    child.call_kernel(
                        0,
                        &[],
                        &[],
                        &[],
                        Box::new(|leaf| {
                            let v = leaf.constant(3);
                            leaf.ret(v);
                            Ok(())
                        }),
                    )?;
                    Err("specialization rejected".into())
                }),
            );
            assert!(failed.is_err());
            let result = root.call_kernel(
                0,
                &[],
                &[],
                &[],
                Box::new(|child| {
                    let v = child.constant(7);
                    child.ret(v);
                    Ok(())
                }),
            )?;
            root.ret(result);
            Ok(())
        }),
    )
    .unwrap();
    assert_eq!(kernel.call(&[], &mut []), Ok(7));
}

#[test]
fn nested_bodies_keep_their_llvm_insertion_points_and_propagate_faults() {
    let kernel = Llvm::build(
        1,
        &[BufferKind::Write],
        Box::new(|f| {
            let index = f.arg(0);
            let input = f.constant(4);
            let value = f.call_kernel(
                1,
                &[],
                &[input],
                &[],
                Box::new(|child| {
                    let a = child.arg(0);
                    let product = child.call_kernel(
                        1,
                        &[],
                        &[a],
                        &[],
                        Box::new(|leaf| {
                            let n = leaf.arg(0);
                            let square = leaf.binary(Binary::Mul, n, n);
                            leaf.ret(square);
                            Ok(())
                        }),
                    )?;
                    let sum = child.binary(Binary::Add, product, a);
                    child.ret(sum);
                    Ok(())
                }),
            )?;
            let zero = f.constant(0);
            f.store(Buffer(0), zero, value);
            let loaded = f.call_kernel(
                1,
                &[BufferKind::Read],
                &[index],
                &[Buffer(0)],
                Box::new(|child| {
                    let v = child.load(Buffer(0), child.arg(0));
                    child.ret(v);
                    Ok(())
                }),
            )?;
            f.store(Buffer(0), zero, loaded);
            f.ret(loaded);
            Ok(())
        }),
    )
    .unwrap();
    let mut data = [0, 73];
    assert_eq!(
        kernel.call(&[1], &mut [BufferArg::Write(&mut data)]),
        Ok(73)
    );
    assert_eq!(data[0], 73);
    assert!(
        kernel
            .call(&[2], &mut [BufferArg::Write(&mut data)])
            .is_err()
    );
    assert_eq!(data[0], 20);
}

#[test]
fn direct_builder_rejects_uninitialized_values_on_a_backedge_and_diamond() {
    for bad_loop in [true, false] {
        let result = Llvm::build(
            1,
            &[],
            Box::new(move |b| {
                let test = b.arg(0);
                let result = b.local();
                let head = b.label();
                let write = b.label();
                let done = b.label();
                b.jump(head);
                b.mark(head);
                b.branch(test, write, done);
                b.mark(write);
                b.assign(result, test);
                b.jump(if bad_loop { head } else { done });
                b.mark(done);
                b.ret(result);
                Ok(())
            }),
        );
        assert!(result.err().unwrap().contains("uninitialized"));
    }
}

#[test]
fn direct_builder_checks_handles_control_flow_and_borrow_contracts() {
    let bad: Vec<Body<'_>> = vec![
        Box::new(|b| {
            b.ret(Reg(1234));
            Ok(())
        }),
        Box::new(|b| {
            b.jump(Label(1234));
            Ok(())
        }),
        Box::new(|b| {
            let l = b.label();
            b.jump(l);
            Ok(())
        }),
        Box::new(|b| {
            let l = b.label();
            b.mark(l);
            b.mark(l);
            Ok(())
        }),
        Box::new(|_| Ok(())),
        Box::new(|b| {
            let k = b.constant(0);
            b.store(Buffer(0), k, k);
            b.ret(k);
            Ok(())
        }),
        Box::new(|b| {
            let k = b.buffer_len(Buffer(4));
            b.ret(k);
            Ok(())
        }),
        Box::new(|b| {
            b.call_kernel(
                0,
                &[BufferKind::Write, BufferKind::Read],
                &[],
                &[Buffer(1), Buffer(1)],
                Box::new(|c| {
                    let n = c.constant(0);
                    c.ret(n);
                    Ok(())
                }),
            )?;
            Ok(())
        }),
    ];
    for body in bad {
        assert!(Llvm::build(0, &[BufferKind::Read, BufferKind::Write], body).is_err());
    }
}
