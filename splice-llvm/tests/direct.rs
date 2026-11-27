use splice_core::frontend::Body;
use splice_core::frontend::Signature;
use splice_core::hir::{Binary, Buffer, Label, Reg, ValueType};
use splice_core::hir::{Constant, ScalarType};
use splice_core::{BufferArg, BufferKind, BuildOptions, Codegen, Executable, Mode};
use splice_llvm::Llvm;

#[test]
fn failed_nested_specialization_can_be_replaced_before_installation() {
    let kernel = Llvm::build_with_options(
        Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[]),
        Box::new(|root| {
            let failed = root.call_kernel(
                Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[]),
                &[],
                &[],
                Box::new(|child| {
                    child.call_kernel(
                        Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[]),
                        &[],
                        &[],
                        Box::new(|leaf| {
                            let v = leaf.constant(Constant::from(3))?;
                            leaf.ret(v)?;
                            Ok(())
                        }),
                    )?;
                    Err(splice_core::BuildError::unsupported(
                        "specialization rejected",
                    ))
                }),
            );
            assert!(failed.is_err());
            let result = root.call_kernel(
                Signature::new(&[ScalarType::I32; 0], ScalarType::I32, &[]),
                &[],
                &[],
                Box::new(|child| {
                    let v = child.constant(Constant::from(7))?;
                    child.ret(v)?;
                    Ok(())
                }),
            )?;
            root.ret(result)?;
            Ok(())
        }),
        BuildOptions::host(Mode::Release).unwrap(),
    )
    .unwrap();
    assert_eq!(kernel.call(&[], &mut []), Ok(7));
}

#[test]
fn nested_bodies_keep_their_llvm_insertion_points_and_propagate_faults() {
    let kernel = Llvm::build_with_options(
        Signature::new(
            &[ScalarType::I32; 1],
            ScalarType::I32,
            &[BufferKind::WRITE_I32],
        ),
        Box::new(|f| {
            let index = f.arg(0)?;
            let input = f.constant(Constant::from(4))?;
            let value = f.call_kernel(
                Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
                &[input],
                &[],
                Box::new(|child| {
                    let a = child.arg(0)?;
                    let product = child.call_kernel(
                        Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
                        &[a],
                        &[],
                        Box::new(|leaf| {
                            let n = leaf.arg(0)?;
                            let square = leaf.binary(Binary::Mul, n, n)?;
                            leaf.ret(square)?;
                            Ok(())
                        }),
                    )?;
                    let sum = child.binary(Binary::Add, product, a)?;
                    child.ret(sum)?;
                    Ok(())
                }),
            )?;
            let zero = f.constant(Constant::from(0usize))?;
            f.store(Buffer(0), zero, value)?;
            let loaded = f.call_kernel(
                Signature::new(
                    &[ScalarType::I32; 1],
                    ScalarType::I32,
                    &[BufferKind::READ_I32],
                ),
                &[index],
                &[Buffer(0)],
                Box::new(|child| {
                    let at = child.cast(child.arg(0)?, ScalarType::Index)?;
                    let v = child.load(Buffer(0), at)?;
                    child.ret(v)?;
                    Ok(())
                }),
            )?;
            f.store(Buffer(0), zero, loaded)?;
            f.ret(loaded)?;
            Ok(())
        }),
        BuildOptions::host(Mode::Release).unwrap(),
    )
    .unwrap();
    let mut data = [0, 73];
    assert_eq!(
        kernel.call(&[1], &mut [BufferArg::write(&mut data)]),
        Ok(73)
    );
    assert_eq!(data[0], 73);
    assert!(
        kernel
            .call(&[2], &mut [BufferArg::write(&mut data)])
            .is_err()
    );
    assert_eq!(data[0], 20);
}

#[test]
fn direct_builder_rejects_uninitialized_values_on_a_backedge_and_diamond() {
    for bad_loop in [true, false] {
        let result = Llvm::build_with_options(
            Signature::new(&[ScalarType::I32; 1], ScalarType::I32, &[]),
            Box::new(move |b| {
                let test = b.arg(0)?;
                let result = b.local(ValueType::Scalar(ScalarType::I32))?;
                let head = b.label()?;
                let write = b.label()?;
                let done = b.label()?;
                b.jump(head)?;
                b.mark(head)?;
                let zero = b.constant(Constant::from(0))?;
                let nonzero = b.binary(Binary::Ne, test, zero)?;
                b.branch(nonzero, write, done)?;
                b.mark(write)?;
                b.assign(result, test)?;
                b.jump(if bad_loop { head } else { done })?;
                b.mark(done)?;
                b.ret(result)?;
                Ok(())
            }),
            BuildOptions::host(Mode::Release).unwrap(),
        );
        assert!(
            matches!(result.err().unwrap(), splice_core::BuildError::Verify { reason, .. } if reason.contains("uninitialized"))
        );
    }
}

#[test]
fn direct_builder_checks_handles_control_flow_and_borrow_contracts() {
    let bad: Vec<Body<'_>> = vec![
        Box::new(|b| {
            b.ret(Reg(1234))?;
            Ok(())
        }),
        Box::new(|b| {
            b.jump(Label(1234))?;
            Ok(())
        }),
        Box::new(|b| {
            let l = b.label()?;
            b.jump(l)?;
            Ok(())
        }),
        Box::new(|b| {
            let l = b.label()?;
            b.mark(l)?;
            b.mark(l)?;
            Ok(())
        }),
        Box::new(|_| Ok(())),
        Box::new(|b| {
            let k = b.constant(Constant::from(0))?;
            b.store(Buffer(0), k, k)?;
            b.ret(k)?;
            Ok(())
        }),
        Box::new(|b| {
            let k = b.buffer_len(Buffer(4))?;
            b.ret(k)?;
            Ok(())
        }),
        Box::new(|b| {
            b.call_kernel(
                Signature::new(
                    &[ScalarType::I32; 0],
                    ScalarType::I32,
                    &[BufferKind::WRITE_I32, BufferKind::READ_I32],
                ),
                &[],
                &[Buffer(1), Buffer(1)],
                Box::new(|c| {
                    let n = c.constant(Constant::from(0))?;
                    c.ret(n)?;
                    Ok(())
                }),
            )?;
            Ok(())
        }),
    ];
    for body in bad {
        assert!(
            Llvm::build_with_options(
                Signature::new(
                    &[ScalarType::I32; 0],
                    ScalarType::I32,
                    &[BufferKind::READ_I32, BufferKind::WRITE_I32]
                ),
                body,
                BuildOptions::host(Mode::Release).unwrap()
            )
            .is_err()
        );
    }
}
