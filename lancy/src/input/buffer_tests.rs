use super::*;
use BufferKind::{Read, Write};

fn noop(kinds: Vec<BufferKind>) -> Function {
    let mut b = FunctionBuilder::with_buffers(0, kinds);
    let n = b.constant(7);
    b.ret(n);
    b.finish().unwrap()
}

#[test]
fn nested_calls_forward_buffers_and_keep_prior_stores_on_a_bounds_error() {
    let mut helper = FunctionBuilder::with_buffers(1, vec![Read, Write]);
    let zero = helper.constant(0);
    let x = helper.load(Buffer(0), zero);
    helper.store(Buffer(1), zero, x);
    // Its result is dead, but moving or deleting this load would change which
    // writes survive failure.
    helper.load(Buffer(0), helper.arg(0));
    let len = helper.buffer_len(Buffer(0));
    helper.store(Buffer(1), zero, len);
    helper.ret(len);
    let mut b = FunctionBuilder::with_buffers(1, vec![Write, Read]);
    let value = b
        .call_with_buffers(
            helper.finish().unwrap(),
            &[b.arg(0)],
            &[Buffer(1), Buffer(0)],
        )
        .unwrap();
    b.ret(value);
    let original = b.finish().unwrap();
    for f in [&original, &original.simplified().unwrap()] {
        assert!(f.interpret(&[1]).is_err());
        for index in [1, 2, -1] {
            let mut dst = [0, 99];
            let result = f.interpret_with_buffers(
                &[index],
                &mut [BufferArg::Write(&mut dst), BufferArg::Read(&[42, -1])],
            );
            if index == 1 {
                assert_eq!(result, Ok(2));
                assert_eq!(dst, [2, 99]);
            } else {
                assert!(result.unwrap_err().contains("out of bounds"));
                assert_eq!(dst, [42, 99]);
            }
        }
    }
}

#[test]
fn enforce_declared_permissions_and_forwarding_alias_contract() {
    let mut bad = FunctionBuilder::with_buffers(1, vec![Read]);
    bad.store(Buffer(0), bad.arg(0), bad.arg(0));
    bad.ret(bad.arg(0));
    assert!(bad.finish().is_err());
    let mut missing = FunctionBuilder::new(0);
    let len = missing.buffer_len(Buffer(5));
    missing.ret(len);
    assert!(missing.finish().is_err());
    let mut bad_index = FunctionBuilder::with_buffers(0, vec![Write]);
    let uninitialized = bad_index.local();
    let v = bad_index.load(Buffer(0), uninitialized);
    bad_index.ret(v);
    assert!(bad_index.finish().is_err());

    let mut b = FunctionBuilder::with_buffers(0, vec![Read, Write]);
    assert!(
        b.call_with_buffers(noop(vec![Write]), &[], &[Buffer(0)])
            .is_err()
    );
    for kinds in [vec![Write, Read], vec![Read, Write], vec![Write, Write]] {
        assert!(
            b.call_with_buffers(noop(kinds), &[], &[Buffer(1), Buffer(1)])
                .is_err()
        );
    }
    let value = b
        .call_with_buffers(noop(vec![Read, Read]), &[], &[Buffer(1), Buffer(1)])
        .unwrap();
    b.ret(value);
    let f = b.finish().unwrap();
    assert_eq!(
        f.interpret_with_buffers(&[], &mut [BufferArg::Read(&[]), BufferArg::Write(&mut [])]),
        Ok(7)
    );
    assert!(
        f.interpret_with_buffers(&[], &mut [BufferArg::Read(&[]), BufferArg::Read(&[])])
            .is_err()
    );
}

#[test]
fn known_scalar_can_remove_a_branch_but_preserves_a_buffer_call() {
    let mut write = FunctionBuilder::with_buffers(1, vec![Write]);
    let yes = write.label();
    let no = write.label();
    write.branch(write.arg(0), yes, no);
    write.mark(yes);
    let zero = write.constant(0);
    write.store(Buffer(0), zero, write.arg(0));
    write.mark(no);
    write.ret(write.arg(0));

    let mut b = FunctionBuilder::with_buffers(0, vec![Write]);
    let five = b.constant(5);
    b.call_with_buffers(write.finish().unwrap(), &[five], &[Buffer(0)])
        .unwrap();
    b.ret(five);
    let opt = b.finish().unwrap().simplified().unwrap();
    let callee = opt
        .instructions()
        .iter()
        .find_map(|op| match op {
            Instr::Call { function, .. } => Some(function),
            _ => None,
        })
        .unwrap();
    assert!(
        callee
            .instructions()
            .iter()
            .all(|op| !matches!(op, Instr::Branch { .. }))
    );
    let mut output = [0];
    assert_eq!(
        opt.interpret_with_buffers(&[], &mut [BufferArg::Write(&mut output)]),
        Ok(5)
    );
    assert_eq!(output, [5]);
}

#[test]
fn signed_min_max_at_i32_extremes() {
    for op in [Binary::Min, Binary::Max] {
        let mut b = FunctionBuilder::new(2);
        let val = b.binary(op, b.arg(0), b.arg(1));
        b.ret(val);
        let f = b.finish().unwrap();
        let jit = crate::ScalarKernel::compile(&f).unwrap();
        for (a, b) in [(i32::MAX, i32::MIN), (-7, -7), (-8, 2)] {
            let expected = if op == Binary::Min {
                a.min(b)
            } else {
                a.max(b)
            };
            assert_eq!(f.interpret(&[a, b]), Ok(expected));
            assert_eq!(jit.call(&[a, b]).unwrap(), expected);
        }
    }
}
