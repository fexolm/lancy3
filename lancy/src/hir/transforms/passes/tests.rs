use super::{inline_all, simplify};
use crate::hir::{Binary, Function, Op, Terminator};
use crate::{CallError, Inlining, ScalarKernel};
use std::sync::Arc;

#[test]
fn repeated_calls_clone_loops_and_each_early_return_into_its_own_continuation() {
    let mut f = Function::scalar(1);
    let entry = f.entry();
    let x = f.params(entry)[0];
    let loop_ = f.create_block();
    let n = f.append_param(loop_);
    let early = f.create_block();
    let done = f.create_block();
    let zero = f.append(entry, Op::Const(0)).unwrap();
    let negative = f.append(entry, Op::Bin(Binary::Lt, x, zero)).unwrap();
    let yes = f.add_edge(early, &[]);
    let no = f.add_edge(loop_, &[x]);
    f.set_terminator(
        entry,
        Terminator::Branch {
            cond: negative,
            yes,
            no,
        },
    );
    f.set_terminator(early, Terminator::Return(x));
    let one = f.append(loop_, Op::Const(1)).unwrap();
    let next = f.append(loop_, Op::Bin(Binary::Sub, n, one)).unwrap();
    let yes = f.add_edge(loop_, &[next]);
    let no = f.add_edge(done, &[]);
    f.set_terminator(loop_, Terminator::Branch { cond: n, yes, no });
    f.set_terminator(done, Terminator::Return(one));
    let callee = Arc::new(f);
    let mut root = Function::scalar(2);
    let b = root.entry();
    let [x, y] = root.params(b).try_into().unwrap();
    let first = root.add_call(callee.clone(), &[x]);
    let a = root.append(b, Op::Call(first)).unwrap();
    let second = root.add_call(callee, &[y]);
    let c = root.append(b, Op::Call(second)).unwrap();
    let answer = root.append(b, Op::Bin(Binary::Sub, a, c)).unwrap();
    root.set_terminator(b, Terminator::Return(answer));

    let (source, ids) = root.verified_module().unwrap();
    let expanded = inline_all(&source, &ids).unwrap();
    assert_eq!(expanded.verified_module().unwrap().0.functions.len(), 1);
    for policy in [Inlining::Always, Inlining::Never] {
        let code = ScalarKernel::compile_with_inlining(&root, policy).unwrap();
        for (x, y) in [(7, -5), (-5, 7), (0, 1), (-3, -9)] {
            let expected = (if x < 0 { x } else { 1 }) - (if y < 0 { y } else { 1 });
            assert_eq!(code.call(&[x, y]), Ok(expected));
        }
    }
}

#[test]
fn non_returning_callee_removes_only_the_unreachable_continuation() {
    let mut forever = Function::scalar(0);
    let head = forever.create_block();
    forever.jump(forever.entry(), head, &[]);
    forever.jump(head, head, &[]);
    let mut root = Function::scalar(1);
    let b = root.entry();
    let x = root.params(b)[0];
    let yes = root.create_block();
    let no = root.create_block();
    let site = root.add_call(Arc::new(forever), &[]);
    let never = root.append(yes, Op::Call(site)).unwrap();
    root.set_terminator(yes, Terminator::Return(never));
    root.set_terminator(no, Terminator::Return(x));
    let yes = root.add_edge(yes, &[]);
    let no = root.add_edge(no, &[]);
    root.set_terminator(b, Terminator::Branch { cond: x, yes, no });
    let (source, ids) = root.verified_module().unwrap();
    let expanded = inline_all(&source, &ids).unwrap();
    expanded.verify().unwrap();
    assert_eq!(
        expanded
            .blocks()
            .filter(|&b| matches!(expanded.terminator(b), Terminator::Return(_)))
            .count(),
        1
    );
    assert_eq!(ScalarKernel::compile(&root).unwrap().call(&[0]), Ok(0));
}

#[test]
fn remove_dominated_checks_but_keep_sibling_checks_and_unused_trapping_math() {
    let mut f = Function::scalar(3);
    let entry = f.entry();
    let [cond, i, n] = f.params(entry).try_into().unwrap();
    let a = f.create_block();
    let b = f.create_block();
    let end = f.create_block();
    let yes = f.add_edge(a, &[]);
    let no = f.add_edge(b, &[]);
    f.set_terminator(entry, Terminator::Branch { cond, yes, no });
    f.append(a, Op::BoundsCheck(i, n));
    f.append(a, Op::BoundsCheck(i, n));
    f.jump(a, end, &[]);
    f.jump(b, end, &[]);
    f.append(end, Op::BoundsCheck(i, n));
    let zero = f.append(end, Op::Const(0)).unwrap();
    f.append(end, Op::Bin(Binary::Div, n, zero));
    f.set_terminator(end, Terminator::Return(zero));
    simplify(&mut f).unwrap();
    f.verify().unwrap();
    assert_eq!(f.instructions(a).count(), 1);
    assert_eq!(
        f.instructions(end)
            .filter(|&i| matches!(f.op(i), Op::BoundsCheck(..)))
            .count(),
        1
    );
    let kernel = ScalarKernel::compile(&f).unwrap();
    for cond in [0, 1] {
        assert_eq!(kernel.call(&[cond, 1, 1]), Err(CallError::OutOfBounds));
        assert_eq!(kernel.call(&[cond, 0, 1]), Err(CallError::DivisionByZero));
    }
}
