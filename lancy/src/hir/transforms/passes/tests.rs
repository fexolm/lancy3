use super::{inline_all, simplify};
use crate::hir::{Binary, Function, Op, Terminator};
use crate::{CallError, Inlining, Kernel, Trap};
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

    let source = root.verified_module().unwrap();
    let expanded = inline_all(&source, crate::lir::MAX_VALUES).unwrap();
    assert_eq!(expanded.verified_module().unwrap().functions.len(), 1);
    for policy in [Inlining::Always, Inlining::Never] {
        let code =
            Kernel::compile(&root, &crate::test_target::options().with_inlining(policy)).unwrap();
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
    let source = root.verified_module().unwrap();
    let expanded = inline_all(&source, crate::lir::MAX_VALUES).unwrap();
    expanded.verify().unwrap();
    assert_eq!(
        expanded
            .blocks()
            .filter(|&b| matches!(expanded.terminator(b), Terminator::Return(_)))
            .count(),
        1
    );
    assert_eq!(
        Kernel::compile(&root, &crate::test_target::options())
            .unwrap()
            .call(&[0]),
        Ok(0)
    );
}

#[test]
fn expansion_counts_only_the_live_parts_of_reused_templates() {
    let mut forever = Function::scalar(1);
    let spin = forever.create_block();
    forever.jump(forever.entry(), spin, &[]);
    forever.jump(spin, spin, &[]);
    let mut callee = Arc::new(forever);
    for _ in 0..32 {
        let mut caller = Function::scalar(1);
        let entry = caller.entry();
        let input = caller.params(entry)[0];
        let first = caller.add_call(Arc::clone(&callee), &[input]);
        let value = caller.append(entry, Op::Call(first)).unwrap();
        let second = caller.add_call(callee, &[value]);
        let value = caller.append(entry, Op::Call(second)).unwrap();
        caller.set_terminator(entry, Terminator::Return(value));
        callee = Arc::new(caller);
    }
    let source = callee.verified_module().unwrap();
    let expanded = inline_all(&source, crate::lir::MAX_VALUES).unwrap();
    assert_eq!(expanded.blocks().count(), 2);
    assert_eq!(expanded.verified_module().unwrap().functions.len(), 1);
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
    f.append_inst(a, Op::BoundsCheck(i, n)).unwrap();
    f.append_inst(a, Op::BoundsCheck(i, n)).unwrap();
    f.jump(a, end, &[]);
    f.jump(b, end, &[]);
    f.append_inst(end, Op::BoundsCheck(i, n)).unwrap();
    let zero = f.append(end, Op::Const(0)).unwrap();
    f.append_inst(end, Op::Bin(Binary::Div, n, zero)).unwrap();
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
    let kernel = Kernel::compile(&f, &crate::test_target::options()).unwrap();
    for cond in [0, 1] {
        assert_eq!(
            kernel.call(&[cond, 1, 1]),
            Err(CallError::Trap(Trap::Bounds))
        );
        assert_eq!(
            kernel.call(&[cond, 0, 1]),
            Err(CallError::Trap(Trap::DivisionByZero))
        );
    }
}

/// The closed form of `sum(start..bound)` halves the trip count as unsigned:
/// a signed `<` loop runs up to `u32::MAX` times when `start` is negative.
#[test]
fn recurrence_closed_form_holds_for_trip_counts_past_i32_max() {
    let text = "fn main(start:i32 bound:i32) -> i32
  one = const 1
  zero = const 0
  jump head(start zero)
block head(i:i32 acc:i32)
  more = lt i bound
  br more body() | done(acc)
block body()
  next_i = add i one
  next_acc = add acc i
  jump head(next_i next_acc)
block done(out:i32)
  ret out";
    let f = crate::hir::text::parse(text, &[]).unwrap();
    let mut optimized = f.clone();
    simplify(&mut optimized).unwrap();
    assert!(
        optimized
            .blocks()
            .all(|b| !matches!(optimized.terminator(b), Terminator::Branch { .. })),
        "the loop must be replaced by its closed form"
    );
    let kernel = Kernel::compile(&f, &crate::test_target::options()).unwrap();
    for (start, bound) in [
        (i32::MIN, i32::MAX),
        (i32::MIN, 0),
        (-1, i32::MAX),
        (i32::MIN + 1, i32::MAX),
        (-3, 4),
        (5, 5),
        (7, -7),
    ] {
        let trip = (i128::from(bound) - i128::from(start)).max(0);
        let sum = trip * i128::from(start) + trip * (trip - 1) / 2;
        assert_eq!(
            kernel.call(&[start, bound]),
            Ok(sum as i32),
            "sum({start}..{bound})"
        );
    }
}

/// Unswitching clones a loop and abandons the edges of the original; after
/// the pipeline the operand arena holds only the arguments that live edges
/// and calls use.
#[test]
fn optimized_functions_keep_no_retired_operands() {
    let text = "fn main(n:i32 mode:i32) -> i32
  z = const 0
  one = const 1
  jump head(z z)
block head(i:i32 acc:i32)
  more = lt i n
  br more choose() | done()
block choose()
  br mode yes() | no()
block yes()
  total = add acc one
  jump latch(total)
block no()
  other = sub acc one
  jump latch(other)
block latch(sum:i32)
  next = add i one
  jump head(next sum)
block done()
  ret acc";
    let f = crate::hir::text::parse(text, &[]).unwrap();
    let mut optimized = f.clone();
    simplify(&mut optimized).unwrap();
    let (arena, live) = optimized.operand_arena();
    assert_eq!(arena, live);
    assert!(
        optimized.blocks().count() > f.blocks().count(),
        "unswitched"
    );
    let kernel = Kernel::compile(&f, &crate::test_target::options()).unwrap();
    for (n, mode) in [(0, 0), (3, 1), (3, 0), (-2, 1)] {
        let expected = if mode != 0 { n.max(0) } else { -n.max(0) };
        assert_eq!(kernel.call(&[n, mode]), Ok(expected));
    }
}
